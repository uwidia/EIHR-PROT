import csv
import json
import logging
import platform
import random
import sys
from pathlib import Path
from typing import Sequence
import esm
from esm import FastaBatchedDataset
import numpy as np
import torch
from tqdm import tqdm
from reliability_aware.parser import get_dataset_hashes
from reliability_aware.preprocessing import sha256_file
from reliability_aware.shard_handling import BufferState, flush_shard

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "shard_number",
    "local_seq_idx",
    "global_seq_idx",
    "label",
    "sequence_length",
    "was_truncated",
    "truncated_length",
]

def extract_fasta_embeddings(
    fasta_path: str,
    output_dir: str,
    valid_hashes_path: str = "hashlist.txt",
    manifest_filename: str = "manifest",
    model_name: str = "esm2_t33_650M_UR50D",
    toks_per_batch: int = 4096,
    truncation_seq_length: int = 1022,
    repr_layer: int | None = None,
    shard_size: int = 1000,
    use_fp16: bool = True,
    seed: int = 0,
    deterministic: bool = True,
    device: str | None = None,
) -> None:
    """Extract per-residue ESM embeddings from a FASTA file and save them in shards."""
    fasta_path, output_dir, valid_hashes_path = _prepare_paths(
        fasta_path=fasta_path,
        output_dir=output_dir,
        valid_hashes_path=valid_hashes_path,
    )

    _set_reproducibility(seed=seed, deterministic=deterministic)

    device = _resolve_device(device)
    _warn_if_cpu_fp16(device, use_fp16)

    model, alphabet = _load_esm_model(model_name, device=device)
    repr_layer = _resolve_repr_layer(model, repr_layer)

    _warn_if_fasta_hash_differs(fasta_path, valid_hashes_path)

    dataset, batch_sampler, data_loader = _build_data_loader(
        fasta_path=fasta_path,
        alphabet=alphabet,
        toks_per_batch=toks_per_batch,
        truncation_seq_length=truncation_seq_length,
        device=device,
    )

    manifest_path = create_manifest(output_dir, manifest_filename)
    shard_buffer = BufferState()

    running_idx = 0
    local_seq_idx = 0

    with torch.inference_mode():
        progress_bar = tqdm(
            data_loader,
            total=len(batch_sampler),
            desc="Extracting embeddings",
        )

        for batch_idx, (labels, sequences, toks) in enumerate(progress_bar):
            progress_bar.set_postfix(
                batch=f"{batch_idx + 1}/{len(batch_sampler)}",
                seqs=len(labels),
                shard=shard_buffer.shard_id,
            )

            toks = toks.to(device, non_blocking=(device == "cuda"))
            representations = _run_model_on_batch(model, toks, repr_layer, batch_idx)
            if representations is None:
                continue

            local_seq_idx, manifest_rows = _process_batch(
                labels = labels,
                sequences = sequences,
                representations = representations,
                shard_buffer = shard_buffer,
                global_start_idx = running_idx,
                local_seq_idx = local_seq_idx,
                truncation_seq_length = truncation_seq_length,
                shard_size = shard_size,
                output_dir = output_dir,
                use_fp16 = use_fp16,
            )
            append_manifest_rows(manifest_path, manifest_rows)
            running_idx += len(labels)

    flush_shard(shard_buffer, output_dir)
    logger.info("Embedding extraction complete")

    _write_run_metadata(
        output_dir=output_dir,
        model_name=model_name,
        repr_layer=repr_layer,
        truncation_seq_length=truncation_seq_length,
        toks_per_batch=toks_per_batch,
        seed=seed,
        deterministic=deterministic,
        device=device,
        fasta_path=fasta_path,
        dataset=dataset,
    )

def create_manifest(output_dir: Path, manifest_name: str) -> Path:
    """Create the manifest CSV if it does not already exist."""
    manifest_path = output_dir / f"{manifest_name}.csv"

    if not manifest_path.exists():
        with manifest_path.open("w", newline="") as manifest_file:
            writer = csv.writer(manifest_file)
            writer.writerow(MANIFEST_COLUMNS)
        logger.info("Manifest created at %s", manifest_path)

    return manifest_path

def append_manifest_rows(manifest_path: Path, rows: list[list[object]]) -> None:
    """Append a batch of rows to the manifest."""
    if not rows:
        return

    with manifest_path.open("a", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerows(rows)

def _set_reproducibility(seed: int = 0, deterministic: bool = True) -> None:
    """Set seeds and deterministic flags for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True


def _load_esm_model(model_name: str, device: str):
    """Load a pretrained ESM model and freeze it for inference."""
    if not hasattr(esm.pretrained, model_name):
        raise ValueError(f"Unknown model: {model_name}")

    logger.info("Loading model: %s", model_name)
    model_loader = getattr(esm.pretrained, model_name)
    model, alphabet = model_loader()
    model = model.to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, alphabet


def _prepare_paths(
    fasta_path: str,
    output_dir: str,
    valid_hashes_path: str,
) -> tuple[Path, Path, Path]:
    """Resolve input/output paths and create the output directory if needed."""
    fasta_file = Path(fasta_path).resolve()
    output_path = Path(output_dir).resolve()
    hashes_file = Path(valid_hashes_path).resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    return fasta_file, output_path, hashes_file


def _resolve_device(device: str | None) -> str:
    """Choose a compute device if one is not explicitly provided."""
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", resolved)
    return resolved


def _warn_if_cpu_fp16(device: str, use_fp16: bool) -> None:
    """Warn when fp16 is requested on CPU, which is usually very slow."""
    if device == "cpu" and use_fp16:
        logger.warning(
            "fp16 on CPU is valid but can be extremely slow and is unsupported by many CPU ops."
        )


def _resolve_repr_layer(model, repr_layer: int | None) -> int:
    """Use the final model layer unless a layer is explicitly requested."""
    if repr_layer is None:
        return model.num_layers

    if not 0 <= repr_layer <= model.num_layers:
        raise ValueError(
            f"repr_layer must be between 0 and {model.num_layers}, got {repr_layer}."
        )

    return repr_layer


def _warn_if_fasta_hash_differs(fasta_path: Path, valid_hashes_path: Path) -> None:
    """Warn when the FASTA file hash differs from the expected preprocessing hashes."""
    fasta_hash = sha256_file(fasta_path)
    valid_hashes = get_dataset_hashes(valid_hashes_path)

    if fasta_hash not in valid_hashes:
        logger.warning(
            "FASTA file differs from the preprocessed dataset used by the creators. "
            "Downstream results may differ."
        )


def _build_data_loader(
    fasta_path: Path,
    alphabet,
    toks_per_batch: int,
    truncation_seq_length: int,
    device: str,
):
    """Create the dataset, batch sampler, and dataloader for FASTA inference."""
    dataset = FastaBatchedDataset.from_file(fasta_path)
    batch_sampler = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batch_sampler,
        num_workers=0,  # Shard loading cache is not thread-safe.
        pin_memory=(device == "cuda"),
    )

    logger.info("Loaded %d sequences from %s", len(dataset), fasta_path)
    return dataset, batch_sampler, data_loader


def _run_model_on_batch(
    model,
    toks: torch.Tensor,
    repr_layer: int,
    batch_idx: int,
) -> torch.Tensor | None:
    """Run inference for one batch and return per-token representations on CPU."""
    try:
        outputs = model(toks, repr_layers=[repr_layer], return_contacts=False)
    except Exception:
        logger.exception("Model inference failed for batch %d", batch_idx)
        return None

    return outputs["representations"][repr_layer].to("cpu")


def _process_batch(
    labels: Sequence[str],
    sequences: Sequence[str],
    representations: torch.Tensor,
    shard_buffer: BufferState,
    global_start_idx: int,
    local_seq_idx: int,
    truncation_seq_length: int,
    shard_size: int,
    output_dir: Path,
    use_fp16: bool,
) -> tuple[int, list[list[object]]]:
    """Extract sequence embeddings, add them to the shard buffer, and build manifest rows."""
    manifest_rows: list[list[object]] = []

    for batch_offset, label in enumerate(labels):
        sequence = sequences[batch_offset]
        sequence_length = len(sequence)
        truncated_length = min(truncation_seq_length, sequence_length)
        global_seq_idx = global_start_idx + batch_offset

        # Token 0 is <CLS>; residues begin at token 1.
        embedding = representations[batch_offset, 1 : truncated_length + 1].clone()
        if use_fp16:
            embedding = embedding.half()

        shard_buffer.append(embedding, label, sequence_length, truncated_length)
        was_truncated = int(sequence_length > truncation_seq_length)

        manifest_rows.append(
            [
                shard_buffer.shard_id,
                local_seq_idx,
                global_seq_idx,
                label,
                sequence_length,
                was_truncated,
                truncated_length,
            ]
        )
        local_seq_idx += 1

        if len(shard_buffer.buffer_reps) >= shard_size:
            flush_shard(shard_buffer, output_dir)
            local_seq_idx = 0

    return local_seq_idx, manifest_rows


def _write_run_metadata(
    output_dir: Path,
    model_name: str,
    repr_layer: int,
    truncation_seq_length: int,
    toks_per_batch: int,
    seed: int,
    deterministic: bool,
    device: str,
    fasta_path: Path,
    dataset,
) -> None:
    """Write a reproducibility record for the embedding run."""
    metadata = {
        "model_name": model_name,
        "repr_layer": repr_layer,
        "pooling": "no pooling applied",
        "truncation_seq_length": truncation_seq_length,
        "toks_per_batch": toks_per_batch,
        "seed": seed,
        "deterministic": deterministic,
        "device": device,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "esm_version": getattr(esm, "__version__", "unknown"),
        "fasta_path": str(fasta_path.resolve()),
        "fasta_sha256": sha256_file(fasta_path),
        "num_sequences": len(dataset),
    }

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    logger.info("Saved run metadata to %s", metadata_path)
