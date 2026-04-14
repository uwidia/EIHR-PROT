import csv
import json
import platform
import random
import sys
from pathlib import Path
import numpy as np
import torch
import esm
from esm import FastaBatchedDataset
import logging
from reliability_aware.utils import setup_logging
from reliability_aware.preprocessing import sha256_file
from reliability_aware.parser import get_dataset_hashes
from tqdm import tqdm
from reliability_aware.sharding import BufferState, flush_shard

setup_logging()
logger = logging.getLogger(__name__)


def set_reproducibility(seed: int = 0, deterministic: bool = True):
    """
    Supports reproducibility by setting seeds and ensuring determinism.
    Args:
        seed (int): Controls randomness and supports reproducibility
        deterministic (bool): Ensures reproducibility on similar machine/software stacks
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True


def load_model(model_name: str, device: str):
    """
    Loads specified pretrained esm_model
    Args:
        model_name(str): Esm model name
        device (str): Device type
    Returns:
        tuple(model, alphabet (tokenizer))
    """
    if not hasattr(esm.pretrained, model_name):
        raise ValueError(f"Unknown model: {model_name}")
    loader = getattr(esm.pretrained, model_name)
    logger.info(f"Loading model: {model_name}")
    model, alphabet = loader()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, alphabet


def create_manifest(output_dir: Path, filename: str):
    """
    Creates manifest (csv) for tracking essential information for all sequences in shard directory 
    """
    manifest_path = output_dir / f"{filename}.csv"

    if not manifest_path.exists():
        with manifest_path.open("w", newline="") as f_manifest:
            writer = csv.writer(f_manifest)
            writer.writerow(
                [
                    "shard_number",
                    "local_seq_idx",
                    "global_seq_idx",
                    "label",
                    "sequence_length",
                    "was_truncated",
                    "truncated_length",
                ]
            )
        logger.info(f"Manifest created. Saved to {manifest_path}")

def update_manifest( output_dir: Path, filename: str, manifest_list: list[list]):
    """
    Appends new sequence information to manifest csv
    Args:
        output_dir (Path): Directory path for manifest
        filename (str): manifest file name 
        manifest_list (List[List]): Contains shard_no, local index, global index, label name, seq length, 
        and truncation info for multiple sequences
    """
    manifest_path = output_dir / f"{filename}.csv"
    with manifest_path.open("a", newline="") as f_manifest:
        writer = csv.writer(f_manifest)
        for entry in manifest_list:
            writer.writerow(entry)

def _prepare_paths(
    fasta_path: str,
    output_dir: str,
    valid_hashes_path: str,
    manifest_filename: str,
) -> tuple[Path, Path, Path, str]:
    fasta_path = Path(fasta_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_hashes_path = Path(valid_hashes_path).resolve()
    filename = f"{manifest_filename}"
    return fasta_path, output_dir, valid_hashes_path, filename


def _resolve_device(device: str | None) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to {device}. {device} will be used to load model")
    return device


def _warn_if_cpu_fp16(device: str, use_fp16: bool) -> None:
    if device == "cpu" and use_fp16:
        logger.warning("""Warning: You're currently casting to fp16 using a CPU.
        This action is valid but can be extremely slow and is unsupported by many CPU ops.""")


def _resolve_repr_layer(model, repr_layer: int | None) -> int:
    if repr_layer is None:
        return model.num_layers
    return repr_layer


def _validate_fasta_hash(fasta_path: Path, valid_hashes_path: Path) -> None:
    batching_dataset_hash = sha256_file(fasta_path)
    valid_hashes = get_dataset_hashes(valid_hashes_path)
    if batching_dataset_hash not in valid_hashes:
        logger.warning("""Fasta file used for batching and loading is different from the preprocessed dataset used by the creators. 
        Downstream results may be different""")


def _build_data_loader(
    fasta_path: Path,
    alphabet,
    toks_per_batch: int,
    truncation_seq_length: int,
    device: str,
):
    dataset = FastaBatchedDataset.from_file(fasta_path)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
        num_workers=0,  # ESMShardDataset._load_shard cache is not thread-safe; keep at 0
        pin_memory=(device == "cuda"),
    )
    logger.info(f"Loaded {len(dataset)} sequences from {fasta_path}")
    return dataset, batches, data_loader


def _run_model_on_batch(model, toks, repr_layer: int, batch_idx: int):
    try:
        out = model(
            toks,
            repr_layers=[repr_layer],
            return_contacts=False,
        )
    except Exception as e:
        logger.error(f"There was an error loading the model during batch {batch_idx}. Error: {e}")
        return None

    return out["representations"][repr_layer].to("cpu")


def _process_batch_sequences(
    labels,
    strs,
    reps,
    shard_buffer,
    manifest_list: list,
    running_idx: int,
    local_seq_idx: int,
    truncation_seq_length: int,
    shard_size: int,
    output_dir: Path,
    use_fp16: bool,
) -> int:
    for i, label in enumerate(labels):
        global_seq_idx = running_idx + i
        seq_len = len(strs[i])
        trunc_len = min(truncation_seq_length, seq_len)

        # residue tokens are positions 1 : trunc_len + 1. Position 0 is the <CLS> token.
        emb = reps[i, 1 : trunc_len + 1].clone()

        if use_fp16:
            emb = emb.half()

        shard_buffer.append(emb, label, seq_len, trunc_len)
        was_truncated = 1 if seq_len > truncation_seq_length else 0

        manifest_list.append([
            shard_buffer.shard_id,
            local_seq_idx,
            global_seq_idx,
            label,
            seq_len,
            was_truncated,
            trunc_len,
        ])
        local_seq_idx += 1

        if len(shard_buffer.buffer_reps) >= shard_size:
            flush_shard(shard_buffer, output_dir)
            local_seq_idx = 0

    return local_seq_idx


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
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    logger.info(f"run_metadata.json saved to {output_dir}/run_metadata.json")


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
):
    """
    Extract embeddings from sequences in a FASTA file using an ESM model.

    Batches sequences, runs model inference, extracts embeddings, and saves
    them in shards with an accompanying manifest. Also records run metadata
    for reproducibility.

    repr_layer: defaults to the final layer if None.
    use_fp16: valid on CPU but very slow; prefer GPU if casting to fp16.
    """
    fasta_path, output_dir, valid_hashes_path, filename = _prepare_paths(
        fasta_path=fasta_path,
        output_dir=output_dir,
        valid_hashes_path=valid_hashes_path,
        manifest_filename=manifest_filename,
    )

    set_reproducibility(seed=seed, deterministic=deterministic)

    device = _resolve_device(device)
    _warn_if_cpu_fp16(device, use_fp16)

    model, alphabet = load_model(model_name, device=device)
    repr_layer = _resolve_repr_layer(model, repr_layer)

    # Validate fasta file similarity with preprocessing output (for reproducibility)
    _validate_fasta_hash(fasta_path, valid_hashes_path)

    # Batching and data loading
    dataset, batches, data_loader = _build_data_loader(
        fasta_path=fasta_path,
        alphabet=alphabet,
        toks_per_batch=toks_per_batch,
        truncation_seq_length=truncation_seq_length,
        device=device,
    )

    shard_buffer = BufferState()
    create_manifest(output_dir, filename)

    # Extract embeddings for each batch
    with torch.inference_mode():
        running_idx = 0
        local_seq_idx = 0
        manifest_list = []
        pbar = tqdm(data_loader, total=len(batches), desc="Extracting embeddings")

        for batch_idx, (labels, strs, toks) in enumerate(pbar):
            pbar.set_postfix(
                batch=f"{batch_idx + 1}/{len(batches)}",
                seqs=len(labels),
                shard=shard_buffer.shard_id,
            )

            toks = toks.to(device, non_blocking=(device == "cuda"))
            reps = _run_model_on_batch(model, toks, repr_layer, batch_idx)

            if reps is None:
                continue

            local_seq_idx = _process_batch_sequences(
                labels=labels,
                strs=strs,
                reps=reps,
                shard_buffer=shard_buffer,
                manifest_list=manifest_list,
                running_idx=running_idx,
                local_seq_idx=local_seq_idx,
                truncation_seq_length=truncation_seq_length,
                shard_size=shard_size,
                output_dir=output_dir,
                use_fp16=use_fp16,
            )

            running_idx += len(labels)
            update_manifest(output_dir, filename, manifest_list)

    flush_shard(shard_buffer, output_dir)
    logger.info("Embedding extraction complete!")

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

#Custom collate_fn for dataloader for the ESMShardDataset object. It makes all sequences equal length with the longest sequence by padding
def collate_fn(batch: list[list]):
    reps = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    lengths = [r.shape[0] for r in reps]
    max_len = max(lengths)
    dim = reps[0].shape[1]

    padded = torch.zeros(len(reps), max_len, dim)
    mask = torch.zeros(len(reps), max_len, dtype=torch.bool)

    for i, r in enumerate(reps):
        L = r.shape[0]
        padded[i, :L] = r
        mask[i, :L] = 1

    return padded, mask, labels











