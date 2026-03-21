import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import esm
from esm import FastaBatchedDataset


class ESMShardDataset(Dataset):
    def __init__(self, shard_dir):
        self.shard_dir = Path(shard_dir)
        self.shard_files = sorted(self.shard_dir.glob("*.pt"))

        assert len(self.shard_files) > 0, "No shard files found"

        # Build index mapping: global_idx -> (shard_id, local_idx)
        self.index = []
        self.shard_sizes = []

        for shard_id, shard_file in enumerate(self.shard_files):
            obj = torch.load(shard_file, map_location="cpu")
            n = len(obj["representations"])
            self.shard_sizes.append(n)

            for i in range(n):
                self.index.append((shard_id, i))

        # Cache (only one shard in memory at a time)
        self._current_shard_id = None
        self._current_shard = None

    def __len__(self):
        return len(self.index)

    def _load_shard(self, shard_id):
        if self._current_shard_id != shard_id:
            shard_file = self.shard_files[shard_id]
            self._current_shard = torch.load(shard_file, map_location="cpu")
            self._current_shard_id = shard_id

    def __getitem__(self, idx):
        shard_id, local_idx = self.index[idx]

        self._load_shard(shard_id)

        rep = self._current_shard["representations"][local_idx]  # (L, 1280)
        label = self._current_shard["labels"][local_idx]

        return rep, label

class BufferState:
    def __init__(self):
        self.shard_id = 0
        self.buffer_reps = []
        self.buffer_labels = []
        self.buffer_seq_lens = []
        self.buffer_trunc_lens = []
    
    def append(self, rep, label, seq_len, trunc_len):
        self.buffer_reps.append(rep)
        self.buffer_labels.append(label)
        self.buffer_seq_lens.append(seq_len)
        self.buffer_trunc_lens.append(trunc_len)

    def next_shard(self):
        self.shard_id += 1
    
    def reset(self):
        self.buffer_reps = []
        self.buffer_labels = []
        self.buffer_seq_lens = []
        self.buffer_trunc_lens = []


def set_reproducibility(seed: int = 0, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Helps with reproducibility on the same machine / software stack. Note that determinism isn't guaranteed if your hardware and package versions are different

    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True


def sha256_file(path: Path, chunk_size: int = 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_model(model_name: str, device: str):
    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, alphabet


def flush_shard(shard_buffer: BufferState, output_dir: Path):
    if len(shard_buffer.buffer_reps) == 0:
        return

    output_file = output_dir / f"part_{shard_buffer.shard_id:04d}.pt"

    torch.save(
        {
            "representations": shard_buffer.buffer_reps,
            "labels": shard_buffer.buffer_labels,
            "seq_lengths": shard_buffer.buffer_seq_lens
            "trunc_lengths": shard_buffer.buffer_trunc_lens
        }, 
        output_file,
    )

    print(f"Saved {output_file} ({len(shard_buffer.buffer_reps)} proteins)")

    shard_buffer.next_shard()
    shard_buffer.reset()


def extract_fasta_embeddings(
    fasta_path: Path,
    output_dir: str,
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_reproducibility(seed=seed, deterministic=deterministic)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, alphabet = load_model(model_name, device=device)

    if repr_layer is None:
        repr_layer = model.num_layers

    # Batching and data loading 
    dataset = FastaBatchedDataset.from_file(fasta_path)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
        num_workers=0, 
        # pin_memory=(device == "cuda"),
    )

    print(f"Loaded {len(dataset)} sequences from {fasta_path}")

    shard_buffer = BufferState()


    # CSV file for quick look ups for shard_number or other info across split
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f_manifest:
        writer = csv.writer(f_manifest)
        writer.writerow(
            [
                "shard_number",
                "index",
                "label",
                "sequence_length",
                "was_truncated",
                "truncated_length",
                "embedding_file",
            ]
        )

        #Extract embeddings for each batch
        with torch.inference_mode():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(
                    f"Batch {batch_idx+1}/{len(batches)} "
                    f"({len(labels)} sequences)"
                )
                
                if device == "cuda":
                    toks = toks.to(device, non_blocking=True)
                else:
                    toks = toks.to(device)

                out = model(
                    toks,
                    repr_layers=[repr_layer],
                    return_contacts=False,
                )
                reps = out["representations"][repr_layer].to("cpu")

                for i, label in enumerate(labels):
                    seq_len = len(strs[i])
                    trunc_len = min(truncation_seq_length, seq_len)

                    # residue tokens are positions 1 : trunc_len + 1. Position 0 is the <CLS> token. 
                    emb = reps[i, 1 : trunc_len + 1].clone()
                    
                    if use_fp16:
                        emb = emb.half()

                    shard_buffer.append(emb, label, seq_len, trunc_len)
                    if len(shard_buffer.buffer_reps) >= shard_size:
                        flush_shard(shard_buffer, output_dir)

                    seq_idx = sum(len(b) for b in batches[:batch_idx]) + i
                    was_truncated = 1 if seq_len > truncation_seq_length else 0
                    file_name = f"{seq_idx:08d}_{label}.pt"
                    file_path = output_dir / file_name

                    torch.save(
                        {
                            "label": label,
                            "sequence_length": seq_len,
                            "truncated_length": trunc_len,
                            "model_name": model_name,
                            "repr_layer": repr_layer,
                            "embedding": emb,
                        },
                        file_path,
                    )

                    writer.writerow(
                        [shard_buffer.shard_id,seq_idx, label, seq_len, was_truncated, trunc_len, file_name]
                    )

                print(
                    f"Processed batch {batch_idx + 1}/{len(batches)} "
                    f"({len(labels)} sequences)"
                )

        flush_shard(shard_buffer, output_dir)
        print("Done!")
    
    metadata = {
        "model_name": model_name,
        "repr_layer": repr_layer,
        "pooling": "mean_over_residue_tokens_only",
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

#Custom collate_fn for dataloader for the ESMShardDataset object. It makes all sequences equal length with the longest sequence by padding
def collate_fn(batch):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--toks_per_batch", type=int, default=4096)
    parser.add_argument("--truncation_seq_length", type=int, default=1022)
    parser.add_argument("--repr_layer", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    extract_fasta_embeddings(
        fasta_path=args.fasta,
        output_dir=args.outdir,
        model_name=args.model,
        toks_per_batch=args.toks_per_batch,
        truncation_seq_length=args.truncation_seq_length,
        repr_layer=args.repr_layer,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
    )


if __name__ == "__main__":
    main()










