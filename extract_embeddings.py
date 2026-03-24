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
import logging
from preprocessing import sha256_file
from parser import get_dataset_hashes

setup_logging()
logger = logging.getLogger(__name__)

class ESMShardDataset(Dataset):
    """
    Dataset for loading embeddings from sharded .pt files using a manifest index.
    Handles shard loading, validation, and caching of a single shard at a time.
    """

    def __init__(self, shard_dir: str, manifest_path: str):

        """
        Initialize dataset from shard directory and manifest file.

        Args:
            shard_dir (str): Path to directory containing .pt shard files.
            manifest_path (str): Path to CSV file mapping global to shard indices.

        Raises:
            FileNotFoundError: If shard directory or manifest is missing.

        """
        self.shard_dir = Path(shard_dir).resolve()
        self.manifest_path = Path(manifest_path).resolve()

        if not self.shard_dir.exists() or not self.manifest_path.exists():
            raise FileNotFoundError("Directory path for shard files or manifest doesn't exist")
            

        self.shard_files = {f.stem: f for f in sorted(self.shard_dir.glob("*.pt"))}

        if len(self.shard_files) == 0:
            raise FileNotFoundError(f"No .pt shard files found in {self.shard_dir}")

        # Build index mapping: global_seq_idx -> (shard_id, local_seq_idx)
        self.index = []
        with open(self.manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shard_id = int(row["shard_number"])
                local_seq_idx = int(row["local_seq_idx"])
                global_seq_idx = int(row["global_seq_idx"])
                self.index.append((shard_id, local_seq_idx, global_seq_idx))

        # Cache (only one shard in memory at a time)
        self._current_shard_id = None
        self._current_shard = None

    def __len__(self):
        """
        Return:
            Total number of sequences in shard directory
        """
        return len(self.index)
    
    def _validate_shard(self, shard: dict, shard_file: Path):
         """
        Validate structure and basic integrity of a shard file.
        Performs lightweight checks on keys, types, and tensor shapes.
        """

        required_keys = [
            "representations",
            "labels",
            "seq_lengths",
            "trunc_lengths",
        ]

        for key in required_keys:
            if key not in shard:
                raise ValueError(f"{shard_file} missing key: {key}")

        reps = shard["representations"]
        labels = shard["labels"]

        if not isinstance(reps, list) or not isinstance(labels, list):
            raise TypeError(f"{shard_file}: representations/labels must be lists")

        if len(reps) != len(labels):
            raise ValueError(f"{shard_file}: mismatch reps vs labels length")

        # Optional deeper validation (can be expensive)
        for i, r in enumerate(reps[:5]):  # sample first 5 only
            if not isinstance(r, torch.Tensor):
                raise TypeError(f"{shard_file}: rep[{i}] is not tensor")

            if r.ndim != 2:
                raise ValueError(f"{shard_file}: rep[{i}] is not 2D")

            if r.shape[1] != 1280:
                raise ValueError(f"{shard_file}: rep[{i}] wrong embedding dim")

    def _load_shard(self, shard_id: int):
        """
        Load shard into memory if not already cached
        """
        if self._current_shard_id != shard_id:
            shard_file_name = f"part_{shard_id:04d}"
            if shard_file_name not in self.shard_files:
                raise FileNotFoundError(f"Shard file {shard_file_name}.pt not found — it may have been deleted")
            shard_file = self.shard_files[shard_file_name]

            try:
                shard = torch.load(shard_file, map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"Failed to load shard {shard_file}: {e}")
                
            self._validate_shard(shard, shard_file)
            self._current_shard = shard
            self._current_shard_id = shard_id

    def __getitem__(self, idx: int):
        """
        Retrieves representation and label name for sequence based on global index.
        Args:
            idx (int): Global sequence index. 
        Return:
            tuple (representation tensor (L, 1280), sequence label)
        """
        shard_id, local_seq_idx, global_seq_idx = self.index[idx]
        self._load_shard(shard_id)
        rep = self._current_shard["representations"][local_seq_idx]  # (L, 1280)
        label = self._current_shard["labels"][local_seq_idx]

        return rep, label

class BufferState:
    """
    Saves essential information for a single shard at a time.
    Provides reset buffer logic and shard_id tracking.  
    """
    def __init__(self):
        
        self.shard_id = 0
        self.buffer_reps = []
        self.buffer_labels = []
        self.buffer_seq_lens = []
        self.buffer_trunc_lens = []
    
    def append(self, rep: torch.Tensor, label: str, seq_len: int, trunc_len: int):
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

def setup_logging():
    """
    Configuration settings for logger
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        filename= "log_embeddings.log",
        filemode = "a"
    )

def set_reproducibility(seed: int = 0, deterministic: bool = True):
    """
    Supports reproducibility by setting seeds and ensuring determinism.
    Args:
        seed (int): Fixed seed for python, numpy, and pytorch RNG.
        deterministic (bool): Ensures determinism on similar machine/software stacks
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    #Note that determinism isn't guaranteed if your hardware and package versions are different

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


def flush_shard(shard_buffer: BufferState, output_dir: Path):
    """
    Resets shard buffer values, saves values in a .pt file, and iterates to the next shard.
    Args:
        shard_buffer(BufferState): Selected shard buffer
        output_dir(Path): Directory to save .pt file
    """
    if len(shard_buffer.buffer_reps) == 0:
        return #if buffer is empty, return immediately

    output_file = output_dir / f"part_{shard_buffer.shard_id:04d}.pt"

    torch.save(
        {
            "representations": shard_buffer.buffer_reps,
            "labels": shard_buffer.buffer_labels,
            "seq_lengths": shard_buffer.buffer_seq_lens,
            "trunc_lengths": shard_buffer.buffer_trunc_lens
        }, 
        output_file,
    )

    logger.info(f"Saved {output_file} ({len(shard_buffer.buffer_reps)} proteins). Flushing shard...")
    

    shard_buffer.next_shard()
    shard_buffer.reset()

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


def extract_fasta_embeddings(
    fasta_path: str,
    output_dir: str,
    valid_hashes_path: str = "hashlist.txt",
    model_name: str = "esm2_t33_650M_UR50D",
    toks_per_batch: int = 4096,
    truncation_seq_length: int = 1022,
    repr_layer: int | None = None,
    shard_size: int = 1000,
    use_fp16: bool = True,
    seed: int = 0,
    deterministic: bool = True,
    device: str | None = None
    ):

    """
    Extracts sequence information from fasta file, batches and loads sequences to esm model.
    Runs inference and extracts per token embeddings for each sequence.
    Saves per sequence index mapping and other information to a manifest.
    Saves sequence representations across multiple shards based on specified shard length

    Args:
        fasta_path(str): Path to fasta file
        output_dir: Directory for saving shards
        valid_hashes_path (str): Path to valid hashes for original preprocessed fasta file
        model_name (str): Selected esm model
        toks_per_batch (int): Max tokens permitted for each batch
        truncation_seq_length (int): Length limit for sequences
        repr_layer (int): Model layer for extracting sequence representations
        shard_size (int): Max number of sequences per shard
        use_fp16 (bool): Convert sequence representations to float16
        seed (int): 
        deterministic (bool):
        device (str): Device type (i.e cuda or cpu)
    """

    fasta_path = Path(fasta_path).resolve()
    filename = f"{fasta_path.stem}.csv"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_hashes_path= Path(valid_hashes_path).resolve()

    set_reproducibility(seed=seed, deterministic=deterministic)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to {device}. {device} will be used to load model")

    if device == "cpu" and use_fp16:
        logger.warning("""Warning: You're currently casting to fp16 using a CPU.
        This action is valid but can be extremely slow and is unsupported by many CPU ops.""")

    model, alphabet = load_model(model_name, device=device)

    if repr_layer is None:
        repr_layer = model.num_layers

    #Validate fasta file similarity with preprocessing output (for reproducibility)
    batching_dataset_hash = sha256_file(fasta_path)
    valid_hashes = get_dataset_hashes(valid_hashes_path)
    if batching_dataset_hash not in valid_hashes:
        logger.warning("""Fasta file used for batching and loading is different from the preprocessed dataset used by the creators. 
        Downstream results may be different""")

    # Batching and data loading 
    dataset = FastaBatchedDataset.from_file(fasta_path)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
        num_workers=0, #ESMShardDataset._load_shard cache is not thread-safe; keep at 0
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Loaded {len(dataset)} sequences from {fasta_path}")

    shard_buffer = BufferState()
    create_manifest(output_dir, filename)

    #Extract embeddings for each batch
    with torch.inference_mode():
        running_idx = 0
        local_seq_idx = 0
        manifest_list = []
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            logger.info(
                f"Batch {batch_idx+1}/{len(batches)} "
                f"({len(labels)} sequences)"
            )
            
            toks = toks.to(device, non_blocking=(device == "cuda"))
        
            try: 
                out = model(
                    toks,
                    repr_layers=[repr_layer],
                    return_contacts=False,
                )
            except Exception as e:
                logger.error(f"There was an error loading the model during batch {batch_idx}. Error: {e}")
                continue
    
            reps = out["representations"][repr_layer].to("cpu")

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
                    shard_buffer.shard_id, local_seq_idx, global_seq_idx, label, seq_len, was_truncated, trunc_len]
                    )
                local_seq_idx += 1
                if len(shard_buffer.buffer_reps) >= shard_size:
                    flush_shard(shard_buffer, output_dir)
                    local_seq_idx = 0

        
            running_idx += len(labels)

            update_manifest(output_dir, filename, manifest_list)
            logger.info(f"Manifest update for batch {batch_idx} completed successfully")
            manifest_list = []
            logger.info(
                f"Processed batch {batch_idx + 1}/{len(batches)} ({len(labels)} sequences)"
            )

    flush_shard(shard_buffer, output_dir)
    logger.info("Embedding extraction complete!")

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
    logger.info(f"run_metadata.json saved to {output_dir}/run_metadata.json")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--valid_hashes", type=str, required=True)
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
        valid_hashes_path = args.valid_hashes,
        model_name=args.model,
        toks_per_batch=args.toks_per_batch,
        truncation_seq_length=args.truncation_seq_length,
        repr_layer=args.repr_layer,
        shard_size = args.shard_size,
        use_fp16 = args.use_fp16,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
    )


if __name__ == "__main__":
    main()




"""
Suggestions for later:
Do tqdm
Set up logic for resumption of embedding download
handle possibility of failed/broken downloads or failed batching
Add: A --dry_run flag that validates the FASTA file, prints batch sizes, and estimates GPU memory usage without actually running inference. 
It's a small addition but shows you think about usability, which is exactly what entry-level ML engineering roles look for.
"""






