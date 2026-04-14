from torch.utils.data import Dataset
import torch
import csv
from pathlib import Path

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
        shard_id, local_seq_idx, _ = self.index[idx]
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

    shard_buffer.next_shard()
    shard_buffer.reset()