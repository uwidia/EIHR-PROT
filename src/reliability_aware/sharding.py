from torch.utils.data import Dataset
import torch
import csv
from pathlib import Path
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ESMShardDataset(Dataset):
    """
    Assumes manifest contains:
    - shard_number
    - local_seq_idx
    - global_seq_idx
    - sequence_length
    """

    def __init__(self, shard_dir: str, manifest_path: str, cache_size: int = 3):
        self.shard_dir = Path(shard_dir).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.cache_size = cache_size

        if not self.shard_dir.exists() or not self.manifest_path.exists():
            raise FileNotFoundError("Shard dir or manifest path does not exist")

        self.shard_files = {
            int(f.stem.split("_")[-1]): f
            for f in sorted(self.shard_dir.glob("*.pt"))
        }
        if not self.shard_files:
            raise FileNotFoundError(f"No .pt shard files found in {self.shard_dir}")

        self.index = []               # dataset idx -> (shard_id, local_idx, global_idx)
        self.lengths = []             # dataset idx -> sequence_length
        self.indices_by_shard = defaultdict(list)

        with open(self.manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for dataset_idx, row in enumerate(reader):
                shard_id = int(row["shard_number"])
                local_idx = int(row["local_seq_idx"])
                global_idx = int(row["global_seq_idx"])
                seq_len = int(row["sequence_length"])

                self.index.append((shard_id, local_idx, global_idx))
                self.lengths.append(seq_len)
                self.indices_by_shard[shard_id].append(dataset_idx)

        self._cache = OrderedDict()
        self._validated_shards = set()

    def __len__(self):
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

        if len(shard["seq_lengths"]) != len(reps):
            raise ValueError(f"{shard_file}: mismatch seq_lengths vs representations length")

        if len(shard["trunc_lengths"]) != len(reps):
            raise ValueError(f"{shard_file}: mismatch trunc_lengths vs representations length")

        for i, r in enumerate(reps[:5]):  # sample first 5 only
            if not isinstance(r, torch.Tensor):
                raise TypeError(f"{shard_file}: rep[{i}] is not tensor")

            if r.ndim != 2:
                raise ValueError(f"{shard_file}: rep[{i}] is not 2D")

            if r.shape[1] != 1280:
                raise ValueError(f"{shard_file}: rep[{i}] wrong embedding dim")

    def _load_shard(self, shard_id: int):
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]

        shard_file = self.shard_files[shard_id]
        shard = torch.load(shard_file, map_location="cpu")

        if shard_id not in self._validated_shards:
            self._validate_shard(shard, shard_file)
            self._validated_shards.add(shard_id)

        self._cache[shard_id] = shard
        self._cache.move_to_end(shard_id)

        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return shard

    def __getitem__(self, idx: int):
        shard_id, local_idx, global_idx = self.index[idx]
        shard = self._load_shard(shard_id)
        rep = shard["representations"][local_idx]  # (L, 1280)
        return rep, global_idx

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

    