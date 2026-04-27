from torch.utils.data import Dataset
from torch.utils.data import Dataset
import torch
import csv
from pathlib import Path
import csv
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset
from collections import OrderedDict, defaultdict
import copy

def filter_index_by_keep_ids(index, lengths, keep_ids):
    """
    Filter dataset index to only labels in keep_ids.
    Preserves shard_id/local_idx so shard alignment is not broken.
    """
    if keep_ids is None:
        indices_by_shard = defaultdict(list)
        for dataset_idx, (shard_id, *_rest) in enumerate(index):
            indices_by_shard[shard_id].append(dataset_idx)
        return index, lengths, indices_by_shard

    keep_ids = set(keep_ids)

    new_index = []
    new_lengths = []
    new_indices_by_shard = defaultdict(list)

    for old_idx, (record, seq_len) in enumerate(zip(index, lengths)):
        shard_id, local_idx, global_idx, label = record

        if label not in keep_ids:
            continue

        new_dataset_idx = len(new_index)
        new_index.append(record)
        new_lengths.append(seq_len)
        new_indices_by_shard[shard_id].append(new_dataset_idx)

    return new_index, new_lengths, new_indices_by_shard

class ESMGraphHomologyShardDataset(Dataset):
    def __init__(
        self,
        esm_shard_dir: str,
        graph_shard_dir: str,
        homology_shard_dir: str,
        manifest_path: str,
        keep_ids = None,
        require_graph: bool = True,
    ):
        self.seq_graph_ds = ESMGraphShardDataset(
            esm_shard_dir=esm_shard_dir,
            graph_shard_dir=graph_shard_dir,
            manifest_path=manifest_path,
            require_graph=require_graph,
            keep_ids = keep_ids,
        )
        self.homology_ds = HomologyShardDataset(
            homology_shard_dir=homology_shard_dir,
            manifest_path=manifest_path,
            keep_ids = keep_ids,
        )

        if len(self.seq_graph_ds) != len(self.homology_ds):
            raise ValueError("Dataset length mismatch between seq+graph and homology datasets.")
        
        self.lengths = self.seq_graph_ds.lengths
        self.indices_by_shard = self.seq_graph_ds.indices_by_shard

    def __len__(self):
        return len(self.seq_graph_ds)

    def __getitem__(self, idx: int):
        a = self.seq_graph_ds[idx]
        h = self.homology_ds[idx]

        if a["global_idx"] != h["global_idx"] or a["label"] != h["label"]:
            raise ValueError(
                f"Alignment mismatch at idx={idx}: "
                f"seq_graph=({a['global_idx']}, {a['label']}), "
                f"homology=({h['global_idx']}, {h['label']})"
            )

        return {
            "rep": a["rep"],
            "graph": a["graph"],
            "label": a["label"],
            "global_idx": a["global_idx"],
            "homology_prior": h["prior"].float(),
            "homology_gate": h["homology_gate"].float(),
        }

class ESMGraphShardDataset(Dataset):
    """
    Multimodal dataset for aligned ESM + graph shards.

    Assumes:
      - ESM manifest contains:
          shard_number, local_seq_idx, global_seq_idx, label, sequence_length
      - ESM shard path format:
          part_{shard_id:04d}.pt
      - Graph shard path format:
          graph_shard_{shard_id:04d}.pt
      - Graph shard content:
          {"graphs": [graph0, graph1, ...]}
        where graphs[i] aligns with ESM local_seq_idx=i
    """

    def __init__(
        self,
        esm_shard_dir: str,
        graph_shard_dir: str,
        manifest_path: str,
        esm_cache_size: int = 2,
        graph_cache_size: int = 4,
        require_graph: bool = True,
        keep_ids = None,
    ):
        self.esm_shard_dir = Path(esm_shard_dir).resolve()
        self.graph_shard_dir = Path(graph_shard_dir).resolve()
        self.manifest_path = Path(manifest_path).resolve()

        self.esm_cache_size = esm_cache_size
        self.graph_cache_size = graph_cache_size
        self.require_graph = require_graph

        if not self.esm_shard_dir.exists():
            raise FileNotFoundError(f"ESM shard dir not found: {self.esm_shard_dir}")
        if not self.graph_shard_dir.exists():
            raise FileNotFoundError(f"Graph shard dir not found: {self.graph_shard_dir}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.esm_shard_files = {
            int(f.stem.split("_")[-1]): f
            for f in sorted(self.esm_shard_dir.glob("*.pt"))
        }
        self.graph_shard_files = {
            int(f.stem.split("_")[-1]): f
            for f in sorted(self.graph_shard_dir.glob("graph_shard_*.pt"))
        }

        if not self.esm_shard_files:
            raise FileNotFoundError(f"No ESM shards found in {self.esm_shard_dir}")
        if not self.graph_shard_files:
            raise FileNotFoundError(f"No graph shards found in {self.graph_shard_dir}")

        # dataset idx -> metadata
        self.index = []
        self.lengths = []
        self.indices_by_shard = defaultdict(list)


        with open(self.manifest_path, "r") as f:
            reader = csv.DictReader(f)
            required = {"shard_number", "local_seq_idx", "global_seq_idx", "label", "sequence_length"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

            for dataset_idx, row in enumerate(reader):
                shard_id = int(row["shard_number"])
                local_idx = int(row["local_seq_idx"])
                global_idx = int(row["global_seq_idx"])
                label = row["label"]
                seq_len = int(row["sequence_length"])

                self.index.append((shard_id, local_idx, global_idx, label))
                self.lengths.append(seq_len)
                self.indices_by_shard[shard_id].append(dataset_idx)
        self.index, self.lengths, self.indices_by_shard = filter_index_by_keep_ids(
            self.index,
            self.lengths,
            keep_ids,
            )
        self._esm_cache = OrderedDict()
        self._graph_cache = OrderedDict()
        self._validated_esm_shards = set()
        self._validated_graph_shards = set()
    def __len__(self):
        return len(self.index)

    def _validate_esm_shard(self, shard: dict, shard_file: Path):
        required_keys = ["representations", "labels", "seq_lengths", "trunc_lengths"]
        for key in required_keys:
            if key not in shard:
                raise ValueError(f"{shard_file} missing key: {key}")

        reps = shard["representations"]
        labels = shard["labels"]

        if not isinstance(reps, list) or not isinstance(labels, list):
            raise TypeError(f"{shard_file}: representations/labels must be lists")

        if len(reps) != len(labels):
            raise ValueError(f"{shard_file}: mismatch reps vs labels length")

        for i, r in enumerate(reps[:5]):
            if not isinstance(r, torch.Tensor):
                raise TypeError(f"{shard_file}: rep[{i}] is not tensor")
            if r.ndim != 2:
                raise ValueError(f"{shard_file}: rep[{i}] is not 2D")
            if r.shape[1] != 1280:
                raise ValueError(f"{shard_file}: rep[{i}] wrong embedding dim")

    def _validate_graph_shard(self, shard: dict, shard_file: Path):
        if "graphs" not in shard:
            raise ValueError(f"{shard_file} missing key: graphs")
        if not isinstance(shard["graphs"], list):
            raise TypeError(f"{shard_file}: graphs must be a list")

        # sample-check non-None graphs
        checked = 0
        for g in shard["graphs"]:
            if g is None:
                continue
            if not isinstance(g, dict):
                raise TypeError(f"{shard_file}: graph entry must be dict or None")
            required_keys = {
                "coords", "edge_index", "edge_attr", "edge_weight",
                "has_structure", "confidence_proxy",
                "coverage", "mean_confidence", "std_confidence",
                "resolution",
            }
            missing = required_keys - set(g.keys())
            if missing:
                raise ValueError(f"{shard_file}: graph missing keys {sorted(missing)}")
            checked += 1
            if checked >= 3:
                break
    def _load_esm_shard(self, shard_id: int):
        if shard_id in self._esm_cache:
            self._esm_cache.move_to_end(shard_id)
            return self._esm_cache[shard_id]

        if shard_id not in self.esm_shard_files:
            raise FileNotFoundError(f"Missing ESM shard for shard_id={shard_id}")

        shard_file = self.esm_shard_files[shard_id]
        shard = torch.load(shard_file, map_location="cpu")

        if shard_id not in self._validated_esm_shards:
            self._validate_esm_shard(shard, shard_file)
            self._validated_esm_shards.add(shard_id)

        self._esm_cache[shard_id] = shard
        self._esm_cache.move_to_end(shard_id)

        if len(self._esm_cache) > self.esm_cache_size:
            self._esm_cache.popitem(last=False)

        return shard

    def _load_graph_shard(self, shard_id: int):
        if shard_id in self._graph_cache:
            self._graph_cache.move_to_end(shard_id)
            return self._graph_cache[shard_id]

        if shard_id not in self.graph_shard_files:
            raise FileNotFoundError(f"Missing graph shard for shard_id={shard_id}")

        shard_file = self.graph_shard_files[shard_id]
        shard = torch.load(shard_file, map_location="cpu")

        if shard_id not in self._validated_graph_shards:
            self._validate_graph_shard(shard, shard_file)
            self._validated_graph_shards.add(shard_id)

        self._graph_cache[shard_id] = shard
        self._graph_cache.move_to_end(shard_id)

        if len(self._graph_cache) > self.graph_cache_size:
            self._graph_cache.popitem(last=False)

        return shard
    def __getitem__(self, idx: int):
        shard_id, local_idx, global_idx, label = self.index[idx]

        esm_shard = self._load_esm_shard(shard_id)
        graph_shard = self._load_graph_shard(shard_id)

        reps = esm_shard["representations"]
        esm_labels = esm_shard["labels"]
        graphs = graph_shard["graphs"]

        if local_idx >= len(reps):
            raise IndexError(f"local_idx={local_idx} out of bounds for ESM shard {shard_id}")
        if local_idx >= len(graphs):
            raise IndexError(f"local_idx={local_idx} out of bounds for graph shard {shard_id}")

        rep = reps[local_idx]                  # (L, 1280)
        graph = graphs[local_idx]              # dict or None
        esm_label = esm_labels[local_idx]

        # Ensure manifest label matches ESM shard label
        if esm_label != label:
            raise ValueError(
                f"Label mismatch at dataset idx={idx}: manifest={label}, esm_shard={esm_label}"
            )

        if graph is None and self.require_graph:
            raise ValueError(
                f"Graph missing for idx={idx}, label={label}, shard={shard_id}, local_idx={local_idx}"
            )

        # Avoid in-place mutations
        graph = None if graph is None else copy.deepcopy(graph)

        return {
            "rep": rep,
            "graph": graph,
            "global_idx": global_idx,
            "label": label,
        }

class HomologyShardDataset(torch.utils.data.Dataset):
    """
    Lightweight loader for aligned homology shards.

    Returns:
        {
            "prior": (K,) tensor,
            "homology_gate": (4,) tensor  # [b_max, cov_max, log1p_n_hits, has_hit]
            "stats": dict,
            "label": str,
            "global_idx": int,
        }
    """

    def __init__(
        self,
        homology_shard_dir: str | Path,
        manifest_path: str | Path,
        cache_size: int = 4,
        keep_ids = None,
    ) -> None:
        self.homology_shard_dir = Path(homology_shard_dir).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.cache_size = cache_size

        if not self.homology_shard_dir.exists():
            raise FileNotFoundError(f"Missing homology shard dir: {self.homology_shard_dir}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest path: {self.manifest_path}")

        self.shard_files = {
            int(path.stem.split("_")[-1]): path
            for path in sorted(self.homology_shard_dir.glob("homology_shard_*.pt"))
        }
        if not self.shard_files:
            raise FileNotFoundError(f"No homology shard files found in {self.homology_shard_dir}")

        self.index: list[tuple[int, int, int, str]] = []
        self.lengths: list[int] = []
        self.indices_by_shard: dict[int, list[int]] = defaultdict(list)

        manifest_rows = load_manifest_rows(self.manifest_path)
        for dataset_idx, row in enumerate(manifest_rows):
            shard_id = row["shard_number"]
            local_idx = row["local_seq_idx"]
            global_idx = row["global_seq_idx"]
            label = row["label"]
            seq_len = row["sequence_length"]

            self.index.append((shard_id, local_idx, global_idx, label))
            self.lengths.append(seq_len)
            self.indices_by_shard[shard_id].append(dataset_idx)
        self.index, self.lengths, self.indices_by_shard = filter_index_by_keep_ids(
            self.index,
            self.lengths,
            keep_ids,
            )

        self._cache: dict[int, dict] = {}
        self._cache_order: list[int] = []

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, shard_id: int) -> dict:
        if shard_id in self._cache:
            self._cache_order.remove(shard_id)
            self._cache_order.append(shard_id)
            return self._cache[shard_id]

        shard = torch.load(self.shard_files[shard_id], map_location="cpu")
        self._cache[shard_id] = shard
        self._cache_order.append(shard_id)

        while len(self._cache_order) > self.cache_size:
            old_shard_id = self._cache_order.pop(0)
            self._cache.pop(old_shard_id, None)

        return shard

    def __getitem__(self, idx: int) -> dict:
        shard_id, local_idx, global_idx, label = self.index[idx]
        shard = self._load_shard(shard_id)
        return {
            "prior": shard["priors"][local_idx],
            "homology_gate": shard["gate_features"][local_idx],
            "stats": shard["stats"][local_idx],
            "label": label,
            "global_idx": global_idx,
        }


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

    