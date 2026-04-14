import torch
from torch.utils.data import Sampler
from reliability_aware.sharding import ESMShardDataset
from collections import deque
import random


class HybridBatchSampler(Sampler):
    """
    Short hybrid strategy:
    - keep a small active pool of shards
    - shuffle within each shard every epoch
    - build each batch from the active pool only
    - sort local candidates by length and take a random window
    - bring in new shards as active ones empty out
    """

    def __init__(
        self,
        dataset: ESMShardDataset,
        batch_size: int,
        active_shards: int = 3,
        lookahead_factor: int = 2,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.active_shards = min(active_shards, len(dataset.indices_by_shard))
        self.lookahead = batch_size * lookahead_factor
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        shard_order = list(self.dataset.indices_by_shard.keys())
        rng.shuffle(shard_order)

        shard_queues = {}
        for shard_id in shard_order:
            idxs = list(self.dataset.indices_by_shard[shard_id])
            rng.shuffle(idxs)
            shard_queues[shard_id] = deque(idxs)

        remaining_shards = deque(shard_order)
        active = []

        def refill():
            while len(active) < self.active_shards and remaining_shards:
                active.append(remaining_shards.popleft())

        refill()

        while active:
            active[:] = [sid for sid in active if shard_queues[sid]]
            refill()
            if not active:
                break

            candidates = []
            for sid in active:
                candidates.extend(list(shard_queues[sid])[:self.lookahead])

            if not candidates:
                break

            candidates.sort(key=lambda idx: self.dataset.lengths[idx])

            if len(candidates) <= self.batch_size:
                batch = candidates
            else:
                start = rng.randint(0, len(candidates) - self.batch_size)
                batch = candidates[start:start + self.batch_size]

            if len(batch) < self.batch_size and self.drop_last:
                break

            chosen = set(batch)
            for sid in active:
                shard_queues[sid] = deque(idx for idx in shard_queues[sid] if idx not in chosen)

            yield batch


def collate_fn(batch):
    reps = [x[0] for x in batch]
    global_indices = torch.tensor([x[1] for x in batch], dtype=torch.long)

    lengths = [r.shape[0] for r in reps]
    max_len = max(lengths)
    dim = reps[0].shape[1]

    padded = torch.zeros(len(reps), max_len, dim, dtype=reps[0].dtype)
    mask = torch.zeros(len(reps), max_len, dtype=torch.bool)

    for i, r in enumerate(reps):
        padded[i, :r.shape[0]] = r
        mask[i, :r.shape[0]] = True

    return padded, mask, global_indices