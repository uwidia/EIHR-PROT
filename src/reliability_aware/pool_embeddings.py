import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, DataLoader
from reliability_aware.sharding import ESMShardDataset
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class HybridBatchSampler(Sampler):
    """
    Custom Batch Sampler. Minimizes shard file I/O bottleneck during dataloading
    - keeps a small active pool of shards
    - shuffles within each shard every epoch
    - builds each batch from the active pool only
    - sorts local candidates by length and take a random window
    - brings in new shards as active ones empty out
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

class ScalarAttentionPooling(nn.Module):
    """
    Scalar attention pooling over residue embeddings.

    Input:
        x: Tensor of shape (B, L, D)
        mask: Bool tensor of shape (B, L), where True = valid residue, False = padding
    Output:
        pooled: Tensor of shape (B, D)
        attn: Tensor of shape (B, L) with attention weights
    """

    def __init__(self, input_dim: int = 1280, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()

        if hidden_dim is None:
            self.scorer = nn.Linear(input_dim, 1)
        else:
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args: 
            x: (B, L, D)
            mask: (B, L) bool

        Returns:
            pooled: (B, D)
            attn: (B, L)
        """
        #Compute raw attention scores: (B, L, 1) -> (B, L)
        scores = self.scorer(x).squeeze(-1)

        #Mask out padding before softmax
        scores = scores.masked_fill(~mask, float(-1e9))

        #Attention weights over residues
        attn = torch.softmax(scores, dim = 1) # (B, L)

        #Weighted sum of residue embeddings
        pooled = torch.sum(x * attn.unsqueeze(-1), dim = 1) # (B, D)

        return pooled, attn

class ESMSequenceBranch(nn.Module):
    """
    Sequence branch:
    residue embeddings -> scalar attention pooling -> protein vector
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        attn_hidden_dim: int = 256,
        attn_dropout: float = 0.1,
        out_dim: int | None = None,
        ):
        super()__init__()
        
        self.pool = ScalarAttentionPooling(
            input_dim = esm_dim,
            hidden_dim = attn_hidden_dim,
            dropout = attn_dropout,
        )
        self.proj = None if out_dim is None else nn.Linear(esm_dim, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        pooled, attn = self.pool(x, mask) #pooled: (B, 1280)
        if self.proj is not None:
            pooled = self.proj(pooled)

        return pooled, attn


    

def collate_fn(batch):
    reps = [x[0] for x in batch]
    global_indices = torch.tensor([x[1] for x in batch], dtype=torch.long)

    lengths = [r.shape[0] for r in reps]
    max_len = max(lengths)
    dim = reps[0].shape[1]
    

    padded = torch.zeros(len(reps), max_len, dim, dtype=reps[0].dtype)
    mask = torch.zeros(len(reps), max_len, dtype=torch.bool)

    for i, r in enumerate(reps):
        L = r.shape[0]
        padded[i, :L] = r
        mask[i, :L] = True

    return padded, mask, global_indices



#Add to script file later
# dataset = ESMShardDataset(
#     shard_dir="path/to/shards",
#     manifest_path="path/to/manifest.csv",
#     cache_size=3,
# )

# BATCH_SIZE = 16
# batch_sampler = HybridBatchSampler(
#     dataset,
#     batch_size=BATCH_SIZE,
#     active_shards=3,
#     lookahead_factor=2,
#     drop_last=True,
#     seed=42,
# )

# loader = DataLoader(
#     dataset,
#     batch_sampler=batch_sampler,
#     collate_fn=collate_fn,
#     num_workers=0, #Performs best with 0. num_workers > 0 can cause unnecessary overhead since each worker gets its own shard cache
#     pin_memory=True,
# )

    