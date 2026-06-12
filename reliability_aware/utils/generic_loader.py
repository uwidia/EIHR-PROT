from __future__ import annotations

import random
from collections import deque
from typing import Literal

from torch.utils.data import DataLoader, Sampler



class HybridBatchSampler(Sampler):
    """Shard-aware batch sampler."""

    def __init__(
        self,
        dataset,
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
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

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
                candidates.extend(list(shard_queues[sid])[: self.lookahead])

            if not candidates:
                break

            candidates.sort(key=lambda idx: self.dataset.lengths[idx])
            if len(candidates) <= self.batch_size:
                batch = candidates
            else:
                start = rng.randint(0, len(candidates) - self.batch_size)
                batch = candidates[start : start + self.batch_size]

            if len(batch) < self.batch_size and self.drop_last:
                break

            chosen = set(batch)
            for sid in active:
                shard_queues[sid] = deque(
                    idx for idx in shard_queues[sid] if idx not in chosen
                )

            yield batch


DatasetKind = Literal["sequence", "sequence_homology"]


def _build_dataset_kwargs(
    *,
    dataset_kind: DatasetKind,
    split: Literal["train", "val"],
    train_esm_shard_dir,
    val_esm_shard_dir,
    train_homology_shard_dir,
    val_homology_shard_dir,
    train_manifest_path,
    val_manifest_path,
    train_keep_ids_for_aspect,
    val_keep_ids_for_aspect,
):
    is_train = split == "train"

    esm_shard_dir = train_esm_shard_dir if is_train else val_esm_shard_dir
    homology_shard_dir = (
        train_homology_shard_dir if is_train else val_homology_shard_dir
    )
    manifest_path = train_manifest_path if is_train else val_manifest_path
    keep_ids = train_keep_ids_for_aspect if is_train else val_keep_ids_for_aspect

    if dataset_kind == "sequence":
        return {
            "shard_dir": esm_shard_dir,
            "manifest_path": manifest_path,
            "keep_ids": keep_ids,
        }

    if dataset_kind == "sequence_homology":
        return {
            "esm_shard_dir": esm_shard_dir,
            "homology_shard_dir": homology_shard_dir,
            "manifest_path": manifest_path,
            "keep_ids": keep_ids,
        }

    raise ValueError(f"Unknown dataset_kind: {dataset_kind}")


def _build_collate_fn(
    *,
    dataset_kind: DatasetKind,
    dataset,
    collate_factory,
    label_to_indices,
    num_go_terms: int,
):
    if dataset_kind == "sequence":
        return collate_factory(
            global_idx_to_label=dataset.global_idx_to_label,
            label_to_indices=label_to_indices,
            num_go_terms=num_go_terms,
        )

    return collate_factory(
        label_to_indices=label_to_indices,
        num_go_terms=num_go_terms,
    )


def build_model_loaders(
    *,
    dataset_cls,
    dataset_kind: DatasetKind,
    collate_factory,
    filter_invalid_samples: bool,
    train_esm_shard_dir,
    val_esm_shard_dir,
    train_homology_shard_dir=None,
    val_homology_shard_dir=None,
    train_manifest_path,
    val_manifest_path,
    train_keep_ids_for_aspect,
    val_keep_ids_for_aspect,
    train_label_to_indices,
    val_label_to_indices,
    go_terms,
    batch_size: int = 16,
    seed: int = 42,
    active_shards: int = 3,
    lookahead_factor: int = 3,
):
    train_dataset = dataset_cls(
        **_build_dataset_kwargs(
            dataset_kind=dataset_kind,
            split="train",
            train_esm_shard_dir=train_esm_shard_dir,
            val_esm_shard_dir=val_esm_shard_dir,
            train_homology_shard_dir=train_homology_shard_dir,
            val_homology_shard_dir=val_homology_shard_dir,
            train_manifest_path=train_manifest_path,
            val_manifest_path=val_manifest_path,
            train_keep_ids_for_aspect=train_keep_ids_for_aspect,
            val_keep_ids_for_aspect=val_keep_ids_for_aspect,
        )
    )

    val_dataset = dataset_cls(
        **_build_dataset_kwargs(
            dataset_kind=dataset_kind,
            split="val",
            train_esm_shard_dir=train_esm_shard_dir,
            val_esm_shard_dir=val_esm_shard_dir,
            train_homology_shard_dir=train_homology_shard_dir,
            val_homology_shard_dir=val_homology_shard_dir,
            train_manifest_path=train_manifest_path,
            val_manifest_path=val_manifest_path,
            train_keep_ids_for_aspect=train_keep_ids_for_aspect,
            val_keep_ids_for_aspect=val_keep_ids_for_aspect,
        )
    )



    train_collate_fn = _build_collate_fn(
        dataset_kind=dataset_kind,
        dataset=train_dataset,
        collate_factory=collate_factory,
        label_to_indices=train_label_to_indices,
        num_go_terms=len(go_terms),
    )

    val_collate_fn = _build_collate_fn(
        dataset_kind=dataset_kind,
        dataset=val_dataset,
        collate_factory=collate_factory,
        label_to_indices=val_label_to_indices,
        num_go_terms=len(go_terms),
    )

    train_batch_sampler = HybridBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        active_shards=active_shards,
        lookahead_factor=lookahead_factor,
        drop_last=True,
        seed=seed,
    )

    val_batch_sampler = HybridBatchSampler(
        dataset=val_dataset,
        batch_size=batch_size,
        active_shards=active_shards,
        lookahead_factor=lookahead_factor,
        drop_last=False,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=train_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=val_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader
