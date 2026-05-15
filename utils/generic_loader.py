from __future__ import annotations

from typing import Literal
from torch.utils.data import DataLoader

from models.reliability_aware_model import HybridBatchSampler

DatasetKind = Literal["sequence", "graph", "graph_homology"]


def _build_dataset_kwargs(
    *,
    dataset_kind: DatasetKind,
    split: Literal["train", "val"],
    train_esm_shard_dir,
    val_esm_shard_dir,
    train_graph_shard_dir,
    val_graph_shard_dir,
    train_homology_shard_dir,
    val_homology_shard_dir,
    train_manifest_path,
    val_manifest_path,
    train_keep_ids_for_aspect,
    val_keep_ids_for_aspect,
):
    is_train = split == "train"

    esm_shard_dir = train_esm_shard_dir if is_train else val_esm_shard_dir
    graph_shard_dir = train_graph_shard_dir if is_train else val_graph_shard_dir
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

    if dataset_kind == "graph":
        return {
            "esm_shard_dir": esm_shard_dir,
            "graph_shard_dir": graph_shard_dir,
            "manifest_path": manifest_path,
            "keep_ids": keep_ids,
            "require_graph": True,
        }

    if dataset_kind == "graph_homology":
        return {
            "esm_shard_dir": esm_shard_dir,
            "graph_shard_dir": graph_shard_dir,
            "homology_shard_dir": homology_shard_dir,
            "manifest_path": manifest_path,
            "keep_ids": keep_ids,
            "require_graph": True,
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
    train_graph_shard_dir=None,
    val_graph_shard_dir=None,
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
            train_graph_shard_dir=train_graph_shard_dir,
            val_graph_shard_dir=val_graph_shard_dir,
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
            train_graph_shard_dir=train_graph_shard_dir,
            val_graph_shard_dir=val_graph_shard_dir,
            train_homology_shard_dir=train_homology_shard_dir,
            val_homology_shard_dir=val_homology_shard_dir,
            train_manifest_path=train_manifest_path,
            val_manifest_path=val_manifest_path,
            train_keep_ids_for_aspect=train_keep_ids_for_aspect,
            val_keep_ids_for_aspect=val_keep_ids_for_aspect,
        )
    )

    if filter_invalid_samples:
        train_dataset._filter_invalid_samples()
        val_dataset._filter_invalid_samples()

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
