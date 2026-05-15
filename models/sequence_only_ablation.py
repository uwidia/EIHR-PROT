# sequence_only_ablation.py

from __future__ import annotations

import copy
import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from utils.model_training import train_one_epoch
from utils.shard_handling import ESMShardDataset
from models.reliability_aware_model import HybridBatchSampler
from utils.pool_embeddings import ESMSequenceBranch, NeuralLogitHead
from utils.losses import weighted_bce_on_probs, hierarchy_loss
from utils.metrics import fmax_score, smin_score


class SequenceOnlyESMShardDataset(ESMShardDataset):
    """
    Sequence-only dataset. Instance of ESMShardDataset Dataset class.

    Adds:
      - a global_idx -> label lookup from the manifest
      - optional filtering by keep_ids
      - preserved lengths and indices_by_shard for HybridBatchSampler
    """

    def __init__(
        self,
        shard_dir: str | Path,
        manifest_path: str | Path,
        keep_ids: Sequence[str] | None = None,
        cache_size: int = 3,
    ):
        super().__init__(
            shard_dir=shard_dir, manifest_path=manifest_path, cache_size=cache_size
        )

        self.manifest_path = Path(self.manifest_path)
        keep_ids_set = set(keep_ids) if keep_ids is not None else None

        # Read labels from the manifest in the same order as the base dataset.
        global_idx_to_label: dict[int, str] = {}
        new_index = []
        new_lengths = []
        new_indices_by_shard = defaultdict(list)

        with self.manifest_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for dataset_idx, row in enumerate(reader):
                shard_id, local_idx, global_idx = self.index[dataset_idx]
                label = row["label"]

                if keep_ids_set is not None and label not in keep_ids_set:
                    continue

                new_dataset_idx = len(new_index)
                new_index.append((shard_id, local_idx, global_idx))
                new_lengths.append(self.lengths[dataset_idx])
                new_indices_by_shard[shard_id].append(new_dataset_idx)
                global_idx_to_label[global_idx] = label

        self.index = new_index
        self.lengths = new_lengths
        self.indices_by_shard = new_indices_by_shard
        self.global_idx_to_label = global_idx_to_label

    def label_for_global_idx(self, global_idx: int) -> str:
        return self.global_idx_to_label[global_idx]


def make_sequence_only_collate_fn(
    global_idx_to_label: Mapping[int, str],
    label_to_indices: Mapping[str, Sequence[int]],
    num_go_terms: int,
):
    """
    Batch items come from ESMShardDataset:
        (rep, global_idx)

    This collate:
      - pads residue embeddings
      - builds a boolean mask
      - maps global_idx -> protein label -> GO multi-hot target
    """

    def collate(batch):
        reps = [item[0] for item in batch]
        global_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)

        labels = [global_idx_to_label[int(g)] for g in global_indices.tolist()]

        lengths = [r.shape[0] for r in reps]
        max_len = max(lengths)
        dim = reps[0].shape[1]
        dtype = reps[0].dtype

        padded = torch.zeros(len(batch), max_len, dim, dtype=dtype)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        targets = torch.zeros(len(batch), num_go_terms, dtype=torch.float32)

        for i, (rep, label) in enumerate(zip(reps, labels)):
            L = rep.shape[0]
            padded[i, :L] = rep
            mask[i, :L] = True

            idxs = label_to_indices.get(label, [])
            if idxs:
                targets[i, list(idxs)] = 1.0

        graph_batch = None
        homology_scores = None
        gate_features = None

        batch = {
            "padded": padded,
            "mask": mask,
            "graph_batch": graph_batch,
            "homology_scores": homology_scores,
            "gate_features": gate_features,
            "targets": targets,
            "global_indices": global_indices,
            "labels": labels,
        }

        return batch

    return collate


class SequenceOnlyProteinFunctionModel(nn.Module):
    """
    Frozen ESM sequence embeddings -> attention pooling -> multilabel GO head.
    """

    def __init__(
        self,
        num_go_terms: int,
        attn_hidden_dim: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        self.seq_branch = ESMSequenceBranch(
            esm_dim=1280,
            attn_hidden_dim=attn_hidden_dim,
            attn_dropout=attn_dropout,
            out_dim=None,
        )

        self.head = NeuralLogitHead(
            in_dim=1280,
            num_go_terms=num_go_terms,
            dropout=head_dropout,
        )

    def forward(self, padded: torch.Tensor, mask: torch.Tensor):
        seq_repr, seq_attn = self.seq_branch(padded, mask)  # (B, 1280)
        logits = self.head(seq_repr)  # (B, C)
        probs = torch.sigmoid(logits)

        return {
            "probs": probs,
            "logits": logits,
            "seq_repr": seq_repr,
            "seq_attn": seq_attn,
        }


def run_one_batch_smoke_test_sequence_only(
    model,
    train_loader,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    device,
):
    """
    Runs a single forward+backward pass on a fresh copy of the model to verify
    shapes and loss finiteness. Does NOT modify the original model or optimizer.
    Expects train_loader batches to be dictionaries.
    """
    model_copy = copy.deepcopy(model)
    model_copy.train()
    optimizer_copy = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)

    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    batch = next(iter(train_loader))

    padded = batch["padded"].to(device)
    mask = batch["mask"].to(device)
    targets = batch["targets"].to(device)

    optimizer_copy.zero_grad(set_to_none=True)

    outputs = model_copy(padded=padded, mask=mask)
    probs = outputs["probs"]

    assert (
        probs.shape == targets.shape
    ), f"Shape mismatch: probs={probs.shape}, targets={targets.shape}"

    bce = weighted_bce_on_probs(
        probs=probs,
        targets=targets,
        pos_weight=pos_weight,
    )

    hier = hierarchy_loss(
        fused_probs=probs,
        child_parent_pairs=child_parent_pairs,
    )

    loss = bce + lambda_hier * hier

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

    loss.backward()
    optimizer_copy.step()

    print("Smoke test passed")
    print(f"batch_size: {targets.shape[0]}")
    print(f"num_go_terms: {targets.shape[1]}")
    print(f"bce_loss: {bce.item():.6f}")
    print(f"hier_loss: {hier.item():.6f}")
    print(f"total_loss: {loss.item():.6f}")
    print(f"probs range: {probs.min().item():.6f} to {probs.max().item():.6f}")


def build_seq_only_model(sample_hparams, go_terms, device):
    model = SequenceOnlyProteinFunctionModel(
        num_go_terms=len(go_terms),
        attn_hidden_dim=sample_hparams["attn_hidden_dim"],
        attn_dropout=sample_hparams["attn_dropout"],
        head_dropout=sample_hparams["head_dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=sample_hparams["learning_rate"], weight_decay=1e-4
    )
    return model, optimizer
