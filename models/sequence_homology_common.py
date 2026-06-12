from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from reliability_aware.utils.shard_handling import HomologyShardDataset


class ScalarAttentionPooling(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            self.scorer = nn.Linear(input_dim, 1)
        else:
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        scores = self.scorer(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float(-1e9))
        attn = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return pooled, attn


class ESMSequenceBranch(nn.Module):
    def __init__(
        self,
        esm_dim: int = 1280,
        attn_hidden_dim: int = 256,
        attn_dropout: float = 0.1,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.pool = ScalarAttentionPooling(
            input_dim=esm_dim,
            hidden_dim=attn_hidden_dim,
            dropout=attn_dropout,
        )
        self.proj = None if out_dim is None else nn.Linear(esm_dim, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        pooled, attn = self.pool(x, mask)
        if self.proj is not None:
            pooled = self.proj(pooled)
        return pooled, attn


class SequenceHomologyShardDataset(Dataset):
    """
    Aligned sequence + homology dataset for ESM residue embeddings and
    GO-aspect-specific homology priors/gate features.
    """

    def __init__(
        self,
        esm_shard_dir,
        homology_shard_dir,
        manifest_path,
        keep_ids: Sequence[str] | None = None,
    ):
        from models.sequence_only_ablation import SequenceOnlyESMShardDataset

        self.seq_ds = SequenceOnlyESMShardDataset(
            shard_dir=esm_shard_dir,
            manifest_path=manifest_path,
            keep_ids=keep_ids,
        )
        self.homology_ds = HomologyShardDataset(
            homology_shard_dir=homology_shard_dir,
            manifest_path=manifest_path,
            keep_ids=keep_ids,
        )

        if len(self.seq_ds) != len(self.homology_ds):
            raise ValueError(
                "Dataset length mismatch between sequence and homology datasets: "
                f"sequence={len(self.seq_ds)}, homology={len(self.homology_ds)}"
            )

        self.lengths = self.seq_ds.lengths
        self.indices_by_shard = self.seq_ds.indices_by_shard

    def __len__(self) -> int:
        return len(self.seq_ds)

    def __getitem__(self, idx: int) -> dict:
        rep, global_idx = self.seq_ds[idx]
        label = self.seq_ds.label_for_global_idx(global_idx)
        h = self.homology_ds[idx]

        if h["global_idx"] != global_idx:
            raise ValueError(
                f"Sequence/homology global_idx mismatch at idx={idx}: "
                f"sequence={global_idx}, homology={h['global_idx']}"
            )
        if h["label"] != label:
            raise ValueError(
                f"Sequence/homology label mismatch at idx={idx}: "
                f"sequence={label}, homology={h['label']}"
            )

        return {
            "rep": rep,
            "label": label,
            "global_idx": global_idx,
            "homology_scores": h["prior"].float(),
            "gate_features": h["homology_gate"].float(),
        }


def make_sequence_homology_collate_fn(
    label_to_indices: Mapping[str, Sequence[int]] | None,
    num_go_terms: int,
):
    """
    Collate sequence + homology samples into the standardized batch dictionary.
    """

    def collate(batch: Sequence[dict]) -> dict:
        reps = [item["rep"] for item in batch]
        labels = [item["label"] for item in batch]
        global_indices = torch.tensor(
            [item["global_idx"] for item in batch],
            dtype=torch.long,
        )

        lengths = [rep.shape[0] for rep in reps]
        max_len = max(lengths)
        dim = reps[0].shape[1]
        dtype = reps[0].dtype

        padded = torch.zeros(len(batch), max_len, dim, dtype=dtype)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        targets = (
            torch.zeros(len(batch), num_go_terms, dtype=torch.float32)
            if label_to_indices is not None
            else None
        )

        homology_scores = []
        gate_features = []

        for i, (item, rep, label) in enumerate(zip(batch, reps, labels)):
            if label_to_indices is not None and label not in label_to_indices:
                raise KeyError(f"Missing label in label_to_indices: {label}")

            L = rep.shape[0]
            padded[i, :L] = rep
            mask[i, :L] = True

            if label_to_indices is not None:
                idxs = label_to_indices[label]
                if idxs:
                    targets[i, list(idxs)] = 1.0

            scores = item["homology_scores"].float()
            if scores.shape != (num_go_terms,):
                raise ValueError(
                    f"homology_scores for label={label} must have shape "
                    f"({num_go_terms},), got {tuple(scores.shape)}"
                )
            homology_scores.append(scores)

            gates = item["gate_features"].float()
            if gates.shape != (4,):
                raise ValueError(
                    f"gate_features for label={label} must have shape (4,), "
                    f"got {tuple(gates.shape)}"
                )
            gate_features.append(gates)

        return {
            "padded": padded,
            "mask": mask,
            "graph_batch": None,
            "homology_scores": torch.stack(homology_scores, dim=0),
            "gate_features": torch.stack(gate_features, dim=0),
            "targets": targets,
            "global_indices": global_indices,
            "labels": labels,
        }

    return collate


class SequencePredictionHead(nn.Module):
    """
    Shared sequence prediction head for sequence-only and sequence+homology models.
    """

    def __init__(
        self,
        num_go_terms: int,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_go_terms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
