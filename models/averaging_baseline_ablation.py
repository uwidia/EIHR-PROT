from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from utils.model_training import train_one_epoch

from models.reliability_aware_model import HybridBatchSampler
from utils.losses import hierarchy_loss, weighted_bce_on_probs
from utils.metrics import fmax_score, smin_score
from utils.pool_embeddings import (
    ESMSequenceBranch,
    FusionMLP,
    GATBranch,
    NeuralLogitHead,
)
from utils.shard_handling import ESMGraphHomologyShardDataset

logger = logging.getLogger(__name__)


class AveragingESMGraphHomologyShardDataset(ESMGraphHomologyShardDataset):
    """
    Dataset for the sequence + structure + homology averaging baseline.

    This mirrors the reliability-aware dataset because this ablation still needs:
      - ESM residue embeddings for the sequence branch and graph node features
      - structure graphs for the graph branch
      - homology priors for fixed late fusion

    Unlike the reliability-aware model, homology gate features are loaded by the
    underlying dataset but are not used by the collate function or model.
    """

    def __init__(
        self,
        esm_shard_dir: str | Path,
        graph_shard_dir: str | Path,
        homology_shard_dir: str | Path,
        manifest_path: str | Path,
        keep_ids: Sequence[str] | None = None,
        require_graph: bool = True,
    ):
        super().__init__(
            esm_shard_dir=esm_shard_dir,
            graph_shard_dir=graph_shard_dir,
            homology_shard_dir=homology_shard_dir,
            manifest_path=manifest_path,
            require_graph=require_graph,
            keep_ids=keep_ids,
        )


def make_averaging_collate_fn(
    label_to_indices: Mapping[str, Sequence[int]],
    num_go_terms: int,
):
    """
    Collate for sequence + structure + homology averaging batches.

    Returns:
        padded, mask, graph_batch, homology_scores, targets, global_indices, labels

    The padded/mask tensors feed the sequence branch. The graph_batch feeds the
    GAT branch. The homology_scores tensor is averaged with neural probabilities.
    No reliability-gate features are returned.
    """

    def collate(batch):
        reps = [item["rep"] for item in batch]
        graphs = [item["graph"] for item in batch]
        labels = [item["label"] for item in batch]
        global_indices = torch.tensor(
            [item["global_idx"] for item in batch], dtype=torch.long
        )

        lengths = [rep.shape[0] for rep in reps]
        max_len = max(lengths)
        esm_dim = reps[0].shape[1]
        dtype = reps[0].dtype

        padded = torch.zeros(len(batch), max_len, esm_dim, dtype=dtype)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        targets = torch.zeros(len(batch), num_go_terms, dtype=torch.float32)

        pyg_graphs = []
        homology_scores = []

        for i, (item, rep, graph, label) in enumerate(zip(batch, reps, graphs, labels)):
            if graph is None:
                raise ValueError(f"Graph is None for label={label}")

            if rep.shape[0] != graph["coords"].shape[0]:
                raise ValueError(
                    f"Length mismatch for {label}: rep has {rep.shape[0]} residues, "
                    f"graph has {graph['coords'].shape[0]}"
                )

            seq_len = rep.shape[0]
            padded[i, :seq_len] = rep
            mask[i, :seq_len] = True

            edge_attr = torch.cat(
                [
                    graph["edge_attr"].float(),
                    graph["edge_weight"].float().unsqueeze(1),
                ],
                dim=1,
            )

            pyg_graphs.append(
                Data(
                    x=rep.float(),
                    confidence=graph["confidence"].float(),
                    edge_index=graph["edge_index"].long(),
                    edge_attr=edge_attr,
                    label=label,
                )
            )

            homology_scores = item["homology_scores"].float()
            if homology_scores.shape[0] != num_go_terms:
                raise ValueError(
                    f"Homology prior size mismatch for {label}: "
                    f"expected {num_go_terms}, got {homology_scores.shape[0]}"
                )
            homology_scores.append(homology_scores)

            idxs = label_to_indices.get(label)
            if idxs is None:
                raise KeyError(f"No GO target found for label={label}")
            if idxs:
                targets[i, list(idxs)] = 1.0

        graph_batch = Batch.from_data_list(pyg_graphs)
        homology_scores = torch.stack(homology_scores, dim=0)

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


class AveragingFusionProteinFunctionModel(nn.Module):
    """
    Sequence + structure + homology fixed-averaging baseline.

    Architecture:
      1) sequence branch -> seq_repr
      2) graph branch    -> graph_repr
      3) FusionMLP(seq_repr, graph_repr) -> fused_repr
      4) NeuralLogitHead(fused_repr) -> neural_logits
      5) sigmoid(neural_logits) -> neural_probs
      6) fixed average: fused_probs = 0.5 * neural_probs + 0.5 * homology_scores

    This keeps the late-fusion idea from the reliability-aware model while
    removing the learned reliability gate.
    """

    def __init__(
        self,
        num_go_terms: int,
        fusion_hidden_dim: int = 1024,
        fusion_out_dim: int = 512,
        dropout: float = 0.2,
        attn_hidden_dim: int = 256,
        attn_dropout: float | None = None,
        head_dropout: float | None = None,
        neural_weight: float = 0.5,
    ):
        super().__init__()

        if not 0.0 <= neural_weight <= 1.0:
            raise ValueError(f"neural_weight must be in [0, 1], got {neural_weight}")

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        head_dropout = dropout if head_dropout is None else head_dropout

        self.neural_weight = float(neural_weight)
        self.homology_weight = 1.0 - self.neural_weight

        self.seq_branch = ESMSequenceBranch(
            esm_dim=1280,
            attn_hidden_dim=attn_hidden_dim,
            attn_dropout=attn_dropout,
            out_dim=None,
        )
        self.gat_branch = GATBranch(dropout=dropout)

        self.fusion = FusionMLP(
            seq_dim=1280,
            graph_dim=1280,
            hidden_dim=fusion_hidden_dim,
            out_dim=fusion_out_dim,
            dropout=dropout,
        )
        self.neural_head = NeuralLogitHead(
            in_dim=fusion_out_dim,
            num_go_terms=num_go_terms,
            dropout=head_dropout,
        )

    def forward(
        self,
        padded: torch.Tensor,
        mask: torch.Tensor,
        graph_batch,
        homology_scores: torch.Tensor,
    ):
        seq_repr, seq_attn = self.seq_branch(padded, mask)
        graph_repr, graph_node_alpha = self.gat_branch(graph_batch)

        fused_repr = self.fusion(seq_repr, graph_repr)
        neural_logits = self.neural_head(fused_repr)
        neural_probs = torch.sigmoid(neural_logits)

        homology_scores = homology_scores.to(
            device=neural_probs.device,
            dtype=neural_probs.dtype,
        )
        fused_probs = (
            self.neural_weight * neural_probs + self.homology_weight * homology_scores
        )

        return {
            "probs": fused_probs,
            "fused_probs": fused_probs,
            "neural_probs": neural_probs,
            "neural_logits": neural_logits,
            "homology_scores": homology_scores,
            "seq_repr": seq_repr,
            "graph_repr": graph_repr,
            "fused_repr": fused_repr,
            "seq_attn": seq_attn,
            "graph_node_alpha": graph_node_alpha,
            "neural_weight": self.neural_weight,
            "homology_weight": self.homology_weight,
        }


def run_one_batch_smoke_test_averaging(
    model,
    train_loader,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    device,
):
    """
    Runs one forward/backward update on a deep copy of the model.

    This verifies shape compatibility and loss finiteness without modifying the
    actual model or optimizer used for training.
    """
    model_copy = copy.deepcopy(model)
    model_copy.train()
    optimizer_copy = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)

    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    batch = next(iter(train_loader))
    padded, mask, graph_batch, homology_scores, targets, _, _ = batch

    padded = padded.to(device)
    mask = mask.to(device)
    graph_batch = graph_batch.to(device)
    homology_scores = homology_scores.to(device)
    targets = targets.to(device)

    optimizer_copy.zero_grad(set_to_none=True)

    outputs = model_copy(
        padded=padded,
        mask=mask,
        graph_batch=graph_batch,
        homology_scores=homology_scores,
    )
    fused_probs = outputs["fused_probs"]

    assert (
        fused_probs.shape == targets.shape
    ), f"Shape mismatch: fused_probs={fused_probs.shape}, targets={targets.shape}"
    assert outputs["neural_probs"].shape == targets.shape, (
        f"Shape mismatch: neural_probs={outputs['neural_probs'].shape}, "
        f"targets={targets.shape}"
    )
    assert homology_scores.shape == targets.shape, (
        f"Shape mismatch: homology_scores={homology_scores.shape}, "
        f"targets={targets.shape}"
    )
    assert (
        outputs["seq_repr"].shape == outputs["graph_repr"].shape
    ), "seq_repr and graph_repr should both be (B, 1280)"

    bce = weighted_bce_on_probs(
        probs=fused_probs,
        targets=targets,
        pos_weight=pos_weight,
    )
    hier = hierarchy_loss(
        fused_probs=fused_probs,
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
    print(
        f"fused_probs range: {fused_probs.min().item():.6f} "
        f"to {fused_probs.max().item():.6f}"
    )
    print(
        f"neural_weight={outputs['neural_weight']:.2f}, "
        f"homology_weight={outputs['homology_weight']:.2f}"
    )


def build_averaging_loaders(
    train_esm_shard_dir,
    val_esm_shard_dir,
    train_graph_shard_dir,
    val_graph_shard_dir,
    train_manifest_path,
    val_manifest_path,
    train_homology_shard_dir,
    val_homology_shard_dir,
    train_keep_ids_for_aspect,
    val_keep_ids_for_aspect,
    train_label_to_indices,
    val_label_to_indices,
    go_terms,
    batch_size: int = 16,
    seed: int = 42,
    active_shards: int = 3,
    lookahead_factor: int = 3,
    **kwargs,
):
    train_dataset = AveragingESMGraphHomologyShardDataset(
        esm_shard_dir=train_esm_shard_dir,
        graph_shard_dir=train_graph_shard_dir,
        homology_shard_dir=train_homology_shard_dir,
        manifest_path=train_manifest_path,
        require_graph=True,
        keep_ids=train_keep_ids_for_aspect,
    )

    val_dataset = AveragingESMGraphHomologyShardDataset(
        esm_shard_dir=val_esm_shard_dir,
        graph_shard_dir=val_graph_shard_dir,
        homology_shard_dir=val_homology_shard_dir,
        manifest_path=val_manifest_path,
        require_graph=True,
        keep_ids=val_keep_ids_for_aspect,
    )

    print("Filtering train dataset.")
    train_dataset._filter_invalid_samples()

    print("Filtering validation dataset.")
    val_dataset._filter_invalid_samples()

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
        collate_fn=make_averaging_collate_fn(
            label_to_indices=train_label_to_indices,
            num_go_terms=len(go_terms),
        ),
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=make_averaging_collate_fn(
            label_to_indices=val_label_to_indices,
            num_go_terms=len(go_terms),
        ),
        num_workers=0,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_averaging_model(sample_hparams, go_terms, device):
    dropout = float(sample_hparams.get("dropout", 0.2))

    model = AveragingFusionProteinFunctionModel(
        num_go_terms=len(go_terms),
        fusion_hidden_dim=int(sample_hparams.get("fusion_hidden_dim", 1024)),
        fusion_out_dim=int(sample_hparams.get("fusion_out_dim", 512)),
        dropout=dropout,
        attn_hidden_dim=int(sample_hparams.get("attn_hidden_dim", 256)),
        attn_dropout=float(sample_hparams.get("attn_dropout", dropout)),
        head_dropout=float(sample_hparams.get("head_dropout", dropout)),
        neural_weight=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sample_hparams["learning_rate"],
        weight_decay=float(sample_hparams.get("weight_decay", 1e-4)),
    )

    return model, optimizer
