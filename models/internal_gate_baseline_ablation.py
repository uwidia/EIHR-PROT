from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from utils.model_training import train_one_epoch
from torch_geometric.data import Batch, Data

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


class InternalGateESMGraphHomologyShardDataset(ESMGraphHomologyShardDataset):
    """
    Dataset for the sequence + structure + homology internal-gate baseline.

    This mirrors the reliability-aware dataset because this ablation still needs:
      - ESM residue embeddings for the sequence branch and graph node features
      - structure graphs for the graph branch
      - homology priors for late fusion

    Unlike the reliability-aware model, explicit reliability indicators are not
    exposed to the collate function or model. The parent dataset may load
    homology gate features internally, but this subclass drops them at the
    sample boundary so the ablation cannot use q.
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

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        sample.pop("homology_gate", None)
        return sample


def make_internal_gate_collate_fn(
    label_to_indices: Mapping[str, Sequence[int]],
    num_go_terms: int,
):
    """
    Collate for sequence + structure + homology internal-gate batches.

    Returns a standardized batch dictionary:
        {
            "padded": padded,
            "mask": mask,
            "graph_batch": graph_batch,
            "homology_scores": homology_scores,
            "gate_features": None,
            "targets": targets,
            "global_indices": global_indices,
            "labels": labels,
        }

    The padded/mask tensors feed the sequence branch.
    The graph_batch feeds the GAT branch.
    The homology_scores tensor is fused with neural probabilities by a learned
    gate that only sees internal neural features.

    No external reliability indicator tensor q is built or returned.
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
        homology_score_list = []

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

            homology_score = item["homology_scores"].float()
            if homology_score.shape[0] != num_go_terms:
                raise ValueError(
                    f"Homology score size mismatch for {label}: "
                    f"expected {num_go_terms}, got {homology_score.shape[0]}"
                )
            homology_score_list.append(homology_score)

            idxs = label_to_indices.get(label)
            if idxs is None:
                raise KeyError(f"No GO target found for label={label}")
            if idxs:
                targets[i, list(idxs)] = 1.0

        graph_batch = Batch.from_data_list(pyg_graphs)
        homology_scores = torch.stack(homology_score_list, dim=0)

        return {
            "padded": padded,
            "mask": mask,
            "graph_batch": graph_batch,
            "homology_scores": homology_scores,
            "gate_features": None,
            "targets": targets,
            "global_indices": global_indices,
            "labels": labels,
        }

    return collate


class InternalFeatureGate(nn.Module):
    """
    Learned gate that uses internal neural features only.

    Computes:
        [alpha_n, alpha_h] = softmax(MLP(fused_repr))

    where fused_repr is produced by FusionMLP(seq_repr, graph_repr). This keeps
    the adaptivity mechanism but removes explicit reliability indicators q.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor):
        logits = self.net(features)
        return torch.softmax(logits, dim=-1)


class InternalGateProteinFunctionModel(nn.Module):
    """
    Sequence + structure + homology learned-gate baseline.

    Architecture:
      1) sequence branch -> seq_repr
      2) graph branch    -> graph_repr
      3) FusionMLP(seq_repr, graph_repr) -> fused_repr
      4) NeuralLogitHead(fused_repr) -> neural_logits
      5) sigmoid(neural_logits) -> neural_probs
      6) InternalFeatureGate(fused_repr) -> [alpha_n, alpha_h]
      7) fused_probs = alpha_n * neural_probs + alpha_h * homology_scores

    This tests whether adaptivity alone is sufficient when the gate cannot use
    externally defined reliability indicators.
    """

    def __init__(
        self,
        num_go_terms: int,
        fusion_hidden_dim: int = 1024,
        fusion_out_dim: int = 512,
        gate_hidden_dim: int = 128,
        dropout: float = 0.2,
        gate_dropout: float | None = None,
        attn_hidden_dim: int = 256,
        attn_dropout: float | None = None,
        head_dropout: float | None = None,
    ):
        super().__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        head_dropout = dropout if head_dropout is None else head_dropout
        gate_dropout = dropout if gate_dropout is None else gate_dropout

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
        self.gate = InternalFeatureGate(
            input_dim=fusion_out_dim,
            hidden_dim=gate_hidden_dim,
            dropout=gate_dropout,
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

        gate_weights = self.gate(fused_repr)
        alpha_n = gate_weights[:, 0].unsqueeze(-1)
        alpha_h = gate_weights[:, 1].unsqueeze(-1)

        fused_probs = alpha_n * neural_probs + alpha_h * homology_scores

        return {
            "probs": fused_probs,
            "fused_probs": fused_probs,
            "neural_probs": neural_probs,
            "neural_logits": neural_logits,
            "homology_scores": homology_scores,
            "gate_weights": gate_weights,
            "seq_repr": seq_repr,
            "graph_repr": graph_repr,
            "fused_repr": fused_repr,
            "seq_attn": seq_attn,
            "graph_node_alpha": graph_node_alpha,
        }


def run_one_batch_smoke_test_internal_gate(
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
    graph_batch = batch["graph_batch"].to(device)
    homology_scores = batch["homology_scores"].to(device)
    targets = batch["targets"].to(device)

    optimizer_copy.zero_grad(set_to_none=True)

    outputs = model_copy(
        padded=padded,
        mask=mask,
        graph_batch=graph_batch,
        homology_scores=homology_scores,
    )

    probs = outputs["probs"]

    assert (
        probs.shape == targets.shape
    ), f"Shape mismatch: probs={probs.shape}, targets={targets.shape}"

    assert outputs["neural_probs"].shape == targets.shape, (
        f"Shape mismatch: neural_probs={outputs['neural_probs'].shape}, "
        f"targets={targets.shape}"
    )

    assert homology_scores.shape == targets.shape, (
        f"Shape mismatch: homology_scores={homology_scores.shape}, "
        f"targets={targets.shape}"
    )

    assert outputs["gate_weights"].shape == (targets.shape[0], 2), (
        f"Gate shape mismatch: gate_weights={outputs['gate_weights'].shape}, "
        f"expected={(targets.shape[0], 2)}"
    )

    assert torch.allclose(
        outputs["gate_weights"].sum(dim=1),
        torch.ones(targets.shape[0], device=device),
        atol=1e-5,
    ), "Gate weights should sum to 1 for each sample"

    assert (
        outputs["seq_repr"].shape == outputs["graph_repr"].shape
    ), "seq_repr and graph_repr should both be (B, 1280)"

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
    print(
        "mean gate weights: "
        f"neural={outputs['gate_weights'][:, 0].mean().item():.4f}, "
        f"homology={outputs['gate_weights'][:, 1].mean().item():.4f}"
    )


def build_internal_gate_model(sample_hparams, go_terms, device):
    dropout = float(sample_hparams.get("dropout", 0.2))

    model = InternalGateProteinFunctionModel(
        num_go_terms=len(go_terms),
        fusion_hidden_dim=int(sample_hparams.get("fusion_hidden_dim", 1024)),
        fusion_out_dim=int(sample_hparams.get("fusion_out_dim", 512)),
        gate_hidden_dim=int(sample_hparams.get("gate_hidden_dim", 128)),
        dropout=dropout,
        gate_dropout=float(sample_hparams.get("gate_dropout", dropout)),
        attn_hidden_dim=int(sample_hparams.get("attn_hidden_dim", 256)),
        attn_dropout=float(sample_hparams.get("attn_dropout", dropout)),
        head_dropout=float(sample_hparams.get("head_dropout", dropout)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sample_hparams["learning_rate"],
        weight_decay=float(sample_hparams.get("weight_decay", 1e-4)),
    )

    return model, optimizer
