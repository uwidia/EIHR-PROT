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
from torch_geometric.data import Data, Batch
from utils.model_training import train_one_epoch
from utils.losses import hierarchy_loss, weighted_bce_on_probs
from utils.metrics import fmax_score, smin_score
from utils.pool_embeddings import GATBranch, NeuralLogitHead
from utils.shard_handling import ESMGraphShardDataset
from models.reliability_aware_model import HybridBatchSampler


class StructureOnlyESMGraphShardDataset(ESMGraphShardDataset):
    """
    Structure-only dataset built from aligned ESM + graph shards.

    The ESM residue embeddings are still loaded because they are not stored in
    the graph nodes. They are used as node features when building the PyG graph.
    """

    def __init__(
        self,
        esm_shard_dir: str | Path,
        graph_shard_dir: str | Path,
        manifest_path: str | Path,
        keep_ids: Sequence[str] | None = None,
        require_graph: bool = True,
        esm_cache_size: int = 2,
        graph_cache_size: int = 4,
    ):
        super().__init__(
            esm_shard_dir=esm_shard_dir,
            graph_shard_dir=graph_shard_dir,
            manifest_path=manifest_path,
            esm_cache_size=esm_cache_size,
            graph_cache_size=graph_cache_size,
            require_graph=require_graph,
            keep_ids=keep_ids,
        )

    def _filter_invalid_samples(self):
        """
        Mirrors the reliability-aware filtering path so the loader is resilient
        to bad or missing graph entries.
        """
        keep = []
        skipped = 0

        for idx in range(len(self)):
            try:
                sample = self[idx]
                if sample["graph"] is not None:
                    keep.append(idx)
            except Exception as exc:
                print(f"Skipping sample {idx}: {exc}")
                skipped += 1

        self.index = [self.index[i] for i in keep]
        self.lengths = [self.lengths[i] for i in keep]

        new_indices_by_shard = defaultdict(list)
        for new_idx, old_idx in enumerate(keep):
            shard_id = self.index[new_idx][0]
            new_indices_by_shard[shard_id].append(new_idx)
        self.indices_by_shard = new_indices_by_shard

        print(f"Filtering complete. Remaining samples: {len(keep)}")
        print(f"Skipped samples: {skipped}")


def make_structure_only_collate_fn(
    label_to_indices: Mapping[str, Sequence[int]],
    num_go_terms: int,
):
    """
    Collate function for structure-only batches.

    Returns:
        padded, mask, graph_batch, targets, global_indices, labels

    The padded ESM tensors are included for completeness/debugging, but the
    model itself only consumes the graph batch.
    """

    def collate(batch):
        reps = [item["rep"] for item in batch]
        graphs = [item["graph"] for item in batch]
        labels = [item["label"] for item in batch]
        global_indices = torch.tensor(
            [item["global_idx"] for item in batch], dtype=torch.long
        )

        lengths = [r.shape[0] for r in reps]
        targets = torch.zeros(len(batch), num_go_terms, dtype=torch.float32)

        pyg_graphs = []

        for i, (rep, graph, label) in enumerate(zip(reps, graphs, labels)):
            if graph is None:
                raise ValueError(f"Graph is None for label={label}")

            if rep.shape[0] != graph["coords"].shape[0]:
                raise ValueError(
                    f"Length mismatch for {label}: rep has {rep.shape[0]} residues, "
                    f"graph has {graph['coords'].shape[0]}"
                )

            L = rep.shape[0]

            edge_attr = torch.cat(
                [
                    graph["edge_attr"].float(),
                    graph["edge_weight"].float().unsqueeze(1),
                ],
                dim=1,
            )

            data = Data(
                x=rep.float(),
                confidence=graph["confidence"].float(),
                edge_index=graph["edge_index"].long(),
                edge_attr=edge_attr,
                label=label,
            )
            pyg_graphs.append(data)

            idxs = label_to_indices.get(label)
            if idxs is None:
                raise KeyError(f"No GO target found for label={label}")
            if idxs:
                targets[i, list(idxs)] = 1.0

        graph_batch = Batch.from_data_list(pyg_graphs)

        batch = {
            "padded": None,
            "mask": None,
            "graph_batch": graph_batch,
            "homology_scores": None,
            "gate_features": None,
            "targets": targets,
            "global_indices": global_indices,
            "labels": labels,
        }

        return batch

    return collate


class StructureOnlyProteinFunctionModel(nn.Module):
    """
    Structure-only baseline.

    Uses the graph branch to pool confidence-weighted structural neighborhoods
    over residue embeddings, then predicts GO terms from the resulting graph
    representation.
    """

    def __init__(
        self,
        num_go_terms: int,
        dropout: float = 0.2,
        head_dropout: float | None = None,
    ):
        super().__init__()

        self.gat_branch = GATBranch(dropout=dropout)
        self.head = NeuralLogitHead(
            in_dim=1280,
            num_go_terms=num_go_terms,
            dropout=head_dropout if head_dropout is not None else dropout,
        )

    def forward(self, graph_batch):
        graph_repr, graph_node_alpha = self.gat_branch(graph_batch)  # (B, 1280)
        logits = self.head(graph_repr)  # (B, C)
        probs = torch.sigmoid(logits)

        return {
            "probs": probs,
            "logits": logits,
            "graph_repr": graph_repr,
            "graph_node_alpha": graph_node_alpha,
        }


def run_one_batch_smoke_test_structure_only(
    model,
    train_loader,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    device,
):
    """
    Runs a single forward/backward pass on a deep copy of the model.
    Expects train_loader batches to be dictionaries.
    """
    model_copy = copy.deepcopy(model)
    model_copy.train()
    optimizer_copy = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)

    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    batch = next(iter(train_loader))

    graph_batch = batch["graph_batch"].to(device)
    targets = batch["targets"].to(device)

    optimizer_copy.zero_grad(set_to_none=True)

    outputs = model_copy(graph_batch=graph_batch)
    probs = outputs["probs"]

    assert (
        probs.shape == targets.shape
    ), f"Shape mismatch: probs={probs.shape}, targets={targets.shape}"

    assert (
        outputs["graph_repr"].shape[0] == targets.shape[0]
    ), "Batch dimension mismatch between graph_repr and targets"

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


def build_structure_only_model(sample_hparams, go_terms, device):
    dropout = float(sample_hparams.get("dropout", 0.2))
    head_dropout = sample_hparams.get("head_dropout", dropout)

    model = StructureOnlyProteinFunctionModel(
        num_go_terms=len(go_terms),
        dropout=dropout,
        head_dropout=head_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sample_hparams["learning_rate"],
        weight_decay=float(sample_hparams.get("weight_decay", 1e-4)),
    )

    return model, optimizer
