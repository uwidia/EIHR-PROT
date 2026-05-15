from __future__ import annotations

import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from models.reliability_aware_model import HybridBatchSampler
from utils.model_training import train_one_epoch
from utils.losses import hierarchy_loss, weighted_bce_on_probs
from utils.metrics import fmax_score, smin_score
from utils.pool_embeddings import (
    ESMSequenceBranch,
    FusionMLP,
    GATBranch,
    NeuralLogitHead,
)
from utils.shard_handling import ESMGraphShardDataset

logger = logging.getLogger(__name__)


class SeqStructureESMGraphShardDataset(ESMGraphShardDataset):
    """
    Dataset for the sequence + structure fusion ablation.

    This mirrors the seq+graph part of the reliability-aware dataset, but does
    not load homology priors or homology gate features.

    Returned samples contain:
      - rep:         (L, 1280) frozen ESM residue embeddings
      - graph:       structure graph aligned to the same residue sequence
      - label:       protein ID / subject ID
      - global_idx:  global sequence index from the manifest
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
        Filters out samples whose graph cannot be loaded or is missing.

        This keeps the dataset compatible with HybridBatchSampler by rebuilding
        lengths and indices_by_shard after filtering.
        """
        keep = []
        skipped = 0

        for idx in range(len(self)):
            try:
                sample = self[idx]
                if sample["graph"] is not None:
                    keep.append(idx)
            except Exception as exc:
                logger.info("Skipping sample %s: %s", idx, exc)
                skipped += 1

        self.index = [self.index[i] for i in keep]
        self.lengths = [self.lengths[i] for i in keep]

        new_indices_by_shard = defaultdict(list)
        for new_idx, _old_idx in enumerate(keep):
            shard_id = self.index[new_idx][0]
            new_indices_by_shard[shard_id].append(new_idx)
        self.indices_by_shard = new_indices_by_shard

        print(f"Filtering complete. Remaining samples: {len(keep)}")
        print(f"Skipped samples: {skipped}")


def make_seq_structure_collate_fn(
    label_to_indices: Mapping[str, Sequence[int]],
    num_go_terms: int,
):
    """
    Collate for sequence + structure fusion batches.

    Returns:
        padded, mask, graph_batch, targets, global_indices, labels

    The padded/mask tensors feed the sequence branch. The graph_batch feeds the
    GAT branch. No homology priors or reliability-gate features are produced.
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

        for i, (rep, graph, label) in enumerate(zip(reps, graphs, labels)):
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

            idxs = label_to_indices.get(label)
            if idxs is None:
                raise KeyError(f"No GO target found for label={label}")
            if idxs:
                targets[i, list(idxs)] = 1.0

        graph_batch = Batch.from_data_list(pyg_graphs)

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


class SeqStructureFusionProteinFunctionModel(nn.Module):
    """
    Sequence + structure fusion baseline.

    Architecture:
      1) sequence branch -> seq_repr
      2) graph branch    -> graph_repr
      3) FusionMLP(seq_repr, graph_repr) -> fused_repr
      4) NeuralLogitHead(fused_repr) -> logits
      5) sigmoid(logits) -> probabilities

    No homology expert and no reliability gate are used.
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
    ):
        super().__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        head_dropout = dropout if head_dropout is None else head_dropout

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

    def forward(self, padded: torch.Tensor, mask: torch.Tensor, graph_batch):
        seq_repr, seq_attn = self.seq_branch(padded, mask)
        graph_repr, graph_node_alpha = self.gat_branch(graph_batch)

        fused_repr = self.fusion(seq_repr, graph_repr)
        logits = self.neural_head(fused_repr)
        probs = torch.sigmoid(logits)

        return {
            "probs": probs,
            "fused_probs": probs,
            "logits": logits,
            "neural_logits": logits,
            "seq_repr": seq_repr,
            "graph_repr": graph_repr,
            "fused_repr": fused_repr,
            "seq_attn": seq_attn,
            "graph_node_alpha": graph_node_alpha,
        }


def run_one_batch_smoke_test_seq_structure(
    model,
    train_loader,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    device,
):
    """
    Runs one forward/backward update on a deep copy of the model.

    This checks shape compatibility and loss finiteness without modifying the
    actual model or optimizer used for training.
    """
    model_copy = copy.deepcopy(model)
    model_copy.train()
    optimizer_copy = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)

    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    batch = next(iter(train_loader))
    padded, mask, graph_batch, targets, _, _ = batch

    padded = padded.to(device)
    mask = mask.to(device)
    graph_batch = graph_batch.to(device)
    targets = targets.to(device)

    optimizer_copy.zero_grad(set_to_none=True)

    outputs = model_copy(
        padded=padded,
        mask=mask,
        graph_batch=graph_batch,
    )
    probs = outputs["probs"]

    assert (
        probs.shape == targets.shape
    ), f"Shape mismatch: probs={probs.shape}, targets={targets.shape}"
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


def build_seq_structure_loaders(
    train_esm_shard_dir,
    val_esm_shard_dir,
    train_graph_shard_dir,
    val_graph_shard_dir,
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
    **kwargs,
):
    train_dataset = SeqStructureESMGraphShardDataset(
        esm_shard_dir=train_esm_shard_dir,
        graph_shard_dir=train_graph_shard_dir,
        manifest_path=train_manifest_path,
        require_graph=True,
        keep_ids=train_keep_ids_for_aspect,
    )

    val_dataset = SeqStructureESMGraphShardDataset(
        esm_shard_dir=val_esm_shard_dir,
        graph_shard_dir=val_graph_shard_dir,
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
        collate_fn=make_seq_structure_collate_fn(
            label_to_indices=train_label_to_indices,
            num_go_terms=len(go_terms),
        ),
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=make_seq_structure_collate_fn(
            label_to_indices=val_label_to_indices,
            num_go_terms=len(go_terms),
        ),
        num_workers=0,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_seq_structure_model(sample_hparams, go_terms, device):
    dropout = float(sample_hparams.get("dropout", 0.2))

    model = SeqStructureFusionProteinFunctionModel(
        num_go_terms=len(go_terms),
        fusion_hidden_dim=int(sample_hparams.get("fusion_hidden_dim", 1024)),
        fusion_out_dim=int(sample_hparams.get("fusion_out_dim", 512)),
        dropout=dropout,
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
