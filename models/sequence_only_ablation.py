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

        return padded, mask, targets, global_indices, labels

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
    """
    # FIX 1: removed @torch.no_grad() — gradients are needed for backward().
    # FIX 2: use a detached copy so the real model/optimizer are not touched.
    model_copy = copy.deepcopy(model)
    model_copy.train()
    optimizer_copy = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)

    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    batch = next(iter(train_loader))
    padded, mask, targets, _, _ = batch

    padded = padded.to(device)
    mask = mask.to(device)
    targets = targets.to(device)

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


def train_one_epoch_sequence_only(
    model,
    loader,
    optimizer,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    device,
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    # FIX 3: move pos_weight and child_parent_pairs to device once, outside the loop.
    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    for padded, mask, targets, _, _ in loader:
        padded = padded.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(padded=padded, mask=mask)

        bce_loss = weighted_bce_on_probs(
            probs=outputs["probs"],
            targets=targets,
            pos_weight=pos_weight,
        )
        hier_loss = hierarchy_loss(
            fused_probs=outputs["probs"],
            child_parent_pairs=child_parent_pairs,
        )
        loss = bce_loss + lambda_hier * hier_loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_sequence_only(
    model,
    loader,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    ic,
    device,
):
    model.eval()
    total_loss = 0.0
    bce_total = 0.0
    hier_total = 0.0
    n_batches = 0

    all_probs = []
    all_targets = []

    # FIX 3: move pos_weight and child_parent_pairs to device once, outside the loop.
    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    for padded, mask, targets, _, _ in loader:
        padded = padded.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        outputs = model(padded=padded, mask=mask)

        bce_loss = weighted_bce_on_probs(
            probs=outputs["probs"],
            targets=targets,
            pos_weight=pos_weight,
        )
        hier_loss = hierarchy_loss(
            fused_probs=outputs["probs"],
            child_parent_pairs=child_parent_pairs,
        )
        loss = bce_loss + lambda_hier * hier_loss

        total_loss += loss.item()
        bce_total += bce_loss.item()
        hier_total += hier_loss.item()
        n_batches += 1

        all_probs.append(outputs["probs"].detach().cpu())
        all_targets.append(targets.detach().cpu())

    y_prob = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    fmax = fmax_score(y_true, y_prob)
    smin = smin_score(y_true, y_prob, ic)
    aupr = average_precision_score(y_true.numpy().ravel(), y_prob.numpy().ravel())

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "bce_loss": bce_total / max(n_batches, 1),
        "hierarchy_loss": hier_total / max(n_batches, 1),
        "Fmax": fmax["Fmax"],
        "Fmax_threshold": fmax["threshold"],
        "AUPR": float(aupr),
        "Smin_raw": smin["raw"]["Smin"],
        "Smin_threshold_raw": smin["raw"]["threshold"],
        "Smin_normalized": smin["normalized"]["Smin"],
        "Smin_threshold_normalized": smin["normalized"]["threshold"],
    }


def fit_sequence_only(
    model,
    train_loader,
    val_loader,
    optimizer,
    pos_weight,
    child_parent_pairs,
    ic,
    device,
    *,
    lambda_hier: float = 0.0,
    num_epochs: int = 100,
    patience: int = 10,
    out_dir: str | Path = "runs/sequence_only",
):
    """
    Standard training loop for the ablation.

    lambda_hier defaults to 0.0 for a clean sequence-only BCE baseline.
    Set it > 0 if you want to keep the hierarchy regularizer for comparability.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_Fmax": [],
        "val_AUPR": [],
        "val_Smin_raw": [],
        "val_Smin_normalized": [],
        "val_Smin_raw_threshold": [],
        "val_Smin_normalized_threshold": [],
        "val_Fmax_threshold": [],
    }

    best_fmax = -1.0
    best_epoch = -1  # FIX 5: track best epoch for early stopping message.
    bad_epochs = 0
    best_path = out_dir / "best_model.pt"
    history_path = out_dir / "history.pt"  # FIX 4: save history incrementally.

    for epoch in range(1, num_epochs + 1):
        if hasattr(train_loader, "batch_sampler") and hasattr(
            train_loader.batch_sampler, "set_epoch"
        ):
            train_loader.batch_sampler.set_epoch(epoch)

        train_loss = train_one_epoch_sequence_only(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            pos_weight=pos_weight,
            child_parent_pairs=child_parent_pairs,
            lambda_hier=lambda_hier,
            device=device,
        )

        val_metrics = evaluate_sequence_only(
            model=model,
            loader=val_loader,
            pos_weight=pos_weight,
            child_parent_pairs=child_parent_pairs,
            lambda_hier=lambda_hier,
            ic=ic,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_Fmax"].append(val_metrics["Fmax"])
        history["val_AUPR"].append(val_metrics["AUPR"])
        history["val_Smin_raw"].append(val_metrics["Smin_raw"])
        history["val_Smin_normalized"].append(val_metrics["Smin_normalized"])
        history["val_Smin_raw_threshold"].append(val_metrics["Smin_threshold_raw"])
        history["val_Smin_normalized_threshold"].append(
            val_metrics["Smin_threshold_normalized"]
        )
        history["val_Fmax_threshold"].append(val_metrics["Fmax_threshold"])

        # FIX 4: save history after every epoch so it's never lost on interruption.
        torch.save(history, history_path)

        current_fmax = val_metrics["Fmax"]
        if current_fmax > best_fmax:
            best_fmax = current_fmax
            best_epoch = epoch  # FIX 5
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"Fmax={val_metrics['Fmax']:.4f} | "
                f"AUPR={val_metrics['AUPR']:.4f} | "
                f"Smin_raw={val_metrics['Smin_raw']:.4f}"
            )

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")  # FIX 5
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path))

    return history


def build_sequence_only_loaders(
    train_esm_shard_dir,
    val_esm_shard_dir,
    train_manifest_path,
    val_manifest_path,
    train_keep_ids_for_aspect,
    val_keep_ids_for_aspect,
    train_label_to_indices,
    val_label_to_indices,
    go_terms,
    batch_size: int = 16,
    seed: int = 42,
):
    train_dataset = SequenceOnlyESMShardDataset(
        shard_dir=train_esm_shard_dir,
        manifest_path=train_manifest_path,
        keep_ids=train_keep_ids_for_aspect,
    )

    val_dataset = SequenceOnlyESMShardDataset(
        shard_dir=val_esm_shard_dir,
        manifest_path=val_manifest_path,
        keep_ids=val_keep_ids_for_aspect,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=HybridBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            active_shards=3,
            lookahead_factor=2,
            drop_last=True,
            seed=seed,
        ),
        collate_fn=make_sequence_only_collate_fn(
            global_idx_to_label=train_dataset.global_idx_to_label,
            label_to_indices=train_label_to_indices,
            num_go_terms=len(go_terms),
        ),
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=HybridBatchSampler(
            dataset=val_dataset,
            batch_size=batch_size,
            active_shards=3,
            lookahead_factor=2,
            drop_last=False,
            seed=seed,
        ),
        collate_fn=make_sequence_only_collate_fn(
            global_idx_to_label=val_dataset.global_idx_to_label,
            label_to_indices=val_label_to_indices,
            num_go_terms=len(go_terms),
        ),
        num_workers=0,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader


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
