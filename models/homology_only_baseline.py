from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from reliability_aware.utils.losses import (
    compute_pos_weight_from_label_indices,
    hierarchy_loss,
    weighted_bce_on_probs,
)
from reliability_aware.utils.metrics import fmax_score, smin_score
from reliability_aware.utils.shard_handling import HomologyShardDataset


def _average_precision_score(y_true: torch.Tensor, y_prob: torch.Tensor) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true.numpy().ravel(), y_prob.numpy().ravel()))
    except ModuleNotFoundError:
        y_true_flat = y_true.bool().flatten()
        y_prob_flat = y_prob.float().flatten()
        order = torch.argsort(y_prob_flat, descending=True)
        sorted_true = y_true_flat[order].float()
        total_pos = sorted_true.sum()
        if total_pos.item() == 0:
            return 0.0
        precision = torch.cumsum(sorted_true, dim=0) / (
            torch.arange(1, sorted_true.numel() + 1, dtype=torch.float32)
        )
        return float((precision * sorted_true).sum().item() / total_pos.item())


def make_homology_only_collate_fn(label_to_indices, num_go_terms: int):
    def collate(batch: Sequence[dict]) -> dict:
        homology_scores = []
        targets = torch.zeros(len(batch), num_go_terms, dtype=torch.float32)
        global_indices = torch.tensor(
            [item["global_idx"] for item in batch],
            dtype=torch.long,
        )
        labels = [item["label"] for item in batch]

        for i, (item, label) in enumerate(zip(batch, labels)):
            if label not in label_to_indices:
                raise KeyError(f"Missing label in label_to_indices: {label}")

            scores = item["prior"].float()
            if scores.shape != (num_go_terms,):
                raise ValueError(
                    f"homology_scores for label={label} must have shape "
                    f"({num_go_terms},), got {tuple(scores.shape)}"
                )
            homology_scores.append(scores)

            idxs = label_to_indices[label]
            if idxs:
                targets[i, list(idxs)] = 1.0

        return {
            "probs": torch.stack(homology_scores, dim=0),
            "targets": targets,
            "global_indices": global_indices,
            "labels": labels,
        }

    return collate


@torch.no_grad()
def evaluate_homology_only(
    loader,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    ic,
    device,
):
    total_loss = 0.0
    bce_total = 0.0
    hier_total = 0.0
    n_batches = 0

    all_probs = []
    all_targets = []

    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    for batch in loader:
        probs = batch["probs"].to(device)
        targets = batch["targets"].to(device)

        bce_loss = weighted_bce_on_probs(
            probs=probs,
            targets=targets,
            pos_weight=pos_weight,
        )
        hier_loss = hierarchy_loss(
            fused_probs=probs,
            child_parent_pairs=child_parent_pairs,
        )
        loss = bce_loss + lambda_hier * hier_loss

        total_loss += loss.item()
        bce_total += bce_loss.item()
        hier_total += hier_loss.item()
        n_batches += 1

        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if not all_probs:
        raise RuntimeError("No batches were produced by the homology-only loader.")

    y_prob = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    fmax = fmax_score(y_true, y_prob)
    smin = smin_score(y_true, y_prob, ic)
    aupr = _average_precision_score(y_true, y_prob)

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "bce_loss": bce_total / max(n_batches, 1),
        "hierarchy_loss": hier_total / max(n_batches, 1),
        "Fmax": fmax["Fmax"],
        "Fmax_threshold": fmax["threshold"],
        "AUPR": aupr,
        "Smin_raw": smin["raw"]["Smin"],
        "Smin_threshold_raw": smin["raw"]["threshold"],
        "Smin_normalized": smin["normalized"]["Smin"],
        "Smin_threshold_normalized": smin["normalized"]["threshold"],
    }


def run_homology_only_evaluation(
    *,
    val_homology_shard_dir,
    val_manifest_path,
    val_label_to_indices,
    val_keep_ids_for_aspect,
    train_label_to_indices,
    train_keep_ids_for_aspect,
    go_terms,
    child_parent_pairs,
    ic,
    device,
    lambda_hier: float,
    pos_weight_cap: float = 20.0,
    batch_size: int = 64,
    out_dir: str | Path = "runs/homology_only",
    use_wandb: bool = False,
    wandb_project: str = "reliability-aware-pfp",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    wandb_run_name: str = "homology_only",
):
    pos_weight = compute_pos_weight_from_label_indices(
        label_to_indices=train_label_to_indices,
        num_go_terms=len(go_terms),
        train_ids=train_keep_ids_for_aspect,
        cap=pos_weight_cap,
    )

    val_dataset = HomologyShardDataset(
        homology_shard_dir=val_homology_shard_dir,
        manifest_path=val_manifest_path,
        keep_ids=val_keep_ids_for_aspect,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=make_homology_only_collate_fn(
            label_to_indices=val_label_to_indices,
            num_go_terms=len(go_terms),
        ),
    )

    metrics = evaluate_homology_only(
        loader=val_loader,
        pos_weight=pos_weight,
        child_parent_pairs=child_parent_pairs,
        lambda_hier=lambda_hier,
        ic=ic,
        device=device,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(metrics, out_dir / "homology_only_metrics.pt")
    with (out_dir / "homology_only_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    if use_wandb:
        import wandb

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
            name=wandb_run_name,
            config={
                "ablation": "homology_only",
                "run_type": "evaluation_only",
                "go_terms": len(go_terms),
                "lambda_hier": lambda_hier,
                "pos_weight_cap": pos_weight_cap,
                "batch_size": batch_size,
            },
        )
        wandb.log(metrics)
        run.finish()

    return metrics


__all__ = [
    "HomologyShardDataset",
    "make_homology_only_collate_fn",
    "evaluate_homology_only",
    "run_homology_only_evaluation",
]
