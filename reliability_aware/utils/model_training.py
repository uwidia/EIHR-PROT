"""
Module contains wrappers for model training across the core model and all ablations.

run_seq_only_training: Wrapper for sequence-only ablation
"""

from __future__ import annotations
from copy import deepcopy
import copy
from pathlib import Path
from typing import Literal
import torch
import wandb
from reliability_aware.utils.parser import get_protein_info

from reliability_aware.utils.losses import (
    compute_pos_weight_from_label_indices,
    weighted_bce_on_probs,
    hierarchy_loss,
)
from reliability_aware.utils.go_term_extraction import (
    build_subject_go_index,
    build_go_annotations_list,
    build_child_parent_idx_pairs,
)
from sklearn.metrics import average_precision_score
from reliability_aware.utils.metrics import fmax_score, smin_score

from dataclasses import dataclass


@dataclass
class GOAnnotationData:
    go_terms: list
    go_term_to_idx: dict
    num_go_terms: int
    child_parent_pairs: torch.Tensor
    ic: torch.Tensor
    train_label_to_indices: dict
    val_label_to_indices: dict
    train_keep_ids: set
    val_keep_ids: set


def run_model_training(
    promising_hparams: list[dict],
    *,
    train_loader,
    val_loader,
    train_keep_ids_for_aspect,
    train_label_to_indices,
    go_terms,
    child_parent_pairs,
    ic,
    build_model_fn,
    fit_function,
    device,
    final_epochs: int = 50,
    patience: int = 10,
    base_dir: str | Path = "runs/final",
    use_wandb: bool = False,
    wandb_project: str = "seq_homology_reliability-aware-pfp",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    ablation: str | None = None,
    run_type: str = "full_training",
) -> dict | None:
    """
    Trains each promising hyperparameter configuration to convergence.
    Returns the record for the best final run.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best_score = -1.0
    best_run = None

    for run_id, hparams in enumerate(promising_hparams):
        run_dir = base_dir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[Final] Run {run_id+1}/{len(promising_hparams)}  hparams={hparams}")
        print(f"{'='*60}")

        pos_weight = compute_pos_weight_from_label_indices(
            label_to_indices=train_label_to_indices,
            num_go_terms=len(go_terms),
            train_ids=train_keep_ids_for_aspect,
            cap=hparams["pos_weight_cap"],
        )

        model, optimizer = build_model_fn(hparams, go_terms, device)

        history = fit_function(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            pos_weight=pos_weight.to(device),
            child_parent_pairs=child_parent_pairs.to(device),
            ic=ic,
            device=device,
            lambda_hier=hparams["lambda_hier"],
            num_epochs=final_epochs,
            patience=patience,
            out_dir=run_dir,
            hparams=hparams,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            wandb_run_name=f"{ablation}_final_{run_id:03d}",
            wandb_config={
                "ablation": ablation,
                "run_type": run_type,
                "run_id": run_id,
                "go_terms": len(go_terms),
                **hparams,
            },
        )

        record = build_record(run_id, history, hparams, id_key="run_id")
        best_score, best_run = save_and_track_best(
            record=record,
            records=results,
            best_score=best_score,
            best_record=best_run,
            save_path=run_dir / "final_meta.pt",
            best_save_path=base_dir / "best_final_run.pt",
        )

        print(f"[Run {run_id+1:03d}] val_Fmax={record['score']:.4f}")
        if best_run is record:
            print(f"  *** New best: {best_score:.4f} ***")

    print(f"\n[Final training complete] Best val_Fmax: {best_score:.4f}")
    if best_run:
        print(f"Best run hparams: {best_run['hparams']}")

    return best_run


def build_go_annotation_data(
    train_dataset,
    val_dataset,
    go_annotation_path,
    obo_path,
    go_aspect,
    device,
) -> GOAnnotationData:
    train_protein_info = get_protein_info(train_dataset)
    val_protein_info = get_protein_info(val_dataset)

    train_ids = {protein["full_id"] for protein in train_protein_info}
    val_ids = {protein["full_id"] for protein in val_protein_info}

    train_label_to_go_terms, go_terms = build_go_annotations_list(
        go_annotation_path=go_annotation_path,
        obo_path=obo_path,
        go_aspect=go_aspect,
        keep_ids=train_ids,
        remove_root_term=True,
        min_term_freq=None,
    )

    val_label_to_go_terms, _ = build_go_annotations_list(
        go_annotation_path=go_annotation_path,
        obo_path=obo_path,
        go_aspect=go_aspect,
        keep_ids=val_ids,
        remove_root_term=True,
        min_term_freq=None,
    )

    child_parent_pairs = build_child_parent_idx_pairs(
        obo_path=obo_path,
        go_terms=go_terms,
    )

    go_term_to_idx = {go: i for i, go in enumerate(go_terms)}

    train_label_to_indices = build_subject_go_index(
        train_label_to_go_terms, go_term_to_idx
    )
    val_label_to_indices = build_subject_go_index(val_label_to_go_terms, go_term_to_idx)

    train_keep_ids = {label for label, idxs in train_label_to_indices.items() if idxs}
    val_keep_ids = {label for label, idxs in val_label_to_indices.items() if idxs}

    ic = compute_ic_from_label_indices(
        label_to_indices=train_label_to_indices,
        num_go_terms=len(go_terms),
        train_ids=train_keep_ids,
    ).to(device)

    return GOAnnotationData(
        go_terms=go_terms,
        go_term_to_idx=go_term_to_idx,
        num_go_terms=len(go_terms),
        child_parent_pairs=child_parent_pairs,
        ic=ic,
        train_label_to_indices=train_label_to_indices,
        val_label_to_indices=val_label_to_indices,
        train_keep_ids=train_keep_ids,
        val_keep_ids=val_keep_ids,
    )


def compute_ic_from_label_indices(
    label_to_indices: dict[str, list[int]],
    num_go_terms: int,
    train_ids: set[str],
) -> torch.Tensor:
    counts = torch.zeros(num_go_terms, dtype=torch.float32)
    valid_ids = [label for label in train_ids if label in label_to_indices]
    n = len(valid_ids)

    for label in valid_ids:
        idxs = label_to_indices[label]
        if idxs:
            counts[idxs] += 1.0

    p = (counts + 1.0) / (n + 2.0)
    return -torch.log(p.clamp_min(1e-12))


def save_and_track_best(
    record: dict,
    records: list,
    best_score: float,
    best_record: dict | None,
    save_path: Path,
    best_save_path: Path,
) -> tuple[float, dict | None]:
    """
    Saves a trial/run record, appends to records list, and tracks the best.
    Returns updated (best_score, best_record).
    """
    torch.save(record, save_path)
    records.append(record)

    if record["score"] > best_score:
        best_score = record["score"]
        best_record = record
        torch.save(record, best_save_path)

    return best_score, best_record


def build_record(
    trial_or_run_id: int, history: dict, hparams: dict, id_key: str = "trial"
) -> dict:
    """
    Builds a standard record dict from a completed training history.
    """
    return {
        id_key: trial_or_run_id,
        "score": max(history["val_Fmax"]),
        "metrics": {
            "Fmax": history["val_Fmax"],
            "AUPR": history["val_AUPR"],
            "Smin_raw": history["val_Smin_raw"],
        },
        "hparams": deepcopy(hparams),
    }


def train_one_epoch(
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

    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model_forward_from_batch(model, batch)
        probs = outputs["probs"]

        bce_loss = weighted_bce_on_probs(
            probs=probs,
            targets=batch["targets"],
            pos_weight=pos_weight,
        )
        hier_loss = hierarchy_loss(
            fused_probs=probs,
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


def model_forward_from_batch(model, batch):
    inputs = {}

    if batch.get("padded") is not None:
        inputs["padded"] = batch["padded"]
    if batch.get("mask") is not None:
        inputs["mask"] = batch["mask"]
    if batch.get("graph_batch") is not None:
        inputs["graph_batch"] = batch["graph_batch"]
    if batch.get("homology_scores") is not None:
        inputs["homology_scores"] = batch["homology_scores"]
    if batch.get("gate_features") is not None:
        inputs["gate_features"] = batch["gate_features"]

    return model(**inputs)


def move_batch_to_device(batch: dict, device):
    for key in [
        "padded",
        "mask",
        "graph_batch",
        "homology_scores",
        "gate_features",
        "targets",
    ]:
        if batch.get(key) is not None:
            batch[key] = batch[key].to(device)

    return batch


def _history_key(metric_name: str) -> str:
    """
    Converts validation metric names into history keys.

    Examples:
        "val_loss" -> "val_loss"
        "Fmax" -> "val_Fmax"
        "AUPR" -> "val_AUPR"
        "mean_neural_gate" -> "val_mean_neural_gate"
    """
    if metric_name.startswith("val_"):
        return metric_name
    return f"val_{metric_name}"


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    pos_weight,
    child_parent_pairs,
    ic,
    device,
    *,
    lambda_hier: float = 0.01,
    num_epochs: int = 100,
    patience: int = 10,
    out_dir: str | Path = "runs/model",
    hparams: dict | None = None,
    checkpoint_extra: dict | None = None,
    use_wandb: bool = False,
    wandb_project: str = "seq_homology_reliability-aware-pfp",
    wandb_entity: str | None = None,
    wandb_mode: Literal["online", "offline", "disabled", "shared"] = "online",
    wandb_run_name: str | None = None,
    wandb_config: dict | None = None,
):
    """
    Generic fit loop for all ablations.

    Requirements:
      - train_loader and val_loader return standardized dict batches.
      - model.forward returns a dict with outputs["probs"].
      - evaluate_fn returns a dict containing at least "Fmax".
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
    }

    best_fmax = -1.0
    best_epoch = -1
    bad_epochs = 0

    best_path = out_dir / "best_model.pt"
    history_path = out_dir / "history.pt"

    checkpoint_extra = checkpoint_extra or {}

    wandb_run = None

    if use_wandb:

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=wandb_config or hparams,
            mode=wandb_mode,
            reinit=True,
        )

    for epoch in range(1, num_epochs + 1):
        print(f"Currently running epoch {epoch}:")
        if hasattr(train_loader, "batch_sampler") and hasattr(
            train_loader.batch_sampler, "set_epoch"
        ):
            train_loader.batch_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            pos_weight=pos_weight,
            child_parent_pairs=child_parent_pairs,
            lambda_hier=lambda_hier,
            device=device,
        )

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            pos_weight=pos_weight,
            child_parent_pairs=child_parent_pairs,
            lambda_hier=lambda_hier,
            ic=ic,
            device=device,
        )

        if wandb_run is not None:
            wandb_log = {
                "epoch": epoch,
                "train_loss": train_loss,
            }

            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    wandb_log[f"val/{key}"] = value

            wandb_run.log(wandb_log, step=epoch)

        history["train_loss"].append(train_loss)

        for metric_name, metric_value in val_metrics.items():
            key = _history_key(metric_name)
            history.setdefault(key, []).append(metric_value)

        torch.save(history, history_path)

        current_fmax = val_metrics["Fmax"]

        if current_fmax > best_fmax:
            best_fmax = current_fmax
            best_epoch = epoch
            bad_epochs = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "val_metrics": val_metrics,
                "train_loss": train_loss,
                "hparams": hparams,
            }
            checkpoint.update(checkpoint_extra)

            torch.save(checkpoint, best_path)

        else:
            bad_epochs += 1

        message = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"Fmax={val_metrics['Fmax']:.4f} | "
            f"AUPR={val_metrics['AUPR']:.4f} | "
            f"Smin_raw={val_metrics['Smin_raw']:.4f}"
        )

        if "mean_neural_gate" in val_metrics:
            message += (
                f" | gate_n={val_metrics['mean_neural_gate']:.3f}"
                f" | gate_h={val_metrics['mean_homology_gate']:.3f}"
            )

        print(message)

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if wandb_run is not None:
        wandb_run.finish()

    return history


@torch.no_grad()
def evaluate_model(
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
    all_gate_weights = []

    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        outputs = model_forward_from_batch(model, batch)
        probs = outputs["probs"]
        targets = batch["targets"]

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

        if "gate_weights" in outputs:
            all_gate_weights.append(outputs["gate_weights"].detach().cpu())

    if not all_probs:
        raise RuntimeError("No batches were produced by the validation loader.")

    y_prob = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    fmax = fmax_score(y_true, y_prob)
    smin = smin_score(y_true, y_prob, ic)
    aupr = average_precision_score(y_true.numpy().ravel(), y_prob.numpy().ravel())

    metrics = {
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

    if all_gate_weights:
        gate_weights = torch.cat(all_gate_weights, dim=0)
        metrics["mean_neural_gate"] = float(gate_weights[:, 0].mean().item())
        metrics["mean_homology_gate"] = float(gate_weights[:, 1].mean().item())

    return metrics
