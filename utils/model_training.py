"""
Module contains wrappers for model training across the core model and all ablations.

run_seq_only_training: Wrapper for sequence-only ablation
"""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import torch
from utils.parser import get_protein_info

from models.sequence_only_ablation import (
    SequenceOnlyProteinFunctionModel,
    build_sequence_only_loaders,
    fit_sequence_only,
)
from utils.losses import compute_pos_weight_from_label_indices
from utils.go_term_extraction import (
    build_subject_go_index,
    build_go_annotations_list,
    build_child_parent_idx_pairs,
)
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


# Sequence Only Training Wrapper
def run_seq_only_ablation_training(
    promising_hparams: list[dict],
    *,
    train_esm_shard_dir,
    val_esm_shard_dir,
    train_manifest_path,
    val_manifest_path,
    train_keep_ids_for_aspect,
    val_keep_ids_for_aspect,
    train_label_to_indices,
    val_label_to_indices,
    go_terms,
    child_parent_pairs,
    ic,
    device,
    final_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 16,
    base_dir: str | Path = "runs/seq_only_final",
) -> dict:
    """
    Takes the specified list of promising hyperparameter combinations and trains each to
    convergence. Returns the record for the best final run.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    _, _, train_loader, val_loader = build_sequence_only_loaders(
        train_esm_shard_dir=train_esm_shard_dir,
        val_esm_shard_dir=val_esm_shard_dir,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        train_keep_ids_for_aspect=train_keep_ids_for_aspect,
        val_keep_ids_for_aspect=val_keep_ids_for_aspect,
        train_label_to_indices=train_label_to_indices,
        val_label_to_indices=val_label_to_indices,
        go_terms=go_terms,
        batch_size=batch_size,
    )

    results = []
    best_score = -1.0
    best_run = {}

    for run_id, hparams in enumerate(promising_hparams):
        run_dir = base_dir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[Final] Run {run_id+1}/{len(promising_hparams)}")
        print(f"{'='*60}")

        pos_weight = compute_pos_weight_from_label_indices(
            label_to_indices=train_label_to_indices,
            num_go_terms=len(go_terms),
            train_ids=train_keep_ids_for_aspect,
            cap=hparams["pos_weight_cap"],
        )

        model = SequenceOnlyProteinFunctionModel(
            num_go_terms=len(go_terms),
            attn_hidden_dim=hparams["attn_hidden_dim"],
            attn_dropout=hparams["attn_dropout"],
            head_dropout=hparams["head_dropout"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=1e-4,
        )

        history = fit_sequence_only(
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
        )

        score = max(history["val_Fmax"])
        record = {
            "run_id": run_id,
            "score": score,
            "history": history,
            "hparams": deepcopy(hparams),
        }

        torch.save(record, run_dir / "final_meta.pt")
        results.append(record)

        print(f"[Run {run_id+1:03d}] val_Fmax={score:.4f}")

        if score > best_score:
            best_score = score
            best_run = record
            torch.save(record, base_dir / "best_final_run.pt")
            print(f"  *** New best final run: {best_score:.4f} ***")

    print(f"\n[Final training complete] Best val_Fmax: {best_score:.4f}")
    print(f"Best run hparams: {best_run['hparams']}")

    return best_run
