from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from goatools.obo_parser import GODag

from reliability_aware.utils.go_term_extraction import ROOT_TERMS


LOGGER = logging.getLogger(__name__)

ASPECT_NAMESPACES = {
    "BP": "biological_process",
    "MF": "molecular_function",
    "CC": "cellular_component",
}


def _as_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _propagate_and_filter(
    terms: Iterable[str],
    go_dag: GODag,
    namespace: str,
    *,
    remove_root: str | None,
) -> set[str]:
    propagated: set[str] = set()
    for go_id in terms:
        if go_id not in go_dag:
            continue
        propagated.add(go_id)
        propagated.update(go_dag[go_id].get_all_parents())

    filtered = {
        go_id
        for go_id in propagated
        if go_id in go_dag and go_dag[go_id].namespace == namespace
    }
    if remove_root is not None:
        filtered.discard(remove_root)
    return filtered


def matrix_to_annotation_sets(
    labels: torch.Tensor | np.ndarray,
    go_terms: Sequence[str],
) -> list[set[str]]:
    labels_np = _as_numpy(labels)
    if labels_np.ndim != 2:
        raise ValueError(f"Expected a 2D label matrix, got shape {labels_np.shape}")
    if labels_np.shape[1] != len(go_terms):
        raise ValueError(
            f"GO vocabulary has {len(go_terms)} terms but labels have "
            f"{labels_np.shape[1]} columns"
        )
    return [
        {go_terms[index] for index in np.flatnonzero(row)}
        for row in labels_np
    ]


def propagate_annotation_sets(
    annotations: Iterable[Iterable[str]],
    *,
    go_dag: GODag,
    go_aspect: str,
    remove_root: bool,
) -> list[set[str]]:
    aspect = go_aspect.upper()
    if aspect not in ASPECT_NAMESPACES:
        raise ValueError("go_aspect must be one of: BP, MF, CC")
    root = ROOT_TERMS[aspect] if remove_root else None
    namespace = ASPECT_NAMESPACES[aspect]
    return [
        _propagate_and_filter(terms, go_dag, namespace, remove_root=root)
        for terms in annotations
    ]


def compute_deepgoplus_ic(
    propagated_annotations: Iterable[set[str]],
    *,
    go_dag: GODag,
    go_aspect: str,
) -> dict[str, float]:
    """Compute IC(term) = log2(min direct-parent count / term count)."""
    aspect = go_aspect.upper()
    namespace = ASPECT_NAMESPACES[aspect]
    counts = Counter(
        go_id
        for annotations in propagated_annotations
        for go_id in annotations
    )

    ic: dict[str, float] = {}
    for go_id, term_count in counts.items():
        if term_count <= 0 or go_id not in go_dag:
            continue
        parent_counts = [
            counts[parent.id]
            for parent in go_dag[go_id].parents
            if parent.namespace == namespace and counts[parent.id] > 0
        ]
        min_parent_count = min(parent_counts) if parent_counts else term_count
        ic[go_id] = math.log2(min_parent_count / term_count)
    return ic


def evaluate_cafa(
    *,
    y_true: torch.Tensor | np.ndarray,
    y_prob: torch.Tensor | np.ndarray,
    go_terms: Sequence[str],
    go_aspect: str,
    obo_path: str | Path,
    train_annotations: Iterable[Iterable[str]],
) -> tuple[dict[str, float | int | str], list[dict[str, float]]]:
    """Evaluate CAFA Fmax, AUPR, and Smin using training-annotation IC."""
    y_true_np = _as_numpy(y_true)
    y_prob_np = _as_numpy(y_prob).astype(np.float64, copy=False)

    if y_true_np.shape != y_prob_np.shape:
        raise ValueError(
            f"y_true and y_prob must have identical shapes, got "
            f"{y_true_np.shape} and {y_prob_np.shape}"
        )
    if y_true_np.ndim != 2:
        raise ValueError(f"Expected (N, C) arrays, got shape {y_true_np.shape}")
    if len(go_terms) != y_prob_np.shape[1]:
        raise ValueError(
            f"len(go_terms)={len(go_terms)} does not match C={y_prob_np.shape[1]}"
        )
    aspect = go_aspect.upper()
    if aspect not in ASPECT_NAMESPACES:
        raise ValueError("go_aspect must be one of: BP, MF, CC")

    go_dag = GODag(str(obo_path), optional_attrs={"relationship"})
    prediction_term_sets = [
        _propagate_and_filter(
            [go_id],
            go_dag,
            ASPECT_NAMESPACES[aspect],
            remove_root=ROOT_TERMS[aspect],
        )
        for go_id in go_terms
    ]
    raw_true_sets = matrix_to_annotation_sets(y_true_np, go_terms)
    true_sets_all = propagate_annotation_sets(
        raw_true_sets,
        go_dag=go_dag,
        go_aspect=aspect,
        remove_root=True,
    )
    evaluated_indices = [index for index, terms in enumerate(true_sets_all) if terms]
    true_sets = [true_sets_all[index] for index in evaluated_indices]
    y_prob_np = y_prob_np[evaluated_indices]

    n_evaluated = len(true_sets)
    LOGGER.info(
        "CAFA evaluation retained %d/%d proteins with non-empty %s ground truth",
        n_evaluated,
        y_true_np.shape[0],
        aspect,
    )
    if n_evaluated == 0:
        LOGGER.warning("No proteins have non-empty %s ground truth for CAFA evaluation", aspect)

    train_propagated = propagate_annotation_sets(
        train_annotations,
        go_dag=go_dag,
        go_aspect=aspect,
        remove_root=False,
    )
    ic = compute_deepgoplus_ic(
        train_propagated,
        go_dag=go_dag,
        go_aspect=aspect,
    )

    curve: list[dict[str, float]] = []
    best_f = {
        "Fmax": 0.0,
        "threshold": 0.01,
        "precision": 0.0,
        "recall": 0.0,
    }
    best_s = {
        "Smin": math.inf,
        "threshold": 0.01,
        "ru": 0.0,
        "mi": 0.0,
    }

    for threshold in np.arange(1, 101, dtype=np.float64) / 100.0:
        precision_sum = 0.0
        precision_count = 0
        recall_sum = 0.0
        ru_sum = 0.0
        mi_sum = 0.0

        for row_index, true_terms in enumerate(true_sets):
            predicted_columns = np.flatnonzero(y_prob_np[row_index] >= threshold)
            predicted_terms = set().union(
                *(prediction_term_sets[index] for index in predicted_columns)
            )

            true_positives = predicted_terms & true_terms
            false_positives = predicted_terms - true_terms
            false_negatives = true_terms - predicted_terms

            if predicted_terms:
                precision_sum += len(true_positives) / len(predicted_terms)
                precision_count += 1
            recall_sum += len(true_positives) / len(true_terms)
            ru_sum += sum(ic.get(go_id, 0.0) for go_id in false_negatives)
            mi_sum += sum(ic.get(go_id, 0.0) for go_id in false_positives)

        if precision_count == 0:
            LOGGER.warning("No CAFA predictions at threshold %.2f", threshold)

        precision = precision_sum / precision_count if precision_count else 0.0
        recall = recall_sum / n_evaluated if n_evaluated else 0.0
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        ru = ru_sum / n_evaluated if n_evaluated else 0.0
        mi = mi_sum / n_evaluated if n_evaluated else 0.0
        semantic_distance = math.hypot(ru, mi)

        curve.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "fscore": float(fscore),
                "ru": float(ru),
                "mi": float(mi),
                "s": float(semantic_distance),
            }
        )

        if fscore > best_f["Fmax"]:
            best_f = {
                "Fmax": fscore,
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
            }
        if semantic_distance < best_s["Smin"]:
            best_s = {
                "Smin": semantic_distance,
                "threshold": float(threshold),
                "ru": ru,
                "mi": mi,
            }

    recalls = np.asarray([point["recall"] for point in curve])
    precisions = np.asarray([point["precision"] for point in curve])
    order = np.argsort(recalls, kind="stable")
    aupr = float(np.trapezoid(precisions[order], recalls[order]))

    metrics: dict[str, float | int | str] = {
        "Fmax": float(best_f["Fmax"]),
        "Fmax_threshold": float(best_f["threshold"]),
        "precision_at_Fmax": float(best_f["precision"]),
        "recall_at_Fmax": float(best_f["recall"]),
        "Smin": float(best_s["Smin"]),
        "Smin_threshold": float(best_s["threshold"]),
        "RU_at_Smin": float(best_s["ru"]),
        "MI_at_Smin": float(best_s["mi"]),
        "AUPR": aupr,
        "n_proteins_evaluated": n_evaluated,
        "n_go_terms": len(go_terms),
        "ic_source": "train",
    }
    return metrics, curve
