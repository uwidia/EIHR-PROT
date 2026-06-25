from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence

import torch

from reliability_aware.utils.cafa_metrics import evaluate_cafa


def compute_naive_probabilities(
    *,
    n_queries: int,
    go_terms: Sequence[str],
    train_label_to_indices: dict[str, list[int]],
) -> torch.Tensor:
    """Compute naive baseline probabilities S(p, f) = N_f / N_total."""
    num_go_terms = len(go_terms)
    n_total = len(train_label_to_indices)
    if n_total <= 0:
        return torch.zeros((n_queries, num_go_terms), dtype=torch.float32)

    counts = torch.zeros(num_go_terms, dtype=torch.float32)
    for indices in train_label_to_indices.values():
        if indices:
            counts[indices] += 1.0

    probs = counts / float(n_total)
    return probs.unsqueeze(0).repeat(n_queries, 1)


def evaluate_naive(
    *,
    y_true: torch.Tensor,
    go_terms: Sequence[str],
    go_aspect: str,
    obo_path: str,
    train_annotations: Iterable[Iterable[str]],
    train_label_to_indices: dict[str, list[int]],
) -> tuple[dict[str, float | int | str], list[dict[str, float]], torch.Tensor]:
    y_prob = compute_naive_probabilities(
        n_queries=int(y_true.shape[0]),
        go_terms=go_terms,
        train_label_to_indices=train_label_to_indices,
    )
    metrics, curve = evaluate_cafa(
        y_true=y_true,
        y_prob=y_prob,
        go_terms=go_terms,
        go_aspect=go_aspect,
        obo_path=obo_path,
        train_annotations=train_annotations,
    )
    return metrics, curve, y_prob


__all__ = ["compute_naive_probabilities", "evaluate_naive"]
