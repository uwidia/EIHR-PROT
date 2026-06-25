from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch

from reliability_aware.utils.cafa_metrics import evaluate_cafa
from reliability_aware.utils.diamond_homology import (
    DiamondSearchConfig,
    _parse_diamond_hits,
    _retain_valid_hits,
)


def _load_subject_go_index(path: str | Path) -> dict[str, list[int]]:
    payload = json.loads(Path(path).read_text())
    return {str(label): [int(i) for i in indices] for label, indices in payload.items()}


def compute_diamond_score_probabilities(
    *,
    query_labels: Sequence[str],
    go_terms: Sequence[str],
    diamond_hits_path: str | Path,
    subject_go_index_path: str | Path,
    search_config: DiamondSearchConfig | None = None,
    exclude_self_hits: bool = False,
) -> torch.Tensor:
    """Compute DiamondScore probabilities using bitscore-only normalization.

    S(q, f) = sum_s bitscore(q,s) * I(f in T_s) / sum_s bitscore(q,s)
    where retained hits s are DIAMOND hits after the repository's filtering policy.
    """
    config = search_config or DiamondSearchConfig()
    hits_by_query = _parse_diamond_hits(diamond_hits_path)
    subject_to_indices = _load_subject_go_index(subject_go_index_path)

    num_go_terms = len(go_terms)
    y_prob = torch.zeros((len(query_labels), num_go_terms), dtype=torch.float32)

    for row, query_id in enumerate(query_labels):
        hits = _retain_valid_hits(
            hits_by_query.get(query_id, []),
            config,
            exclude_self=exclude_self_hits,
            query_id=query_id,
        )
        if not hits:
            continue

        denom = float(sum(max(hit.bitscore, 0.0) for hit in hits))
        if denom <= 0.0:
            continue

        prior = y_prob[row]
        for hit in hits:
            bitscore = max(hit.bitscore, 0.0)
            if bitscore <= 0.0:
                continue
            weight = bitscore / denom
            go_indices = subject_to_indices.get(hit.sseqid, [])
            if not go_indices:
                continue
            prior[go_indices] += weight

    y_prob.clamp_(0.0, 1.0)
    return y_prob


def evaluate_diamond_score(
    *,
    y_true: torch.Tensor,
    query_labels: Sequence[str],
    go_terms: Sequence[str],
    go_aspect: str,
    obo_path: str | Path,
    train_annotations: Iterable[Iterable[str]],
    diamond_hits_path: str | Path,
    subject_go_index_path: str | Path,
    search_config: DiamondSearchConfig | None = None,
) -> tuple[dict[str, float | int | str], list[dict[str, float]], torch.Tensor]:
    y_prob = compute_diamond_score_probabilities(
        query_labels=query_labels,
        go_terms=go_terms,
        diamond_hits_path=diamond_hits_path,
        subject_go_index_path=subject_go_index_path,
        search_config=search_config,
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


__all__ = [
    "compute_diamond_score_probabilities",
    "evaluate_diamond_score",
]
