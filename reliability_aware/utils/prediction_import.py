"""Explicit importer for external full prediction matrices (DeepGOPlus/TALE+)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from reliability_aware.utils.prediction_cache import PredictionCache, read_prediction_cache, write_prediction_cache


def import_long_csv(*, csv_path: Path, reference_cache_path: Path, output_path: Path,
                    model_id: str, overwrite: bool = False) -> Path:
    """Import exact long schema: protein_id, go_term, probability.

    Every reference protein/term must occur exactly once; zero filling or term
    intersections are intentionally forbidden.
    """
    reference = read_prediction_cache(reference_cache_path)
    row = {value: i for i, value in enumerate(reference.protein_ids)}
    col = {value: i for i, value in enumerate(reference.go_terms)}
    matrix = np.empty(reference.probabilities.shape, dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_id", "go_term", "probability"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("External CSV must have protein_id,go_term,probability columns")
        for item in reader:
            if item["protein_id"] not in row or item["go_term"] not in col:
                raise ValueError("External CSV contains an unknown protein ID or GO term")
            key = (row[item["protein_id"]], col[item["go_term"]])
            if key in seen:
                raise ValueError("External CSV has duplicate protein_id/go_term rows")
            matrix[key] = float(item["probability"]); seen.add(key)
    if len(seen) != matrix.size or not np.isfinite(matrix).all():
        raise ValueError("External CSV must cover every reference protein/GO cell exactly once")
    return write_prediction_cache(PredictionCache(reference.protein_ids, reference.go_terms, matrix,
        reference.labels, reference.eligibility, {"dataset": reference.metadata.get("dataset"), "model_id": model_id,
        "import_contract": "long CSV exact matrix aligned to reference cache", "reference_cache": str(reference_cache_path),
        "train_annotations": reference.metadata.get("train_annotations")}), output_path, overwrite=overwrite)
