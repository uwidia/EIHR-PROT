"""Lossless, versioned full-output caches for checkpoint-free evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "eihr_prediction_cache.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_ids_hash(values: list[str] | np.ndarray) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode()).hexdigest()


@dataclass(frozen=True)
class PredictionCache:
    protein_ids: np.ndarray
    go_terms: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    eligibility: np.ndarray
    metadata: dict[str, Any]
    gate_weights: np.ndarray | None = None

    def validate(self) -> None:
        if self.probabilities.ndim != 2:
            raise ValueError("Cache probabilities must have shape (N, C)")
        n, c = self.probabilities.shape
        if self.labels.shape != (n, c) or self.protein_ids.shape != (n,) or self.go_terms.shape != (c,):
            raise ValueError("Cache arrays do not have compatible N/C dimensions")
        if self.eligibility.shape != (n,):
            raise ValueError("Cache eligibility must have one value per protein")
        if len(set(self.protein_ids.tolist())) != n or len(set(self.go_terms.tolist())) != c:
            raise ValueError("Prediction cache has duplicate protein IDs or GO terms")
        if not np.isfinite(self.probabilities).all() or not np.isfinite(self.labels).all():
            raise ValueError("Prediction cache has non-finite matrix values")
        if self.gate_weights is not None:
            if self.gate_weights.shape != (n, 2) or not np.isfinite(self.gate_weights).all():
                raise ValueError("Gate weights must be finite (N, 2)")
            if np.any(self.gate_weights < -1e-6) or np.any(self.gate_weights > 1.000001):
                raise ValueError("Gate weights are outside [0, 1]")
            if not np.allclose(self.gate_weights.sum(1), 1., atol=1e-5, rtol=1e-5):
                raise ValueError("Gate-weight rows must sum to one")


def write_prediction_cache(cache: PredictionCache, path: Path, *, overwrite: bool = False) -> Path:
    """Write ``.npz`` plus JSON provenance atomically; never overwrite implicitly."""
    cache.validate()
    path = path.with_suffix(".npz")
    meta_path = path.with_suffix(".json")
    if (path.exists() or meta_path.exists()) and not overwrite:
        raise FileExistsError(f"Cache already exists: {path}; use explicit overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(cache.metadata)
    metadata.update({"schema_version": SCHEMA_VERSION, "n_proteins": len(cache.protein_ids), "n_go_terms": len(cache.go_terms),
                     "protein_ids_sha256": canonical_ids_hash(cache.protein_ids), "go_terms_sha256": canonical_ids_hash(cache.go_terms),
                     "has_gate_weights": cache.gate_weights is not None})
    arrays: dict[str, Any] = {"protein_ids": cache.protein_ids.astype(str), "go_terms": cache.go_terms.astype(str),
        "probabilities": cache.probabilities, "labels": cache.labels, "eligibility": cache.eligibility.astype(bool)}
    if cache.gate_weights is not None:
        arrays["gate_weights"] = cache.gate_weights
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", mode="w", delete=False) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            temp_meta = Path(handle.name)
        os.replace(temp_meta, meta_path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_prediction_cache(path: Path) -> PredictionCache:
    path = path.with_suffix(".npz")
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Expected cache and JSON metadata: {path}")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported or stale cache schema: {metadata.get('schema_version')!r}")
    with np.load(path, allow_pickle=False) as data:
        cache = PredictionCache(data["protein_ids"].astype(str), data["go_terms"].astype(str), data["probabilities"], data["labels"],
            data["eligibility"].astype(bool), metadata, data["gate_weights"] if "gate_weights" in data.files else None)
    cache.validate()
    if metadata.get("protein_ids_sha256") != canonical_ids_hash(cache.protein_ids):
        raise ValueError("Cache protein-ID checksum mismatch")
    return cache


def align_caches(reference: PredictionCache, comparator: PredictionCache) -> tuple[PredictionCache, PredictionCache]:
    """Exact ID/GO alignment. Different sets are an error, never an intersection."""
    reference.validate()
    comparator.validate()
    if set(reference.protein_ids) != set(comparator.protein_ids):
        raise ValueError("Caches have missing/extra protein IDs; refusing to shrink cohort")
    if set(reference.go_terms) != set(comparator.go_terms):
        raise ValueError("Caches have different GO vocabularies; explicit harmonization is required")
    row_lookup = {x: i for i, x in enumerate(comparator.protein_ids)}
    col_lookup = {x: i for i, x in enumerate(comparator.go_terms)}
    rows = np.asarray([row_lookup[x] for x in reference.protein_ids])
    cols = np.asarray([col_lookup[x] for x in reference.go_terms])
    other = PredictionCache(reference.protein_ids.copy(), reference.go_terms.copy(), comparator.probabilities[rows][:, cols], comparator.labels[rows][:, cols], comparator.eligibility[rows], comparator.metadata, comparator.gate_weights[rows] if comparator.gate_weights is not None else None)
    if not np.array_equal(reference.labels, other.labels):
        raise ValueError("Aligned caches disagree on ground-truth labels")
    if not np.array_equal(reference.eligibility, other.eligibility):
        raise ValueError("Aligned caches disagree on annotation eligibility")
    return reference, other
