"""Exact protein bootstrap with prepared CAFA contributions.

No B×N×C array is formed: duplicated sampled proteins become multiplicities.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from goatools.obo_parser import GODag

from reliability_aware.utils.cafa_metrics import (
    ASPECT_NAMESPACES, ROOT_TERMS, _propagate_and_filter,
    compute_deepgoplus_ic, matrix_to_annotation_sets, propagate_annotation_sets,
)

THRESHOLDS = np.arange(1, 101, dtype=np.float64) / 100.0


@dataclass(frozen=True)
class ResamplingPlan:
    indices: np.ndarray
    metadata: dict

    @classmethod
    def create(cls, *, dataset: str, aspect: str, cohort_name: str,
               protein_ids: Iterable[str], n_resamples: int = 10000,
               seed: int = 42) -> "ResamplingPlan":
        ids = list(map(str, protein_ids))
        if not ids:
            raise ValueError("Cannot create a resampling plan for an empty cohort")
        key = {"dataset": dataset, "aspect": aspect, "cohort": cohort_name,
               "protein_ids_sha256": hashlib.sha256("\0".join(ids).encode()).hexdigest(),
               "seed": seed, "n_resamples": n_resamples, "n": len(ids),
               "rng": "numpy.random.default_rng/PCG64", "numpy_version": np.__version__}
        indices = np.random.default_rng(seed).integers(0, len(ids), size=(n_resamples, len(ids)), dtype=np.int32)
        metadata = dict(key)
        metadata["plan_checksum"] = hashlib.sha256(indices.tobytes()).hexdigest()
        return cls(indices, metadata)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, indices=self.indices, metadata=json.dumps(self.metadata, sort_keys=True))

    @classmethod
    def load_or_create(cls, path: Path, **kwargs: object) -> "ResamplingPlan":
        expected = cls.create(**kwargs)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                loaded = cls(data["indices"], json.loads(str(data["metadata"])))
            if loaded.metadata != expected.metadata:
                raise ValueError(f"Resampling-plan configuration/checksum mismatch: {path}")
            return loaded
        expected.save(path)
        return expected


@dataclass
class PreparedCafaEvaluator:
    # Rows are thresholds; columns are already-frozen eligible proteins.
    precision: np.ndarray
    predicted: np.ndarray
    recall: np.ndarray
    ru: np.ndarray
    mi: np.ndarray
    metadata: dict

    @classmethod
    def prepare(cls, *, y_true: np.ndarray, y_prob: np.ndarray, go_terms: list[str],
                go_aspect: str, obo_path: Path,
                train_annotations: Iterable[Iterable[str]]) -> tuple["PreparedCafaEvaluator", np.ndarray]:
        if y_true.shape != y_prob.shape:
            raise ValueError("y_true and y_prob must have identical shape")
        aspect = go_aspect.upper()
        if aspect not in ASPECT_NAMESPACES:
            raise ValueError("go_aspect must be BP, MF, or CC")
        dag = GODag(str(obo_path), optional_attrs={"relationship"}, prt=None)
        namespace = ASPECT_NAMESPACES[aspect]
        term_sets = [_propagate_and_filter([go], dag, namespace, remove_root=ROOT_TERMS[aspect]) for go in go_terms]
        truths_all = propagate_annotation_sets(matrix_to_annotation_sets(y_true, go_terms), go_dag=dag, go_aspect=aspect, remove_root=True)
        eligible = np.asarray([bool(x) for x in truths_all], dtype=bool)
        truths = [value for value in truths_all if value]
        probs = y_prob[eligible]
        train = propagate_annotation_sets(train_annotations, go_dag=dag, go_aspect=aspect, remove_root=False)
        ic = compute_deepgoplus_ic(train, go_dag=dag, go_aspect=aspect)
        shape = (len(THRESHOLDS), len(truths))
        precision = np.zeros(shape, dtype=np.float64)
        predicted = np.zeros(shape, dtype=np.float64)
        recall = np.zeros(shape, dtype=np.float64)
        ru = np.zeros(shape, dtype=np.float64)
        mi = np.zeros(shape, dtype=np.float64)
        for t, threshold in enumerate(THRESHOLDS):
            for i, truth in enumerate(truths):
                pred = set().union(*(term_sets[j] for j in np.flatnonzero(probs[i] >= threshold)))
                tp, fp, fn = pred & truth, pred - truth, truth - pred
                if pred:
                    predicted[t, i] = 1.0
                    precision[t, i] = len(tp) / len(pred)
                recall[t, i] = len(tp) / len(truth)
                ru[t, i] = sum(ic.get(go, 0.0) for go in fn)
                mi[t, i] = sum(ic.get(go, 0.0) for go in fp)
        return cls(precision, predicted, recall, ru, mi,
                   {"thresholds": THRESHOLDS.tolist(), "ic_source": "train", "go_aspect": aspect}), eligible

    def metrics(self, multiplicities: np.ndarray) -> dict[str, float]:
        n = int(multiplicities.sum())
        if n == 0:
            return {"Fmax": math.nan, "AUPR": math.nan, "Smin": math.nan}
        predicted_count = self.predicted @ multiplicities
        precision = (self.precision @ multiplicities) / np.where(predicted_count, predicted_count, 1.0)
        precision[predicted_count == 0] = 0.0
        recall = (self.recall @ multiplicities) / n
        fscore = np.divide(2 * precision * recall, precision + recall,
                           out=np.zeros_like(precision), where=(precision + recall) > 0)
        s = np.hypot((self.ru @ multiplicities) / n, (self.mi @ multiplicities) / n)
        order = np.argsort(recall, kind="stable")
        return {"Fmax": float(np.max(fscore)), "AUPR": float(np.trapezoid(precision[order], recall[order])), "Smin": float(np.min(s))}

    def bootstrap(self, plan: ResamplingPlan, batch_size: int = 64) -> dict[str, np.ndarray]:
        n = self.precision.shape[1]
        if plan.indices.shape[1] != n:
            raise ValueError("Resampling plan does not match frozen cohort")
        result = {metric: np.empty(len(plan.indices), dtype=np.float64) for metric in ("Fmax", "AUPR", "Smin")}
        for begin in range(0, len(plan.indices), batch_size):
            block = plan.indices[begin:begin + batch_size]
            counts = np.zeros((len(block), n), dtype=np.int32)
            np.add.at(counts, (np.arange(len(block))[:, None], block), 1)
            for offset, count in enumerate(counts):
                for metric, value in self.metrics(count).items():
                    result[metric][begin + offset] = value
        return result


def percentile_interval(values: np.ndarray, confidence_level: float = .95) -> tuple[float, float]:
    if not np.isfinite(values).all():
        raise ValueError("Non-finite bootstrap replicate; no silent retry/drop is allowed")
    alpha = (1.0 - confidence_level) / 2.0
    return tuple(map(float, np.percentile(values, [100 * alpha, 100 * (1 - alpha)], method="linear")))


def summary(point: float, replicates: np.ndarray, confidence_level: float = .95) -> dict[str, float | int]:
    low, high = percentile_interval(replicates, confidence_level)
    return {"point": float(point), "ci_lower": low, "ci_upper": high,
            "n_requested": len(replicates), "n_valid": int(np.isfinite(replicates).sum())}
