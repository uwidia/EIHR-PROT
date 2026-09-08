from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from reliability_aware.utils.bootstrap import PreparedCafaEvaluator, ResamplingPlan, percentile_interval
from reliability_aware.utils.cafa_metrics import evaluate_cafa
from reliability_aware.utils.prediction_cache import PredictionCache, align_caches, read_prediction_cache, write_prediction_cache

OBO = '''format-version: 1.2

[Term]
id: GO:0008150
name: root
namespace: biological_process

[Term]
id: GO:0000001
name: parent
namespace: biological_process
is_a: GO:0008150 ! root

[Term]
id: GO:0000002
name: child
namespace: biological_process
is_a: GO:0000001 ! parent
'''


def test_prepared_metrics_match_authoritative_evaluator_and_duplicates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        obo = Path(tmp) / "go.obo"; obo.write_text(OBO)
        truth = np.array([[0, 1], [1, 0]], dtype=float)
        probs = np.array([[0.1, .9], [.8, .1]], dtype=float)
        direct, _ = evaluate_cafa(y_true=truth, y_prob=probs, go_terms=["GO:0000001", "GO:0000002"], go_aspect="BP", obo_path=obo, train_annotations=[{"GO:0000002"}])
        prepared, eligible = PreparedCafaEvaluator.prepare(y_true=truth, y_prob=probs, go_terms=["GO:0000001", "GO:0000002"], go_aspect="BP", obo_path=obo, train_annotations=[{"GO:0000002"}])
        assert eligible.all()
        point = prepared.metrics(np.ones(2, dtype=int))
        for metric in ("Fmax", "AUPR", "Smin"):
            assert point[metric] == pytest.approx(direct[metric])
        duplicated = prepared.metrics(np.array([2, 1]))
        direct_dup, _ = evaluate_cafa(y_true=truth[[0, 0, 1]], y_prob=probs[[0, 0, 1]], go_terms=["GO:0000001", "GO:0000002"], go_aspect="BP", obo_path=obo, train_annotations=[{"GO:0000002"}])
        for metric in ("Fmax", "AUPR", "Smin"):
            assert duplicated[metric] == pytest.approx(direct_dup[metric])


def test_plan_is_deterministic_and_identical_pair_is_zero() -> None:
    plan = ResamplingPlan.create(dataset="toy", aspect="BP", cohort_name="bin", protein_ids=["a", "b"], n_resamples=12)
    assert np.array_equal(plan.indices, ResamplingPlan.create(dataset="toy", aspect="BP", cohort_name="bin", protein_ids=["a", "b"], n_resamples=12).indices)
    values = np.array([.2, .7])
    delta = values[plan.indices].mean(1) - values[plan.indices].mean(1)
    assert percentile_interval(delta) == (0.0, 0.0)


def test_cache_alignment_and_missing_ids_fail() -> None:
    base = PredictionCache(np.array(["a", "b"]), np.array(["x", "y"]), np.ones((2, 2)), np.zeros((2, 2)), np.ones(2, bool), {"schema_version": "unused"}, np.array([[.5,.5],[.5,.5]]))
    permuted = PredictionCache(np.array(["b", "a"]), np.array(["y", "x"]), np.ones((2, 2)), np.zeros((2, 2)), np.ones(2, bool), {}, np.array([[.5,.5],[.5,.5]]))
    _, aligned = align_caches(base, permuted)
    assert aligned.protein_ids.tolist() == ["a", "b"]
    missing = PredictionCache(np.array(["a", "c"]), np.array(["x", "y"]), np.ones((2, 2)), np.zeros((2, 2)), np.ones(2, bool), {}, np.array([[.5,.5],[.5,.5]]))
    with pytest.raises(ValueError, match="missing/extra"):
        align_caches(base, missing)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_prediction_cache(base, Path(tmp) / "cache")
        assert read_prediction_cache(path).probabilities.shape == (2, 2)
