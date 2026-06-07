from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from reliability_aware.utils.cafa_metrics import evaluate_cafa


OBO = """format-version: 1.2

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0000001
name: parent
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0000002
name: child
namespace: biological_process
is_a: GO:0000001 ! parent
"""


class CafaMetricsTest(unittest.TestCase):
    def test_set_metrics_propagate_predictions_and_filter_empty_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obo_path = Path(tmpdir) / "go.obo"
            obo_path.write_text(OBO)

            metrics, curve = evaluate_cafa(
                y_true=np.asarray([[0, 1], [0, 0]], dtype=np.float32),
                y_prob=np.asarray([[0, 0.9], [1, 1]], dtype=np.float32),
                go_terms=["GO:0000001", "GO:0000002"],
                go_aspect="BP",
                obo_path=obo_path,
                train_annotations=[{"GO:0000002"}],
                ic_source="train",
            )

        self.assertEqual(metrics["n_proteins_evaluated"], 1)
        self.assertAlmostEqual(metrics["Fmax"], 1.0)
        self.assertAlmostEqual(metrics["Fmax_threshold"], 0.01)
        self.assertAlmostEqual(metrics["Smin"], 0.0)
        self.assertEqual(len(curve), 100)
        self.assertEqual(curve[0]["threshold"], 0.01)
        self.assertEqual(curve[-1]["threshold"], 1.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obo_path = Path(tmpdir) / "go.obo"
            obo_path.write_text(OBO)

            with self.assertRaises(ValueError):
                evaluate_cafa(
                    y_true=np.zeros((1, 2)),
                    y_prob=np.zeros((2, 2)),
                    go_terms=["GO:0000001", "GO:0000002"],
                    go_aspect="BP",
                    obo_path=obo_path,
                    train_annotations=[],
                )


if __name__ == "__main__":
    unittest.main()
