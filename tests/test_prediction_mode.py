from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from models.sequence_homology_common import make_sequence_homology_collate_fn
from reliability_aware.utils.inference_utils import (
    enrich_topk_rows,
    load_go_term_metadata,
    parse_inference_args,
    run_prediction_pass,
    save_topk_csv,
)

OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: example process
namespace: biological_process
def: "An example biological process." [GOC:test]
"""


class PredictionModeTest(unittest.TestCase):
    def test_prediction_is_the_default_mode(self) -> None:
        args = parse_inference_args(
            description="test",
            argv=["--go_aspect", "BP"],
        )
        self.assertEqual(args.mode, "predict")

    def test_prediction_mode_parses_without_annotation_arguments(self) -> None:
        args = parse_inference_args(
            description="test",
            argv=["--go_aspect", "BP", "--mode", "predict"],
        )
        self.assertEqual(args.mode, "predict")

    def test_sequence_homology_collate_can_omit_targets(self) -> None:
        collate = make_sequence_homology_collate_fn(
            label_to_indices=None,
            num_go_terms=2,
        )
        batch = collate(
            [
                {
                    "rep": torch.ones(3, 1280),
                    "label": "unlabeled-protein",
                    "global_idx": 0,
                    "homology_scores": torch.tensor([0.2, 0.8]),
                    "gate_features": torch.tensor([1.0, 0.5, 0.7, 1.0]),
                }
            ]
        )
        self.assertIsNone(batch["targets"])

    def test_prediction_pass_does_not_require_targets(self) -> None:
        result = run_prediction_pass(
            model=None,
            loader=[
                {
                    "probs": torch.tensor([[0.2, 0.8]]),
                    "targets": None,
                    "global_indices": torch.tensor([0]),
                    "labels": ["unlabeled-protein"],
                }
            ],
            go_terms=["GO:0000001", "GO:0000002"],
            top_k=1,
            device=torch.device("cpu"),
        )
        self.assertNotIn("y_true", result)
        self.assertEqual(result["topk_rows"][0]["go_term"], "GO:0000002")

    def test_topk_rows_include_obo_metadata_in_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obo_path = Path(tmpdir) / "go.obo"
            csv_path = Path(tmpdir) / "topk_predictions.csv"
            obo_path.write_text(OBO)

            metadata = load_go_term_metadata(obo_path, ["GO:0000001"])
            rows = [
                {
                    "protein_id": "protein-1",
                    "global_idx": 0,
                    "rank": 1,
                    "go_term": "GO:0000001",
                    "probability": 0.9,
                    "neural_probability": 0.8,
                    "homology_probability": 1.0,
                    "neural_gate": 0.5,
                    "homology_gate": 0.5,
                }
            ]
            enrich_topk_rows(rows, metadata)
            save_topk_csv(rows, csv_path)

            with csv_path.open(newline="") as handle:
                saved = next(csv.DictReader(handle))

        self.assertEqual(saved["go_id"], "GO:0000001")
        self.assertEqual(saved["go_name"], "example process")
        self.assertEqual(saved["aspect"], "BP")
        self.assertEqual(saved["definition"], "An example biological process.")
        self.assertEqual(saved["probability_score"], "0.9")


if __name__ == "__main__":
    unittest.main()
