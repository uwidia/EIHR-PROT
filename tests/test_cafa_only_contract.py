from __future__ import annotations

import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.homology_only_baseline import evaluate_homology_only
from reliability_aware.utils.cafa_metrics import evaluate_cafa
from reliability_aware.utils.inference_utils import save_metrics_json
from reliability_aware.utils.model_training import evaluate_model
from scripts.experiments import run_all_similarity_bin_inference


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


class EchoProbabilities(torch.nn.Module):
    def forward(self, padded):
        return {"probs": padded}


class CafaOnlyContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.obo_path = Path(self.tempdir.name) / "go.obo"
        self.obo_path.write_text(OBO)
        self.go_terms = ["GO:0000001", "GO:0000002"]
        self.train_annotations = [{"GO:0000002"}]
        self.probs = torch.tensor([[0.2, 0.9]], dtype=torch.float32)
        self.targets = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        self.pos_weight = torch.ones(2)
        self.child_parent_pairs = torch.empty((0, 2), dtype=torch.long)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def assert_cafa_only(self, metrics: dict) -> None:
        legacy_raw = "Smin_" + "raw"
        legacy_normalized = "Smin_" + "normalized"
        self.assertEqual(metrics["ic_source"], "train")
        self.assertIn("Fmax", metrics)
        self.assertIn("AUPR", metrics)
        self.assertIn("Smin", metrics)
        self.assertNotIn(legacy_raw, metrics)
        self.assertNotIn(legacy_normalized, metrics)

    def test_cafa_api_has_no_ic_source_switch(self) -> None:
        self.assertNotIn("ic_source", inspect.signature(evaluate_cafa).parameters)

    def test_training_validation_uses_cafa_metrics(self) -> None:
        metrics = evaluate_model(
            model=EchoProbabilities(),
            loader=[{"padded": self.probs, "targets": self.targets}],
            pos_weight=self.pos_weight,
            child_parent_pairs=self.child_parent_pairs,
            lambda_hier=0.0,
            go_terms=self.go_terms,
            go_aspect="BP",
            obo_path=self.obo_path,
            train_annotations=self.train_annotations,
            device=torch.device("cpu"),
        )
        self.assert_cafa_only(metrics)

    def test_homology_validation_uses_cafa_metrics(self) -> None:
        metrics = evaluate_homology_only(
            loader=[{"probs": self.probs, "targets": self.targets}],
            pos_weight=self.pos_weight,
            child_parent_pairs=self.child_parent_pairs,
            lambda_hier=0.0,
            go_terms=self.go_terms,
            go_aspect="BP",
            obo_path=self.obo_path,
            train_annotations=self.train_annotations,
            device=torch.device("cpu"),
        )
        self.assert_cafa_only(metrics)

    def test_batch_inference_has_no_metric_switches(self) -> None:
        args = run_all_similarity_bin_inference.parse_args(["--top_k", "7"])
        run = run_all_similarity_bin_inference.InferenceRun(
            ablation="homology_only",
            go_aspect="BP",
            test_fasta=Path("test.fasta"),
            checkpoint=None,
            outdir=Path("output"),
        )
        command = run_all_similarity_bin_inference.build_command(run, args)
        self.assertEqual(
            command[3],
            "scripts/inference/run_inference_hom_only.py",
        )
        self.assertNotIn("--metric_style", command)
        self.assertNotIn("--cafa_" + "ic_source", command)

    def test_metrics_json_has_flat_cafa_metrics(self) -> None:
        output_path = Path(self.tempdir.name) / "metrics.json"
        metrics = {
            "Fmax": 1.0,
            "AUPR": 1.0,
            "Smin": 0.0,
            "n_proteins_evaluated": 1,
            "n_go_terms": 2,
            "ic_source": "train",
        }
        args = argparse.Namespace(
            ablation="sequence_only",
            go_aspect="BP",
            checkpoint=Path("best_model.pt"),
            test_fasta=Path("test.fasta"),
            train_fasta=Path("train.fasta"),
            test_esm_shard_dir=Path("embeddings"),
            test_manifest_path=Path("manifest.csv"),
            test_homology_shard_dir=None,
            go_vocab_path=Path("go_vocab.json"),
            go_annotation_path=Path("annotations.tsv"),
            obo_path=self.obo_path,
        )

        save_metrics_json(
            metrics=metrics,
            path=output_path,
            args=args,
            checkpoint={
                "epoch": 3,
                "val_metrics": {"Smin_" + "raw": 99.0},
            },
        )

        payload = json.loads(output_path.read_text())
        self.assertEqual(payload["metric_style"], "cafa")
        self.assertEqual(payload["metrics"]["Fmax"], 1.0)
        self.assertNotIn("cafa", payload["metrics"])
        self.assertNotIn("checkpoint_val_metrics", payload)
        self.assertFalse(
            (output_path.parent / ("cafa_" + "metrics.json")).exists()
        )


if __name__ == "__main__":
    unittest.main()
