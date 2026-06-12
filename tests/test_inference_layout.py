from __future__ import annotations

import ast
import unittest
from pathlib import Path

from reliability_aware.utils.inference_utils import parse_inference_args
from scripts.experiments import run_all_similarity_bin_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InferenceLayoutTest(unittest.TestCase):
    def test_sequence_homology_gate_defaults_to_confidence(self) -> None:
        args = parse_inference_args(
            description="test",
            ablation_choices=[
                "sequence_homology_internal_gate",
                "sequence_homology_confidence_gate",
            ],
            gate_choices=["internal", "confidence"],
            default_gate="confidence",
            argv=["--go_aspect", "BP"],
        )
        self.assertIsNone(args.ablation)
        self.assertEqual(args.gate, "confidence")

    def test_sequence_homology_gate_can_select_internal(self) -> None:
        args = parse_inference_args(
            description="test",
            ablation_choices=[
                "sequence_homology_internal_gate",
                "sequence_homology_confidence_gate",
            ],
            gate_choices=["internal", "confidence"],
            default_gate="confidence",
            argv=["--go_aspect", "BP", "--gate", "internal"],
        )
        self.assertEqual(args.gate, "internal")

    def test_batch_commands_route_to_model_specific_scripts(self) -> None:
        args = run_all_similarity_bin_inference.parse_args(["--top_k", "3"])
        expected_scripts = {
            "sequence_only": "scripts/inference/run_inference_seq_only.py",
            "sequence_homology_internal_gate": (
                "scripts/inference/run_inference_seq_hom.py"
            ),
            "sequence_homology_confidence_gate": (
                "scripts/inference/run_inference_seq_hom.py"
            ),
            "homology_only": "scripts/inference/run_inference_hom_only.py",
        }

        for ablation, expected_script in expected_scripts.items():
            with self.subTest(ablation=ablation):
                run = run_all_similarity_bin_inference.InferenceRun(
                    ablation=ablation,
                    go_aspect="BP",
                    test_fasta=Path("test.fasta"),
                    checkpoint=None,
                    outdir=Path("output"),
                )
                command = run_all_similarity_bin_inference.build_command(run, args)
                self.assertEqual(command[3], expected_script)
                self.assertIn("--mode", command)
                self.assertEqual(command[command.index("--mode") + 1], "evaluate")
                if ablation.startswith("sequence_homology_"):
                    self.assertIn("--ablation", command)
                else:
                    self.assertNotIn("--ablation", command)

    def test_entry_scripts_are_thin(self) -> None:
        scripts = [
            "run_inference_seq_only.py",
            "run_inference_seq_hom.py",
            "run_inference_hom_only.py",
        ]
        for filename in scripts:
            with self.subTest(filename=filename):
                path = PROJECT_ROOT / "scripts/inference" / filename
                tree = ast.parse(path.read_text(), filename=str(path))
                functions = [
                    node.name
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                ]
                self.assertLessEqual(len(functions), 2)
                self.assertLessEqual(len(path.read_text().splitlines()), 60)

    def test_inference_helpers_have_focused_module_ownership(self) -> None:
        expected_by_module = {
            "inference_runtime.py": {
            "build_label_indices_for_split",
            "build_test_loader",
            "compute_metrics",
            "load_model_from_checkpoint",
            "run_prediction_pass",
            },
            "inference_reporting.py": {
            "save_metrics_json",
            "save_topk_csv",
            },
            "inference_utils.py": {
                "parse_inference_args",
                "run_inference",
            },
        }

        for filename, expected in expected_by_module.items():
            with self.subTest(filename=filename):
                path = PROJECT_ROOT / "reliability_aware/utils" / filename
                tree = ast.parse(path.read_text(), filename=str(path))
                functions = {
                    node.name
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                }
                self.assertTrue(expected.issubset(functions))

        facade_lines = (
            PROJECT_ROOT / "reliability_aware/utils/inference_utils.py"
        ).read_text().splitlines()
        self.assertLess(len(facade_lines), 600)

    def test_inference_utils_preserves_public_helper_imports(self) -> None:
        from reliability_aware.utils import inference_utils

        expected = {
            "InferenceSpec",
            "build_label_indices_for_split",
            "build_test_loader",
            "compute_metrics",
            "enrich_topk_rows",
            "load_go_term_metadata",
            "load_model_from_checkpoint",
            "parse_inference_args",
            "run_inference",
            "run_prediction_pass",
            "save_metrics_json",
            "save_topk_csv",
        }
        self.assertTrue(expected.issubset(set(inference_utils.__all__)))


if __name__ == "__main__":
    unittest.main()
