#!/usr/bin/env python3
"""Run inference across all ablations, GO aspects, and similarity bins."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ABLATIONS = [
    "sequence_only",
    "sequence_homology_internal_gate",
    "sequence_homology_confidence_gate",
    "homology_only",
]
GO_ASPECTS = ["BP", "MF", "CC"]
TEST_FASTAS = [
    Path("data/cleaned_dataset/similarity_bins/30_similarity_test_sequence.fasta"),
    Path("data/cleaned_dataset/similarity_bins/40_similarity_test_sequence.fasta"),
    Path("data/cleaned_dataset/similarity_bins/50_similarity_test_sequence.fasta"),
    Path("data/cleaned_dataset/similarity_bins/70_similarity_test_sequence.fasta"),
    Path("data/cleaned_dataset/similarity_bins/95_similarity_test_sequence.fasta"),
]

ABLATION_TO_MODEL = {
    "sequence_homology_confidence_gate": "confidence_gate",
    "sequence_homology_internal_gate": "internal_gate",
    "sequence_only": "sequence_only",
    "homology_only": "homology_only",
}
ABLATION_TO_SCRIPT = {
    "sequence_only": "scripts/inference/run_inference_seq_only.py",
    "sequence_homology_internal_gate": "scripts/inference/run_inference_seq_hom.py",
    "sequence_homology_confidence_gate": "scripts/inference/run_inference_seq_hom.py",
    "homology_only": "scripts/inference/run_inference_hom_only.py",
}


@dataclass(frozen=True)
class InferenceRun:
    ablation: str
    go_aspect: str
    test_fasta: Path
    checkpoint: Path | None
    outdir: Path


@dataclass
class RunSummary:
    planned: int = 0
    executed: int = 0
    skipped_missing_checkpoints: int = 0
    skipped_missing_fastas: int = 0
    failed: int = 0
    completed: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CAFA-style inference for every ablation, GO aspect, and "
            "similarity-bin test FASTA."
        )
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print valid commands without executing them.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running commands after an inference command fails.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args(argv)


def similarity_bin_name(test_fasta: Path) -> str:
    suffix = "_test_sequence.fasta"
    if not test_fasta.name.endswith(suffix):
        raise ValueError(
            f"Cannot parse similarity bin from FASTA name: {test_fasta.name}"
        )
    return test_fasta.name.removesuffix(suffix)


def build_runs() -> list[InferenceRun]:
    runs: list[InferenceRun] = []
    for ablation in ABLATIONS:
        model = ABLATION_TO_MODEL[ablation]
        for go_aspect in GO_ASPECTS:
            checkpoint = None
            if ablation != "homology_only":
                checkpoint = (
                    PROJECT_ROOT
                    / "runs"
                    / ablation
                    / "final"
                    / go_aspect
                    / "best_model.pt"
                )

            for relative_fasta in TEST_FASTAS:
                test_fasta = PROJECT_ROOT / relative_fasta
                outdir = (
                    PROJECT_ROOT
                    / "inference"
                    / model
                    / similarity_bin_name(relative_fasta)
                    / f"{go_aspect}_cafa"
                )
                runs.append(
                    InferenceRun(
                        ablation=ablation,
                        go_aspect=go_aspect,
                        test_fasta=test_fasta,
                        checkpoint=checkpoint,
                        outdir=outdir,
                    )
                )
    return runs


def build_command(run: InferenceRun, args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        ABLATION_TO_SCRIPT[run.ablation],
        "--go_aspect",
        run.go_aspect,
        "--mode",
        "evaluate",
    ]
    if run.ablation.startswith("sequence_homology_"):
        command.extend(["--ablation", run.ablation])
    if run.checkpoint is not None:
        command.extend(["--checkpoint", str(run.checkpoint)])
    command.extend(
        [
            "--top_k",
            str(args.top_k),
            "--outdir",
            str(run.outdir),
            "--test_fasta",
            str(run.test_fasta),
        ]
    )
    return command


def print_summary(summary: RunSummary, *, dry_run: bool, stopped_early: bool) -> None:
    print("\n=== Batch inference summary ===")
    print(f"Commands planned: {summary.planned}")
    print(f"Commands executed: {summary.executed}")
    print(
        "Skipped (missing checkpoints): "
        f"{summary.skipped_missing_checkpoints}"
    )
    print(f"Skipped (missing FASTA files): {summary.skipped_missing_fastas}")
    print(f"Failed: {summary.failed}")
    print(f"Completed successfully: {summary.completed}")
    if dry_run:
        print("Dry run: no commands were executed.")
    if stopped_early:
        print("Stopped after the first failed command; remaining runs were not attempted.")


def run_all(args: argparse.Namespace) -> int:
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")

    runs = build_runs()
    summary = RunSummary(planned=len(runs))
    stopped_early = False

    for index, run in enumerate(runs, start=1):
        if not run.test_fasta.is_file():
            summary.skipped_missing_fastas += 1
            print(
                f"WARNING [{index}/{summary.planned}]: missing test FASTA; "
                f"skipping {run.ablation}/{run.go_aspect}: {run.test_fasta}"
            )
            continue

        if run.checkpoint is not None and not run.checkpoint.is_file():
            summary.skipped_missing_checkpoints += 1
            print(
                f"WARNING [{index}/{summary.planned}]: missing checkpoint; "
                f"skipping {run.ablation}/{run.go_aspect}: {run.checkpoint}"
            )
            continue

        command = build_command(run, args)
        print(f"\n[{index}/{summary.planned}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue

        summary.executed += 1
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            summary.failed += 1
            print(
                f"ERROR: command failed with exit code {exc.returncode}: "
                f"{shlex.join(command)}"
            )
            if not args.continue_on_error:
                stopped_early = True
                break
        else:
            summary.completed += 1

    print_summary(summary, dry_run=args.dry_run, stopped_early=stopped_early)
    return 1 if summary.failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_all(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
