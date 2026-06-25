#!/usr/bin/env python3
"""Evaluate the Naive baseline with CAFA metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.naive_baseline import evaluate_naive
from reliability_aware.utils.inference_runtime import (
    build_label_indices_for_split,
    fasta_ids,
    load_go_vocab,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_y_true(
    *,
    labels: list[str],
    label_to_indices: dict[str, list[int]],
    num_go_terms: int,
) -> torch.Tensor:
    y_true = torch.zeros((len(labels), num_go_terms), dtype=torch.float32)
    for row, label in enumerate(labels):
        indices = label_to_indices.get(label, [])
        if indices:
            y_true[row, indices] = 1.0
    return y_true


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Naive baseline (N_f / N_total) with CAFA metrics."
    )
    parser.add_argument("--go_aspect", required=True, choices=["BP", "MF", "CC"])
    parser.add_argument(
        "--mode",
        choices=["predict", "evaluate"],
        default="evaluate",
        help="Accepted for compatibility; this script always evaluates with labels.",
    )
    parser.add_argument(
        "--test_fasta",
        type=Path,
        default=PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_test.fasta",
    )
    parser.add_argument(
        "--train_fasta",
        type=Path,
        default=PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_train.fasta",
    )
    parser.add_argument(
        "--go_annotation_path",
        type=Path,
        default=PROJECT_ROOT / "data/HEAL_dataset/nrPDB-GO_2019.06.18_annot.tsv",
    )
    parser.add_argument(
        "--obo_path",
        type=Path,
        default=PROJECT_ROOT / "data/HEAL_dataset/go-basic.obo",
    )
    parser.add_argument("--go_vocab_path", type=Path, default=None)
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Accepted for batch-runner compatibility; unused by naive baseline.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "runs/inference/naive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode != "evaluate":
        raise ValueError("run_inference_naive.py supports only --mode evaluate")

    go_vocab_path = (
        PROJECT_ROOT / "diamond_db" / args.go_aspect / "go_vocab.json"
        if args.go_vocab_path is None
        else args.go_vocab_path
    )

    go_terms = load_go_vocab(go_vocab_path)
    go_term_to_idx = {go: i for i, go in enumerate(go_terms)}

    train_label_to_indices, train_keep_ids = build_label_indices_for_split(
        fasta_path=args.train_fasta,
        go_annotation_path=args.go_annotation_path,
        obo_path=args.obo_path,
        go_aspect=args.go_aspect,
        go_term_to_idx=go_term_to_idx,
    )
    test_label_to_indices, _test_keep_ids = build_label_indices_for_split(
        fasta_path=args.test_fasta,
        go_annotation_path=args.go_annotation_path,
        obo_path=args.obo_path,
        go_aspect=args.go_aspect,
        go_term_to_idx=go_term_to_idx,
    )

    train_annotations = [
        {go_terms[index] for index in train_label_to_indices[label]}
        for label in train_keep_ids
        if label in train_label_to_indices
    ]

    test_labels = fasta_ids(args.test_fasta)
    y_true = _build_y_true(
        labels=test_labels,
        label_to_indices=test_label_to_indices,
        num_go_terms=len(go_terms),
    )

    metrics, _curve, _y_prob = evaluate_naive(
        y_true=y_true,
        go_terms=go_terms,
        go_aspect=args.go_aspect,
        obo_path=str(args.obo_path),
        train_annotations=train_annotations,
        train_label_to_indices=train_label_to_indices,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    output_path = args.outdir / f"{args.go_aspect}_metrics.json"
    payload = {
        "ablation": "naive",
        "go_aspect": args.go_aspect,
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"Fmax: {metrics['Fmax']:.6f}")
    print(f"AUPR: {metrics['AUPR']:.6f}")
    print(f"Smin: {metrics['Smin']:.6f}")
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
