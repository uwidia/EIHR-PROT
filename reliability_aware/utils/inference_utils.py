"""CLI and orchestration facade for prediction and CAFA evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Mapping, Sequence

import torch

from reliability_aware.utils.config import setup_logging
from reliability_aware.utils.inference_reporting import (
    build_per_protein_gate_rows,
    enrich_topk_rows,
    load_go_term_metadata,
    print_prediction_preview,
    save_metrics_json,
    save_per_protein_gate_csv,
    save_prediction_metadata,
    save_topk_csv,
)
from reliability_aware.utils.inference_runtime import (
    DatasetKind,
    InferenceSpec,
    build_label_indices_for_split,
    build_test_loader,
    compute_metrics,
    fasta_ids,
    load_go_vocab,
    load_model_from_checkpoint,
    load_torch_checkpoint,
    require_path,
    run_prediction_pass,
)

LOGGER = logging.getLogger("run_inference")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Re-exported imports above preserve the existing inference_utils public API.
__all__ = [
    "DatasetKind",
    "InferenceSpec",
    "add_file_logger",
    "build_label_indices_for_split",
    "build_per_protein_gate_rows",
    "build_test_loader",
    "compute_metrics",
    "enrich_topk_rows",
    "fasta_ids",
    "load_go_vocab",
    "load_go_term_metadata",
    "load_model_from_checkpoint",
    "load_torch_checkpoint",
    "parse_inference_args",
    "print_prediction_preview",
    "require_path",
    "resolve_path",
    "run_inference",
    "run_prediction_pass",
    "save_metrics_json",
    "save_per_protein_gate_csv",
    "save_prediction_metadata",
    "save_topk_csv",
]


def parse_inference_args(
    *,
    description: str,
    ablation_choices: Sequence[str] | None = None,
    gate_choices: Sequence[str] | None = None,
    default_gate: str | None = None,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)

    if ablation_choices is not None:
        selector = parser.add_mutually_exclusive_group(required=gate_choices is None)
        selector.add_argument(
            "--ablation",
            choices=sorted(ablation_choices),
            help="Sequence+homology architecture represented by the checkpoint.",
        )
        if gate_choices is not None:
            selector.add_argument(
                "--gate",
                choices=sorted(gate_choices),
                default=default_gate,
                help=(
                    "Gate architecture represented by the checkpoint "
                    f"(default: {default_gate})."
                ),
            )

    parser.add_argument(
        "--go_aspect",
        required=True,
        choices=["BP", "MF", "CC"],
        help="GO ontology aspect for this checkpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "evaluate"],
        default="predict",
        help=(
            "Use 'predict' for unlabeled proteins and prediction files only, or "
            "'evaluate' to also load annotations and compute CAFA metrics "
            "(default: predict)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help=(
            "Path to the run-specific best_model.pt file. Required for trained "
            "neural ablations; not used for homology_only."
        ),
    )
    parser.add_argument(
        "--test_fasta",
        type=Path,
        default=PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_test.fasta",
        help="Labeled test FASTA used only in evaluate mode.",
    )
    parser.add_argument(
        "--train_fasta",
        type=Path,
        default=PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_train.fasta",
        help="Training FASTA used only for CAFA IC values in evaluate mode.",
    )
    parser.add_argument(
        "--test_esm_shard_dir",
        type=Path,
        default=PROJECT_ROOT / "esm_embeddings/test",
        help="Directory containing test ESM embedding shards.",
    )
    parser.add_argument(
        "--test_manifest_path",
        type=Path,
        default=PROJECT_ROOT / "esm_embeddings/test/test_manifest.csv",
        help="Manifest CSV produced by get_embeddings.py for the test split.",
    )
    parser.add_argument(
        "--test_homology_shard_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing test homology shards for this GO aspect. "
            "Defaults to diamond_db/<GO_ASPECT>/test_homology_shards."
        ),
    )
    parser.add_argument(
        "--go_vocab_path",
        type=Path,
        default=None,
        help=(
            "GO vocabulary JSON used for this aspect. Defaults to "
            "diamond_db/<GO_ASPECT>/go_vocab.json."
        ),
    )
    parser.add_argument(
        "--go_annotation_path",
        type=Path,
        default=PROJECT_ROOT / "data/HEAL_dataset/nrPDB-GO_2019.06.18_annot.tsv",
        help="Annotation TSV used only in evaluate mode.",
    )
    parser.add_argument(
        "--obo_path",
        type=Path,
        default=PROJECT_ROOT / "data/HEAL_dataset/go-basic.obo",
        help=(
            "GO OBO file used to enrich predictions and compute evaluation metrics."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--print_limit",
        type=int,
        default=20,
        help="Number of proteins to print. Use -1 to print all.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "runs/inference",
        help="Directory where prediction outputs and optional metrics are written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--allow_unlabeled_predictions",
        action="store_true",
        help=(
            "Legacy sequence-only evaluation option. For pure unlabeled inference, "
            "use '--mode predict' instead."
        ),
    )
    return parser.parse_args(argv)


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def add_file_logger(outdir: Path) -> None:
    log_path = outdir / "inference.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(funcName)s:%(lineno)d | %(message)s"
        )
    )
    logging.getLogger().addHandler(handler)


def _select_spec(
    *,
    spec: InferenceSpec | Mapping[str, InferenceSpec],
    args: argparse.Namespace,
    gate_specs: Mapping[str, str] | None,
) -> InferenceSpec:
    if not isinstance(spec, Mapping):
        args.ablation = spec.ablation
        return spec

    selected_ablation = args.ablation
    if selected_ablation is None:
        if gate_specs is None:
            raise ValueError("An ablation selection is required")
        selected_ablation = gate_specs[args.gate]

    selected_spec = spec[selected_ablation]
    args.ablation = selected_spec.ablation
    return selected_spec


def _resolve_args(args: argparse.Namespace, spec: InferenceSpec) -> None:
    for name in (
        "checkpoint",
        "test_fasta",
        "train_fasta",
        "test_esm_shard_dir",
        "test_manifest_path",
        "go_annotation_path",
        "obo_path",
        "outdir",
    ):
        value = getattr(args, name)
        setattr(args, name, resolve_path(value) if value is not None else None)

    args.go_vocab_path = (
        PROJECT_ROOT / "diamond_db" / args.go_aspect / "go_vocab.json"
        if args.go_vocab_path is None
        else resolve_path(args.go_vocab_path)
    )
    if (
        args.test_homology_shard_dir is None
        and spec.dataset_kind in {"sequence_homology", "homology"}
    ):
        args.test_homology_shard_dir = (
            PROJECT_ROOT / "diamond_db" / args.go_aspect / "test_homology_shards"
        )
    elif args.test_homology_shard_dir is not None:
        args.test_homology_shard_dir = resolve_path(args.test_homology_shard_dir)


def _validate_inputs(args: argparse.Namespace, spec: InferenceSpec) -> None:
    if spec.dataset_kind != "homology" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for neural inference")
    if spec.dataset_kind != "homology" and spec.model_builder is None:
        raise ValueError(f"{spec.ablation} requires a model builder")
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    required_paths = [
        (args.test_manifest_path, "test manifest CSV"),
        (args.go_vocab_path, "GO vocabulary JSON"),
        (args.obo_path, "GO OBO file"),
    ]
    if args.mode == "evaluate":
        required_paths.extend(
            [
                (args.test_fasta, "test FASTA"),
                (args.train_fasta, "train FASTA"),
                (args.go_annotation_path, "GO annotation TSV"),
            ]
        )
    if args.checkpoint is not None:
        required_paths.append((args.checkpoint, "checkpoint"))
    if spec.dataset_kind in {"sequence", "sequence_homology"}:
        required_paths.append((args.test_esm_shard_dir, "test ESM shard directory"))
    if spec.dataset_kind in {"sequence_homology", "homology"}:
        required_paths.append(
            (args.test_homology_shard_dir, "test homology shard directory")
        )

    for path, description in required_paths:
        require_path(path, description)


def _build_evaluation_targets(
    *,
    args: argparse.Namespace,
    go_terms: list[str],
) -> tuple[dict[str, list[int]] | None, set[str], list[set[str]]]:
    if args.mode != "evaluate":
        LOGGER.info("Prediction mode: skipping annotations and CAFA metrics.")
        return None, set(), []

    go_term_to_idx = {go: i for i, go in enumerate(go_terms)}
    train_label_to_indices, train_keep_ids = build_label_indices_for_split(
        fasta_path=args.train_fasta,
        go_annotation_path=args.go_annotation_path,
        obo_path=args.obo_path,
        go_aspect=args.go_aspect,
        go_term_to_idx=go_term_to_idx,
    )
    test_label_to_indices, test_keep_ids = build_label_indices_for_split(
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
    LOGGER.info("Training proteins with labels: %d", len(train_keep_ids))
    LOGGER.info("Test proteins with labels: %d", len(test_keep_ids))
    return test_label_to_indices, test_keep_ids, train_annotations


def _load_model(
    *,
    args: argparse.Namespace,
    spec: InferenceSpec,
    go_terms: list[str],
    device: torch.device,
) -> tuple[torch.nn.Module | None, dict | None]:
    if spec.dataset_kind == "homology":
        LOGGER.info("Running homology-only inference; no checkpoint is loaded.")
        return None, None

    model, checkpoint = load_model_from_checkpoint(
        model_builder=spec.model_builder,
        checkpoint_path=args.checkpoint,
        go_terms=go_terms,
        device=device,
    )
    LOGGER.info("Loaded checkpoint epoch=%s", checkpoint.get("epoch"))
    return model, checkpoint


def _save_outputs(
    *,
    args: argparse.Namespace,
    checkpoint: dict | None,
    go_terms: list[str],
    results: dict,
    train_annotations: list[set[str]],
) -> None:
    topk_csv_path = args.outdir / "topk_predictions.csv"
    gate_scores_csv_path = args.outdir / "per_protein_gate_scores.csv"
    metadata_json_path = args.outdir / "prediction_metadata.json"

    save_topk_csv(results["topk_rows"], topk_csv_path)
    if "gate_weights" in results:
        gate_rows = build_per_protein_gate_rows(
            labels=results["labels"],
            global_indices=results["global_indices"],
            gate_weights=results["gate_weights"],
        )
        save_per_protein_gate_csv(gate_rows, gate_scores_csv_path)
    save_prediction_metadata(
        path=metadata_json_path,
        args=args,
        checkpoint=checkpoint,
        n_proteins=len(results["labels"]),
        n_go_terms=len(go_terms),
    )

    if args.mode == "evaluate":
        metrics = compute_metrics(
            y_true=results["y_true"],
            y_prob=results["y_prob"],
            go_terms=go_terms,
            go_aspect=args.go_aspect,
            obo_path=args.obo_path,
            train_annotations=train_annotations,
        )
        if "gate_weights" in results:
            gate_weights = results["gate_weights"]
            metrics["mean_neural_gate"] = float(gate_weights[:, 0].mean().item())
            metrics["mean_homology_gate"] = float(
                gate_weights[:, 1].mean().item()
            )
        metrics_json_path = args.outdir / "metrics.json"
        save_metrics_json(
            metrics=metrics,
            path=metrics_json_path,
            args=args,
            checkpoint=checkpoint,
        )
        print("\n=== Test metrics ===")
        print(
            f"Fmax: {metrics['Fmax']:.6f} @ threshold "
            f"{metrics['Fmax_threshold']:.3f}"
        )
        print(f"AUPR:  {metrics['AUPR']:.6f}")
        print(
            f"Smin: {metrics['Smin']:.6f} @ threshold "
            f"{metrics['Smin_threshold']:.3f}"
        )
        if "mean_neural_gate" in metrics:
            print(
                f"Mean gate: neural={metrics['mean_neural_gate']:.6f}, "
                f"homology={metrics['mean_homology_gate']:.6f}"
            )
        print(f"\nSaved metrics to: {metrics_json_path}")

    print("\n=== Top-k prediction preview ===")
    print_prediction_preview(results["topk_rows"], args.print_limit, args.top_k)
    print(f"Saved full top-{args.top_k} predictions to: {topk_csv_path}")
    print(f"Saved prediction metadata to: {metadata_json_path}")
    if "gate_weights" in results:
        print(f"Saved per-protein gate scores to: {gate_scores_csv_path}")


def run_inference(
    *,
    spec: InferenceSpec | Mapping[str, InferenceSpec],
    description: str,
    gate_specs: Mapping[str, str] | None = None,
    default_gate: str | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    specs = spec if isinstance(spec, Mapping) else None
    if gate_specs is not None:
        if specs is None:
            raise ValueError("gate_specs requires a mapping of inference specs")
        if default_gate not in gate_specs:
            raise ValueError("default_gate must be present in gate_specs")

    args = parse_inference_args(
        description=description,
        ablation_choices=list(specs) if specs is not None else None,
        gate_choices=list(gate_specs) if gate_specs is not None else None,
        default_gate=default_gate,
        argv=argv,
    )
    selected_spec = _select_spec(spec=spec, args=args, gate_specs=gate_specs)
    _resolve_args(args, selected_spec)
    _validate_inputs(args, selected_spec)

    args.outdir.mkdir(parents=True, exist_ok=True)
    setup_logging()
    add_file_logger(args.outdir)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Using device: %s", device)

    go_terms = load_go_vocab(args.go_vocab_path)
    go_metadata = load_go_term_metadata(args.obo_path, go_terms)
    LOGGER.info(
        "Loaded %d %s GO terms from %s",
        len(go_terms),
        args.go_aspect,
        args.go_vocab_path,
    )

    test_label_to_indices, test_keep_ids, train_annotations = (
        _build_evaluation_targets(args=args, go_terms=go_terms)
    )
    model, checkpoint = _load_model(
        args=args,
        spec=selected_spec,
        go_terms=go_terms,
        device=device,
    )
    dataset, loader = build_test_loader(
        spec=selected_spec,
        test_esm_shard_dir=args.test_esm_shard_dir,
        test_manifest_path=args.test_manifest_path,
        test_homology_shard_dir=args.test_homology_shard_dir,
        test_label_to_indices=test_label_to_indices,
        test_keep_ids=test_keep_ids,
        go_terms=go_terms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allow_unlabeled_predictions=args.allow_unlabeled_predictions,
        device=device,
    )
    LOGGER.info(
        "Built test loader with %d proteins and %d batches", len(dataset), len(loader)
    )

    results = run_prediction_pass(
        model=model,
        loader=loader,
        go_terms=go_terms,
        top_k=args.top_k,
        device=device,
    )
    enrich_topk_rows(results["topk_rows"], go_metadata)
    _save_outputs(
        args=args,
        checkpoint=checkpoint,
        go_terms=go_terms,
        results=results,
        train_annotations=train_annotations,
    )
