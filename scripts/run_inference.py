#!/usr/bin/env python3
"""
Run inference for sequence-only, sequence+homology, and homology-only ablations.

This script assumes you have already prepared the test split artifacts with:

  1. scripts/get_embeddings.py
  2. scripts/prepare_diamond_hits.py
  3. scripts/build_homology_shards.py --go_aspect {BP,MF,CC}

It intentionally reuses the repo's existing model builders, datasets, collate
functions, forward helper, and metric utilities instead of reimplementing them.

Example:
  python scripts/run_inference.py \
    --ablation sequence_homology_confidence_gate \
    --go_aspect CC \
    --checkpoint runs/models/updated_sequence_homology_confidence_gate/correct_retryyy/CC/run_000/best_model.pt \
    --top_k 10 \
    --outdir runs/inference/confidence_gate/CC
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

def find_project_root() -> Path:
    """Find the repo root whether this file is in scripts/ or run from the repo root."""
    script_path = Path(__file__).resolve()
    candidates = []

    # Normal placement: <repo>/scripts/run_inference.py
    if len(script_path.parents) >= 2:
        candidates.append(script_path.parents[1])

    # Also support running a copied script from the repository root.
    candidates.append(Path.cwd().resolve())

    for candidate in candidates:
        if (candidate / "models").is_dir() and (candidate / "reliability_aware").is_dir():
            return candidate

    raise RuntimeError(
        "Could not find the repository root. Put this file at scripts/run_inference.py "
        "or run it from the Reliability-Aware-PFP repository root."
    )


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.homology_only_baseline import (  # noqa: E402
    HomologyShardDataset,
    make_homology_only_collate_fn,
)
from models.sequence_homology_ablation import (  # noqa: E402
    build_sequence_homology_confidence_gate_model,
    build_sequence_homology_internal_gate_model,
)
from models.sequence_homology_common import (  # noqa: E402
    SequenceHomologyShardDataset,
    make_sequence_homology_collate_fn,
)
from models.sequence_only_ablation import (  # noqa: E402
    SequenceOnlyESMShardDataset,
    build_seq_only_model,
    make_sequence_only_collate_fn,
)
from reliability_aware.utils.config import setup_logging  # noqa: E402
from reliability_aware.utils.go_term_extraction import (  # noqa: E402
    build_go_annotations_list,
    build_subject_go_index,
)
from reliability_aware.utils.metrics import fmax_score, smin_score  # noqa: E402
from reliability_aware.utils.model_training import (  # noqa: E402
    compute_ic_from_label_indices,
    model_forward_from_batch,
    move_batch_to_device,
)
from reliability_aware.utils.parser import get_protein_info  # noqa: E402


LOGGER = logging.getLogger("run_inference")

SUPPORTED_ABLATIONS = {
    "homology_only": {
        "builder": None,
        "dataset_kind": "homology",
    },
    "sequence_only": {
        "builder": build_seq_only_model,
        "dataset_kind": "sequence",
    },
    "sequence_homology_internal_gate": {
        "builder": build_sequence_homology_internal_gate_model,
        "dataset_kind": "sequence_homology",
    },
    "sequence_homology_confidence_gate": {
        "builder": build_sequence_homology_confidence_gate_model,
        "dataset_kind": "sequence_homology",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test-set inference from an existing best_model.pt checkpoint."
    )

    parser.add_argument(
        "--ablation",
        required=True,
        choices=sorted(SUPPORTED_ABLATIONS),
        help="Which trained ablation/checkpoint architecture to use.",
    )
    parser.add_argument(
        "--go_aspect",
        required=True,
        choices=["BP", "MF", "CC"],
        help="GO ontology aspect for this checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        default=None,
        type=Path,
        help=(
            "Path to the run-specific best_model.pt file. Required for trained "
            "neural ablations; not used for homology_only."
        ),
    )

    # Dataset/artifact paths. Defaults match the repo's current layout.
    parser.add_argument(
        "--test_fasta",
        type=Path,
        default=PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_test.fasta",
        help="FASTA used to generate the test embeddings/homology artifacts.",
    )
    parser.add_argument(
        "--train_fasta",
        type=Path,
        default=PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_train.fasta",
        help="Training FASTA used to define IC values and the training GO vocabulary.",
    )
    parser.add_argument(
        "--test_esm_shard_dir",
        type=Path,
        default=PROJECT_ROOT / "esm_embeddings/test",
        help="Directory containing test ESM embedding shards, e.g. part_0000.pt.",
    )
    parser.add_argument(
        "--test_manifest_path",
        type=Path,
        default=PROJECT_ROOT / "esm_embeddings/test/pdb_test_manifest.csv",
        help="Manifest CSV produced by get_embeddings.py for the test split.",
    )
    parser.add_argument(
        "--test_homology_shard_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing test homology shards for this GO aspect. "
            "Required for sequence+homology models. Defaults to "
            "diamond_db/<GO_ASPECT>/test_homology_shards."
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
        help="Annotation TSV used to build ground-truth test targets.",
    )
    parser.add_argument(
        "--obo_path",
        type=Path,
        default=PROJECT_ROOT / "data/HEAL_dataset/go-basic.obo",
        help="GO OBO file used for annotation propagation.",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--print_limit",
        type=int,
        default=20,
        help="Number of proteins to print top-k predictions for. Use -1 to print all.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "runs/inference",
        help="Directory where metrics and top-k predictions will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cuda, cuda:0, or cpu. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--metrics_steps",
        type=int,
        default=101,
        help="Number of thresholds to scan for Fmax and Smin.",
    )
    parser.add_argument(
        "--allow_unlabeled_predictions",
        action="store_true",
        help=(
            "Allow sequence-only prediction even if some test proteins lack labels for "
            "the selected aspect. Metrics are still computed on the loaded target matrix. "
            "For sequence+homology models, the existing collate function requires labels."
        ),
    )

    return parser.parse_args()


def resolve_path(path: Path | None) -> Path | None:
    """Resolve paths relative to the repository root."""
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def add_file_logger(outdir: Path) -> None:
    """Add a per-run log file in addition to the repo's existing pipeline.log."""
    log_path = outdir / "inference.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        )
    )
    logging.getLogger().addHandler(handler)


def load_torch_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load checkpoints across PyTorch versions, including versions with weights_only."""
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    return checkpoint


def load_go_vocab(go_vocab_path: Path) -> list[str]:
    require_path(go_vocab_path, "GO vocabulary JSON")
    go_terms = json.loads(go_vocab_path.read_text())
    if not isinstance(go_terms, list) or not all(isinstance(x, str) for x in go_terms):
        raise ValueError(f"Invalid GO vocabulary file: {go_vocab_path}")
    if not go_terms:
        raise ValueError(f"Empty GO vocabulary: {go_vocab_path}")
    return go_terms


def fasta_ids(fasta_path: Path) -> list[str]:
    return [protein["full_id"] for protein in get_protein_info(fasta_path)]


def build_label_indices_for_split(
    *,
    fasta_path: Path,
    go_annotation_path: Path,
    obo_path: Path,
    go_aspect: str,
    go_term_to_idx: dict[str, int],
) -> tuple[dict[str, list[int]], set[str]]:
    """Build label -> GO-index targets using the existing annotation utilities."""
    ids = fasta_ids(fasta_path)
    label_to_go_terms, _ = build_go_annotations_list(
        go_annotation_path=go_annotation_path,
        obo_path=obo_path,
        go_aspect=go_aspect,
        keep_ids=ids,
        remove_root_term=True,
        min_term_freq=None,
    )
    label_to_indices = build_subject_go_index(label_to_go_terms, go_term_to_idx)
    keep_ids = {label for label, idxs in label_to_indices.items() if idxs}
    return label_to_indices, keep_ids


def load_model_from_checkpoint(
    *,
    ablation: str,
    checkpoint_path: Path,
    go_terms: list[str],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'model_state_dict'. Make sure you passed run_XXX/best_model.pt, "
            "not best_final_run.pt or final_meta.pt."
        )
    if "hparams" not in checkpoint or checkpoint["hparams"] is None:
        raise KeyError(
            "Checkpoint is missing 'hparams'. The existing model builder needs these to rebuild the architecture."
        )

    hparams = checkpoint["hparams"]
    builder = SUPPORTED_ABLATIONS[ablation]["builder"]

    # Reuse the repo's builder. It returns an optimizer too, but inference does not use it.
    model, _optimizer = builder(hparams, go_terms, device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model mismatch. Check that --ablation, --go_aspect, and --go_vocab_path "
            "match the exact training run."
        ) from exc

    model.eval()
    return model, checkpoint


def build_test_loader(
    *,
    ablation: str,
    test_esm_shard_dir: Path,
    test_manifest_path: Path,
    test_homology_shard_dir: Path | None,
    test_label_to_indices: dict[str, list[int]],
    test_keep_ids: set[str],
    go_terms: list[str],
    batch_size: int,
    num_workers: int,
    allow_unlabeled_predictions: bool,
    device: torch.device,
) -> tuple[Any, DataLoader]:
    """Build a test DataLoader using the existing dataset and collate functions."""
    dataset_kind = SUPPORTED_ABLATIONS[ablation]["dataset_kind"]
    keep_ids = None if allow_unlabeled_predictions and dataset_kind == "sequence" else test_keep_ids

    if dataset_kind == "sequence":
        dataset = SequenceOnlyESMShardDataset(
            shard_dir=test_esm_shard_dir,
            manifest_path=test_manifest_path,
            keep_ids=keep_ids,
        )
        collate_fn = make_sequence_only_collate_fn(
            global_idx_to_label=dataset.global_idx_to_label,
            label_to_indices=test_label_to_indices,
            num_go_terms=len(go_terms),
        )
    elif dataset_kind == "sequence_homology":
        if test_homology_shard_dir is None:
            raise ValueError("sequence+homology models require --test_homology_shard_dir")
        dataset = SequenceHomologyShardDataset(
            esm_shard_dir=test_esm_shard_dir,
            homology_shard_dir=test_homology_shard_dir,
            manifest_path=test_manifest_path,
            keep_ids=keep_ids,
        )
        collate_fn = make_sequence_homology_collate_fn(
            label_to_indices=test_label_to_indices,
            num_go_terms=len(go_terms),
        )
    elif dataset_kind == "homology":
        if test_homology_shard_dir is None:
            raise ValueError("homology_only requires --test_homology_shard_dir")
        dataset = HomologyShardDataset(
            homology_shard_dir=test_homology_shard_dir,
            manifest_path=test_manifest_path,
            keep_ids=test_keep_ids,
        )
        collate_fn = make_homology_only_collate_fn(
            label_to_indices=test_label_to_indices,
            num_go_terms=len(go_terms),
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {dataset_kind}")

    if len(dataset) == 0:
        raise RuntimeError(
            "The test dataset is empty after filtering. This usually means no test proteins had "
            "ground-truth annotations for the selected GO aspect."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return dataset, loader


@torch.no_grad()
def run_prediction_pass(
    *,
    model: torch.nn.Module | None,
    loader: DataLoader,
    go_terms: list[str],
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run one forward pass, collect probabilities/targets, and build top-k rows."""
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_gate_weights: list[torch.Tensor] = []
    labels_all: list[str] = []
    global_indices_all: list[int] = []
    topk_rows: list[dict[str, Any]] = []

    k = min(top_k, len(go_terms))

    for batch in loader:
        labels = list(batch["labels"])
        global_indices = [int(x) for x in batch["global_indices"].tolist()]

        if model is None:
            # Homology-only baseline: the collate function already returns the
            # precomputed homology prior vector as probs. No neural model exists.
            probs = batch["probs"].detach().cpu().float()
            targets = batch["targets"].detach().cpu().float()
            neural_probs = None
            homology_scores = probs
            gate_weights = None
        else:
            batch = move_batch_to_device(batch, device)
            outputs = model_forward_from_batch(model, batch)

            probs = outputs["probs"].detach().cpu().float()
            targets = batch["targets"].detach().cpu().float()

            neural_probs = outputs.get("neural_probs")
            if neural_probs is not None:
                neural_probs = neural_probs.detach().cpu().float()

            homology_scores = outputs.get("homology_scores")
            if homology_scores is not None:
                homology_scores = homology_scores.detach().cpu().float()

            gate_weights = outputs.get("gate_weights")

        all_probs.append(probs)
        all_targets.append(targets)
        labels_all.extend(labels)
        global_indices_all.extend(global_indices)
        if gate_weights is not None:
            gate_weights = gate_weights.detach().cpu().float()
            all_gate_weights.append(gate_weights)

        top_values, top_indices = probs.topk(k=k, dim=1)

        for row_idx, protein_id in enumerate(labels):
            neural_gate = ""
            homology_gate = ""
            if gate_weights is not None:
                neural_gate = float(gate_weights[row_idx, 0].item())
                homology_gate = float(gate_weights[row_idx, 1].item())

            for rank, (go_idx, score) in enumerate(
                zip(top_indices[row_idx].tolist(), top_values[row_idx].tolist()),
                start=1,
            ):
                row = {
                    "protein_id": protein_id,
                    "global_idx": global_indices[row_idx],
                    "rank": rank,
                    "go_term": go_terms[go_idx],
                    "probability": float(score),
                    "neural_probability": "",
                    "homology_probability": "",
                    "neural_gate": neural_gate,
                    "homology_gate": homology_gate,
                }
                if neural_probs is not None:
                    row["neural_probability"] = float(neural_probs[row_idx, go_idx].item())
                if homology_scores is not None:
                    row["homology_probability"] = float(homology_scores[row_idx, go_idx].item())
                topk_rows.append(row)

    y_prob = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    result: dict[str, Any] = {
        "y_prob": y_prob,
        "y_true": y_true,
        "labels": labels_all,
        "global_indices": global_indices_all,
        "topk_rows": topk_rows,
    }
    if all_gate_weights:
        result["gate_weights"] = torch.cat(all_gate_weights, dim=0)
    return result


def compute_metrics(
    *,
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    ic: torch.Tensor,
    steps: int,
) -> dict[str, float]:
    """Compute requested metrics using the repo's existing Fmax/Smin utilities."""
    fmax = fmax_score(y_true, y_prob, steps=steps)
    smin = smin_score(y_true, y_prob, ic, steps=steps)

    try:
        aupr = float(average_precision_score(y_true.numpy().ravel(), y_prob.numpy().ravel()))
    except ValueError:
        aupr = math.nan

    return {
        "Fmax": float(fmax["Fmax"]),
        "Fmax_threshold": float(fmax["threshold"]),
        "Fmax_precision": float(fmax["precision"]),
        "Fmax_recall": float(fmax["recall"]),
        "AUPR": aupr,
        "Smin_raw": float(smin["raw"]["Smin"]),
        "Smin_threshold_raw": float(smin["raw"]["threshold"]),
        "Smin_ru_raw": float(smin["raw"]["ru"]),
        "Smin_mi_raw": float(smin["raw"]["mi"]),
        "Smin_normalized": float(smin["normalized"]["Smin"]),
        "Smin_threshold_normalized": float(smin["normalized"]["threshold"]),
    }


def save_topk_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "protein_id",
        "global_idx",
        "rank",
        "go_term",
        "probability",
        "neural_probability",
        "homology_probability",
        "neural_gate",
        "homology_gate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_prediction_preview(rows: list[dict[str, Any]], print_limit: int, top_k: int) -> None:
    if print_limit == 0:
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["protein_id"], []).append(row)

    for printed, (protein_id, protein_rows) in enumerate(grouped.items()):
        if print_limit >= 0 and printed >= print_limit:
            break

        print(f"\n{protein_id}")
        for pred in protein_rows[:top_k]:
            print(
                f"  #{pred['rank']:02d} {pred['go_term']}  "
                f"p={pred['probability']:.6f}"
            )


def save_metrics_json(
    *,
    metrics: dict[str, float],
    path: Path,
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
    n_proteins: int,
    n_go_terms: int,
) -> None:
    payload = {
        "metrics": metrics,
        "n_proteins_evaluated": n_proteins,
        "n_go_terms": n_go_terms,
        "ablation": args.ablation,
        "go_aspect": args.go_aspect,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "checkpoint_epoch": checkpoint.get("epoch") if checkpoint is not None else None,
        "checkpoint_val_metrics": checkpoint.get("val_metrics") if checkpoint is not None else None,
        "paths": {
            "test_fasta": str(args.test_fasta),
            "train_fasta": str(args.train_fasta),
            "test_esm_shard_dir": str(args.test_esm_shard_dir),
            "test_manifest_path": str(args.test_manifest_path),
            "test_homology_shard_dir": str(args.test_homology_shard_dir)
            if args.test_homology_shard_dir is not None
            else None,
            "go_vocab_path": str(args.go_vocab_path),
            "go_annotation_path": str(args.go_annotation_path),
            "obo_path": str(args.obo_path),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()

    args.checkpoint = resolve_path(args.checkpoint) if args.checkpoint is not None else None
    args.test_fasta = resolve_path(args.test_fasta)
    args.train_fasta = resolve_path(args.train_fasta)
    args.test_esm_shard_dir = resolve_path(args.test_esm_shard_dir)
    args.test_manifest_path = resolve_path(args.test_manifest_path)
    args.go_annotation_path = resolve_path(args.go_annotation_path)
    args.obo_path = resolve_path(args.obo_path)
    args.outdir = resolve_path(args.outdir)

    if args.go_vocab_path is None:
        args.go_vocab_path = PROJECT_ROOT / "diamond_db" / args.go_aspect / "go_vocab.json"
    else:
        args.go_vocab_path = resolve_path(args.go_vocab_path)

    if args.test_homology_shard_dir is None and SUPPORTED_ABLATIONS[args.ablation]["dataset_kind"] in {"sequence_homology", "homology"}:
        args.test_homology_shard_dir = PROJECT_ROOT / "diamond_db" / args.go_aspect / "test_homology_shards"
    elif args.test_homology_shard_dir is not None:
        args.test_homology_shard_dir = resolve_path(args.test_homology_shard_dir)

    args.outdir.mkdir(parents=True, exist_ok=True)
    setup_logging()
    add_file_logger(args.outdir)

    dataset_kind = SUPPORTED_ABLATIONS[args.ablation]["dataset_kind"]

    if dataset_kind != "homology" and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --ablation homology_only")

    required_paths = [
        (args.test_fasta, "test FASTA"),
        (args.train_fasta, "train FASTA"),
        (args.test_manifest_path, "test manifest CSV"),
        (args.go_vocab_path, "GO vocabulary JSON"),
        (args.go_annotation_path, "GO annotation TSV"),
        (args.obo_path, "GO OBO file"),
    ]
    if args.checkpoint is not None:
        required_paths.append((args.checkpoint, "checkpoint"))
    if dataset_kind in {"sequence", "sequence_homology"}:
        required_paths.append((args.test_esm_shard_dir, "test ESM shard directory"))
    if dataset_kind in {"sequence_homology", "homology"}:
        required_paths.append((args.test_homology_shard_dir, "test homology shard directory"))

    for path, description in required_paths:
        require_path(path, description)

    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Using device: %s", device)

    go_terms = load_go_vocab(args.go_vocab_path)
    go_term_to_idx = {go: i for i, go in enumerate(go_terms)}
    LOGGER.info("Loaded %d %s GO terms from %s", len(go_terms), args.go_aspect, args.go_vocab_path)

    # Build train IC and test targets using existing GO propagation/indexing utilities.
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

    ic = compute_ic_from_label_indices(
        label_to_indices=train_label_to_indices,
        num_go_terms=len(go_terms),
        train_ids=train_keep_ids,
    )

    LOGGER.info("Training proteins with %s labels: %d", args.go_aspect, len(train_keep_ids))
    LOGGER.info("Test proteins with %s labels: %d", args.go_aspect, len(test_keep_ids))

    if dataset_kind == "homology":
        model = None
        checkpoint = None
        LOGGER.info("Running homology-only evaluation; no checkpoint/model is loaded.")
    else:
        model, checkpoint = load_model_from_checkpoint(
            ablation=args.ablation,
            checkpoint_path=args.checkpoint,
            go_terms=go_terms,
            device=device,
        )
        LOGGER.info("Loaded checkpoint epoch=%s", checkpoint.get("epoch"))

    dataset, loader = build_test_loader(
        ablation=args.ablation,
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
    LOGGER.info("Built test loader with %d proteins and %d batches", len(dataset), len(loader))

    results = run_prediction_pass(
        model=model,
        loader=loader,
        go_terms=go_terms,
        top_k=args.top_k,
        device=device,
    )

    metrics = compute_metrics(
        y_true=results["y_true"],
        y_prob=results["y_prob"],
        ic=ic,
        steps=args.metrics_steps,
    )

    if "gate_weights" in results:
        gate_weights = results["gate_weights"]
        metrics["mean_neural_gate"] = float(gate_weights[:, 0].mean().item())
        metrics["mean_homology_gate"] = float(gate_weights[:, 1].mean().item())

    topk_csv_path = args.outdir / "topk_predictions.csv"
    metrics_json_path = args.outdir / "metrics.json"
    save_topk_csv(results["topk_rows"], topk_csv_path)
    save_metrics_json(
        metrics=metrics,
        path=metrics_json_path,
        args=args,
        checkpoint=checkpoint,
        n_proteins=results["y_prob"].shape[0],
        n_go_terms=len(go_terms),
    )

    print("\n=== Test metrics ===")
    print(f"Fmax: {metrics['Fmax']:.6f} @ threshold {metrics['Fmax_threshold']:.3f}")
    print(f"AUPR:  {metrics['AUPR']:.6f}")
    print(f"Smin raw: {metrics['Smin_raw']:.6f} @ threshold {metrics['Smin_threshold_raw']:.3f}")
    print(f"Smin normalized: {metrics['Smin_normalized']:.6f} @ threshold {metrics['Smin_threshold_normalized']:.3f}")
    if "mean_neural_gate" in metrics:
        print(f"Mean gate: neural={metrics['mean_neural_gate']:.6f}, homology={metrics['mean_homology_gate']:.6f}")

    print("\n=== Top-k prediction preview ===")
    print_prediction_preview(results["topk_rows"], args.print_limit, args.top_k)

    print(f"\nSaved metrics to: {metrics_json_path}")
    print(f"Saved full top-{args.top_k} predictions to: {topk_csv_path}")


if __name__ == "__main__":
    main()
