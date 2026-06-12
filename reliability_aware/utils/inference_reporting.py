"""GO metadata enrichment and inference output formatting."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from goatools.obo_parser import GODag

from reliability_aware.utils.inference_runtime import require_path

LOGGER = logging.getLogger("run_inference")
NAMESPACE_TO_ASPECT = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}


def _clean_obo_definition(raw_definition: str | None) -> str:
    if not raw_definition:
        return ""
    definition = raw_definition.strip()
    if definition.startswith('"') and '" [' in definition:
        definition = definition[1:].rsplit('" [', 1)[0]
    return definition.replace(r"\"", '"')


def load_go_term_metadata(
    obo_path: Path,
    go_terms: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Load display metadata for an ordered GO vocabulary."""
    require_path(obo_path, "GO OBO file")
    go_dag = GODag(str(obo_path), optional_attrs={"def"}, prt=None)
    metadata: dict[str, dict[str, str]] = {}

    for go_id in go_terms:
        term = go_dag.get(go_id)
        if term is None:
            metadata[go_id] = {"go_name": "", "aspect": "", "definition": ""}
            LOGGER.warning("GO term is missing from the OBO file: %s", go_id)
            continue
        metadata[go_id] = {
            "go_name": term.name or "",
            "aspect": NAMESPACE_TO_ASPECT.get(term.namespace, term.namespace or ""),
            "definition": _clean_obo_definition(getattr(term, "defn", None)),
        }

    return metadata


def enrich_topk_rows(
    rows: list[dict[str, Any]],
    go_metadata: Mapping[str, Mapping[str, str]],
) -> None:
    """Add OBO-derived display fields to top-k prediction rows in place."""
    for row in rows:
        metadata = go_metadata.get(row["go_term"], {})
        row["go_name"] = metadata.get("go_name", "")
        row["aspect"] = metadata.get("aspect", "")
        row["definition"] = metadata.get("definition", "")


def save_topk_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "protein_id",
        "global_idx",
        "rank",
        "go_id",
        "go_name",
        "aspect",
        "probability_score",
        "definition",
        "neural_probability",
        "homology_probability",
        "neural_gate",
        "homology_gate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["go_id"] = csv_row.pop("go_term")
            csv_row["probability_score"] = csv_row.pop("probability")
            writer.writerow(csv_row)


def build_per_protein_gate_rows(
    *,
    labels: list[str],
    global_indices: list[int],
    gate_weights: torch.Tensor,
) -> list[dict[str, Any]]:
    """Build one gate-score row for every protein processed."""
    n_proteins = len(labels)
    if len(global_indices) != n_proteins or gate_weights.shape != (n_proteins, 2):
        raise ValueError(
            "Cannot align per-protein gate scores: "
            f"labels={n_proteins}, global_indices={len(global_indices)}, "
            f"gate_weights_shape={tuple(gate_weights.shape)}"
        )

    return [
        {
            "protein_id": protein_id,
            "global_idx": global_idx,
            "neural_gate": float(weights[0].item()),
            "homology_gate": float(weights[1].item()),
        }
        for protein_id, global_idx, weights in zip(
            labels, global_indices, gate_weights
        )
    ]


def save_per_protein_gate_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["protein_id", "global_idx", "neural_gate", "homology_gate"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_prediction_preview(
    rows: list[dict[str, Any]], print_limit: int, top_k: int
) -> None:
    if print_limit == 0:
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["protein_id"], []).append(row)

    for printed, (protein_id, protein_rows) in enumerate(grouped.items()):
        if print_limit >= 0 and printed >= print_limit:
            break

        print(f"\n{protein_id}")
        widths = {
            "rank": 4,
            "go_id": 10,
            "go_name": 30,
            "aspect": 6,
            "score": 17,
            "definition": 70,
        }
        header = (
            f"{'Rank':<{widths['rank']}} | "
            f"{'GO ID':<{widths['go_id']}} | "
            f"{'GO name':<{widths['go_name']}} | "
            f"{'Aspect':<{widths['aspect']}} | "
            f"{'Probability Score':<{widths['score']}} | Definition"
        )
        print(header)
        print("-" * len(header))

        for pred in protein_rows[:top_k]:
            name_lines = textwrap.wrap(
                pred.get("go_name", "") or "", width=widths["go_name"]
            ) or [""]
            definition_lines = textwrap.wrap(
                pred.get("definition", "") or "", width=widths["definition"]
            ) or [""]
            line_count = max(len(name_lines), len(definition_lines))

            for line_index in range(line_count):
                rank = str(pred["rank"]) if line_index == 0 else ""
                go_id = pred["go_term"] if line_index == 0 else ""
                aspect = pred.get("aspect", "") if line_index == 0 else ""
                score = f"{pred['probability']:.6f}" if line_index == 0 else ""
                name = name_lines[line_index] if line_index < len(name_lines) else ""
                definition = (
                    definition_lines[line_index]
                    if line_index < len(definition_lines)
                    else ""
                )
                print(
                    f"{rank:<{widths['rank']}} | "
                    f"{go_id:<{widths['go_id']}} | "
                    f"{name:<{widths['go_name']}} | "
                    f"{aspect:<{widths['aspect']}} | "
                    f"{score:<{widths['score']}} | {definition}"
                )


def save_metrics_json(
    *,
    metrics: dict[str, float | int | str],
    path: Path,
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> None:
    payload = {
        "metrics": metrics,
        "metric_style": "cafa",
        "n_proteins_evaluated": metrics["n_proteins_evaluated"],
        "n_go_terms": metrics["n_go_terms"],
        "ablation": args.ablation,
        "go_aspect": args.go_aspect,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "checkpoint_epoch": checkpoint.get("epoch") if checkpoint is not None else None,
        "paths": {
            "test_fasta": str(args.test_fasta),
            "train_fasta": str(args.train_fasta),
            "test_esm_shard_dir": str(args.test_esm_shard_dir),
            "test_manifest_path": str(args.test_manifest_path),
            "test_homology_shard_dir": (
                str(args.test_homology_shard_dir)
                if args.test_homology_shard_dir is not None
                else None
            ),
            "go_vocab_path": str(args.go_vocab_path),
            "go_annotation_path": str(args.go_annotation_path),
            "obo_path": str(args.obo_path),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_prediction_metadata(
    *,
    path: Path,
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
    n_proteins: int,
    n_go_terms: int,
) -> None:
    payload = {
        "mode": args.mode,
        "n_proteins": n_proteins,
        "n_go_terms": n_go_terms,
        "ablation": args.ablation,
        "go_aspect": args.go_aspect,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "checkpoint_epoch": checkpoint.get("epoch") if checkpoint is not None else None,
        "paths": {
            "test_esm_shard_dir": str(args.test_esm_shard_dir),
            "test_manifest_path": str(args.test_manifest_path),
            "test_homology_shard_dir": (
                str(args.test_homology_shard_dir)
                if args.test_homology_shard_dir is not None
                else None
            ),
            "go_vocab_path": str(args.go_vocab_path),
            "obo_path": str(args.obo_path),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
