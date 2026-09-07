"""Create mutually exclusive identity bins from final retained DIAMOND hits.

The source of truth for a bin is deliberately the same retained-hit policy used
by :func:`build_aligned_homology_shards`: parse the shared DIAMOND result,
deduplicate query/subject pairs, apply the E-value and query-coverage filters,
then retain the top K hits.  This is not a re-search and does not use the
legacy cumulative similarity-bin CSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reliability_aware.utils.diamond_homology import (
    DiamondSearchConfig,
    HomologyHit,
    _parse_diamond_hits,
    _retain_valid_hits,
    read_fasta_as_dict,
)
from reliability_aware.utils.parser import get_protein_info


CATEGORY_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("identity_lt30", "I_max < 30"),
    ("identity_30_lt40", "30 <= I_max < 40"),
    ("identity_40_lt50", "40 <= I_max < 50"),
    ("identity_50_lt70", "50 <= I_max < 70"),
    ("identity_70_lt95", "70 <= I_max < 95"),
    ("identity_ge95", "95 <= I_max <= 100"),
    ("no_retained_hit", "H_q is empty"),
)

ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    "test_protein_id",
    "category",
    "max_retained_identity",
    "retained_hit_count",
    "max_identity_target_id",
    "selected_alignment_bitscore",
    "selected_alignment_query_coverage_fraction",
    "selected_alignment_query_coverage_percent",
    "selected_alignment_evalue",
    "selected_alignment_qstart",
    "selected_alignment_qend",
    "selected_alignment_sstart",
    "selected_alignment_send",
    "selected_alignment_length",
    "selected_alignment_query_length",
    "selected_alignment_target_length",
)


def category_for_identity(identity: float | None) -> str:
    """Return the single bin for an unrounded DIAMOND percentage identity."""
    if identity is None:
        return "no_retained_hit"
    if not 0.0 <= identity <= 100.0:
        raise ValueError(f"DIAMOND percent identity is outside [0, 100]: {identity}")
    if identity < 30.0:
        return "identity_lt30"
    if identity < 40.0:
        return "identity_30_lt40"
    if identity < 50.0:
        return "identity_40_lt50"
    if identity < 70.0:
        return "identity_50_lt70"
    if identity < 95.0:
        return "identity_70_lt95"
    return "identity_ge95"


def retained_hits_for_queries(
    query_ids: Iterable[str],
    hits_tsv: Path,
    config: DiamondSearchConfig,
) -> dict[str, list[HomologyHit]]:
    """Return final retained hits for every requested test query.

    Test queries are never self-filtered because the database contains training
    proteins only.  Keeping this call here makes the equivalence to test shard
    construction explicit and avoids accidentally treating absent raw rows as
    an identity of zero.
    """
    parsed = _parse_diamond_hits(hits_tsv)
    return {
        query_id: _retain_valid_hits(parsed.get(query_id, []), config)
        for query_id in query_ids
    }


def audit_retention_pipeline(
    query_ids: Sequence[str], hits_tsv: Path, config: DiamondSearchConfig
) -> dict[str, int]:
    """Count each stage of the retained-hit pipeline without changing it."""
    raw_rows = 0
    raw_pairs: set[tuple[str, str]] = set()
    raw_queries: set[str] = set()
    with hits_tsv.open("r", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if len(row) != 9:
                raise ValueError(
                    f"Expected nine DIAMOND outfmt fields, found {len(row)} in {hits_tsv}."
                )
            raw_rows += 1
            raw_queries.add(row[0])
            raw_pairs.add((row[0], row[1]))

    parsed = _parse_diamond_hits(hits_tsv)
    query_set = set(query_ids)
    unexpected_queries = raw_queries - query_set
    if unexpected_queries:
        preview = ", ".join(sorted(unexpected_queries)[:5])
        raise ValueError(f"DIAMOND result contains query IDs outside test FASTA: {preview}")

    eligible_before_top_k = 0
    final_retained = 0
    queries_with_raw_hits = 0
    queries_with_eligible_hits = 0
    top_k_discarded = 0
    for query_id in query_ids:
        candidates = parsed.get(query_id, [])
        if candidates:
            queries_with_raw_hits += 1
        eligible = [
            hit
            for hit in candidates
            if hit.evalue <= config.evalue_max and hit.qcov >= config.min_query_coverage
        ]
        eligible_before_top_k += len(eligible)
        if eligible:
            queries_with_eligible_hits += 1
        retained = _retain_valid_hits(candidates, config)
        final_retained += len(retained)
        top_k_discarded += max(0, len(eligible) - len(retained))

    return {
        "raw_alignment_rows": raw_rows,
        "raw_distinct_query_subject_pairs": len(raw_pairs),
        "duplicate_alignment_rows_discarded": raw_rows - len(raw_pairs),
        "raw_query_count": len(raw_queries),
        "queries_with_raw_hits": queries_with_raw_hits,
        "queries_with_no_raw_hits": len(query_ids) - queries_with_raw_hits,
        "queries_with_eligible_hits_before_top_k": queries_with_eligible_hits,
        "queries_with_raw_hits_but_none_eligible": queries_with_raw_hits
        - queries_with_eligible_hits,
        "eligible_hits_before_top_k": eligible_before_top_k,
        "hits_discarded_by_top_k": top_k_discarded,
        "final_retained_hit_count": final_retained,
    }


def select_max_identity_hit(hits: Sequence[HomologyHit]) -> HomologyHit | None:
    """Select an audit alignment without affecting the bin assignment.

    Equal identities are resolved by decreasing bitscore, decreasing query
    coverage, increasing E-value significance, target ID, alignment length,
    query length, and target length.  The first criterion, identity, is always
    the unrounded DIAMOND ``pident`` value.
    """
    if not hits:
        return None
    return min(
        hits,
        key=lambda hit: (
            -hit.pident,
            -hit.bitscore,
            -hit.qcov,
            hit.evalue,
            hit.sseqid,
            -hit.length,
            hit.qlen,
            hit.slen,
        ),
    )


def make_assignment_rows(
    query_ids: Sequence[str],
    retained_by_query: Mapping[str, Sequence[HomologyHit]],
) -> list[dict[str, Any]]:
    """Construct one complete assignment row per test protein."""
    rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        hits = retained_by_query[query_id]
        selected = select_max_identity_hit(hits)
        if selected is None:
            rows.append(
                {
                    "test_protein_id": query_id,
                    "category": "no_retained_hit",
                    "max_retained_identity": None,
                    "retained_hit_count": 0,
                    "max_identity_target_id": None,
                    "selected_alignment_bitscore": None,
                    "selected_alignment_query_coverage_fraction": None,
                    "selected_alignment_query_coverage_percent": None,
                    "selected_alignment_evalue": None,
                    # qstart/qend/sstart/send were not requested in the saved
                    # DIAMOND outfmt.  Empty fields record unavailable evidence,
                    # rather than implying coordinates of zero.
                    "selected_alignment_qstart": None,
                    "selected_alignment_qend": None,
                    "selected_alignment_sstart": None,
                    "selected_alignment_send": None,
                    "selected_alignment_length": None,
                    "selected_alignment_query_length": None,
                    "selected_alignment_target_length": None,
                }
            )
            continue

        rows.append(
            {
                "test_protein_id": query_id,
                "category": category_for_identity(selected.pident),
                "max_retained_identity": selected.pident,
                "retained_hit_count": len(hits),
                "max_identity_target_id": selected.sseqid,
                "selected_alignment_bitscore": selected.bitscore,
                "selected_alignment_query_coverage_fraction": selected.qcov,
                "selected_alignment_query_coverage_percent": 100.0 * selected.qcov,
                "selected_alignment_evalue": selected.evalue,
                "selected_alignment_qstart": None,
                "selected_alignment_qend": None,
                "selected_alignment_sstart": None,
                "selected_alignment_send": None,
                "selected_alignment_length": selected.length,
                "selected_alignment_query_length": selected.qlen,
                "selected_alignment_target_length": selected.slen,
            }
        )
    return rows


def validate_assignments(
    query_ids: Sequence[str],
    train_ids: set[str],
    rows: Sequence[Mapping[str, Any]],
    retained_by_query: Mapping[str, Sequence[HomologyHit]],
) -> dict[str, Any]:
    """Fail loudly when evidence or mutually-exclusive membership is invalid."""
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("The authoritative test FASTA contains duplicate protein IDs.")
    if len(rows) != len(query_ids):
        raise ValueError("Assignment row count does not equal test protein count.")
    if [row["test_protein_id"] for row in rows] != list(query_ids):
        raise ValueError("Assignment rows are not a one-to-one ordered test-set mapping.")

    known_categories = {name for name, _ in CATEGORY_DEFINITIONS}
    counts = Counter(str(row["category"]) for row in rows)
    if set(counts) - known_categories:
        raise ValueError(f"Unexpected categories: {sorted(set(counts) - known_categories)}")

    for row in rows:
        query_id = str(row["test_protein_id"])
        hits = retained_by_query[query_id]
        identity = row["max_retained_identity"]
        if not hits:
            if row["category"] != "no_retained_hit" or identity is not None:
                raise ValueError(f"{query_id}: empty hit set was not recorded as missing.")
            continue
        if identity is None:
            raise ValueError(f"{query_id}: nonempty retained hit set has missing identity.")
        if row["category"] != category_for_identity(float(identity)):
            raise ValueError(f"{query_id}: category does not match unrounded identity.")
        if int(row["retained_hit_count"]) != len(hits):
            raise ValueError(f"{query_id}: retained-hit count does not match final hit set.")
        missing_targets = sorted({hit.sseqid for hit in hits} - train_ids)
        if missing_targets:
            raise ValueError(
                f"{query_id}: retained target(s) absent from training FASTA: "
                f"{', '.join(missing_targets[:5])}"
            )
        if any(not 0.0 <= hit.pident <= 100.0 for hit in hits):
            raise ValueError(f"{query_id}: invalid retained DIAMOND pident value.")

    return {
        "test_protein_count": len(query_ids),
        "unique_test_protein_count": len(set(query_ids)),
        "assignment_row_count": len(rows),
        "category_counts": {name: counts[name] for name, _ in CATEGORY_DEFINITIONS},
        "categories_pairwise_disjoint": True,
        "categories_union_equals_test_set": sum(counts.values()) == len(query_ids),
        "all_retained_targets_in_training_database": True,
        "all_nonempty_retained_hit_sets_have_identity": True,
        "no_retained_hit_rows_have_empty_final_hit_sets": True,
        "boundary_policy": {
            "30": "identity_30_lt40",
            "40": "identity_40_lt50",
            "50": "identity_50_lt70",
            "70": "identity_70_lt95",
            "95": "identity_ge95",
            "100": "identity_ge95",
        },
    }


def create_retained_hit_identity_bins(
    *,
    test_fasta: Path,
    train_fasta: Path,
    hits_tsv: Path,
    output_dir: Path,
    config: DiamondSearchConfig,
    command: Sequence[str] | None = None,
    diamond_version: str | None = None,
    pipeline_log: Path | None = None,
) -> dict[str, Any]:
    """Create assignments, per-bin ID/FASTA exports, and audit metadata."""
    # Read test records before constructing a mapping: a dict would silently
    # overwrite duplicate FASTA IDs and make an omission look valid.
    test_records = get_protein_info(test_fasta)
    query_ids = [record["full_id"] for record in test_records]
    if len(query_ids) != len(set(query_ids)):
        duplicates = sorted(
            protein_id
            for protein_id, count in Counter(query_ids).items()
            if count > 1
        )
        raise ValueError(
            "The authoritative test FASTA contains duplicate protein IDs: "
            f"{', '.join(duplicates[:5])}"
        )
    test_sequences = {record["full_id"]: record["sequence"] for record in test_records}
    train_sequences = read_fasta_as_dict(train_fasta)
    if not query_ids:
        raise ValueError("The authoritative test FASTA has no protein records.")

    retained = retained_hits_for_queries(query_ids, hits_tsv, config)
    retention_audit = audit_retention_pipeline(query_ids, hits_tsv, config)
    rows = make_assignment_rows(query_ids, retained)
    validation = validate_assignments(query_ids, set(train_sequences), rows, retained)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "assignments.csv", ASSIGNMENT_COLUMNS, rows)

    rows_by_category: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name, _ in CATEGORY_DEFINITIONS
    }
    for row in rows:
        rows_by_category[str(row["category"])].append(row)
    for category, _definition in CATEGORY_DEFINITIONS:
        ids = [str(row["test_protein_id"]) for row in rows_by_category[category]]
        _write_id_list(output_dir / f"{category}.ids.txt", ids)
        _write_fasta(output_dir / f"{category}.fasta", ids, test_sequences)

    summary_rows = [
        {
            "category": category,
            "definition": definition,
            "protein_count": len(rows_by_category[category]),
            "percentage_of_full_test_set": 100.0 * len(rows_by_category[category]) / len(query_ids),
        }
        for category, definition in CATEGORY_DEFINITIONS
    ]
    _write_csv(
        output_dir / "summary.csv",
        ("category", "definition", "protein_count", "percentage_of_full_test_set"),
        summary_rows,
    )

    exported_ids = [
        protein_id
        for category, _ in CATEGORY_DEFINITIONS
        for protein_id in _read_id_list(output_dir / f"{category}.ids.txt")
    ]
    if set(exported_ids) != set(query_ids) or len(exported_ids) != len(set(exported_ids)):
        raise ValueError("Exported ID/FASTA membership does not match assignments.")
    for category, _definition in CATEGORY_DEFINITIONS:
        expected_ids = [
            str(row["test_protein_id"]) for row in rows_by_category[category]
        ]
        id_list_ids = _read_id_list(output_dir / f"{category}.ids.txt")
        fasta_ids = list(read_fasta_as_dict(output_dir / f"{category}.fasta"))
        if id_list_ids != expected_ids or fasta_ids != expected_ids:
            raise ValueError(
                f"Exported ID/FASTA membership does not match assignments for {category}."
            )
    validation["exported_id_and_fasta_membership_agrees_with_assignments"] = True
    validation["expected_test_protein_count_3416"] = len(query_ids) == 3416

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command or sys.argv),
        "input_artifacts": {
            "test_fasta": _artifact_record(test_fasta),
            "training_fasta": _artifact_record(train_fasta),
            "diamond_hits_tsv": _artifact_record(hits_tsv),
            **(
                {"completed_run_pipeline_log": _artifact_record(pipeline_log)}
                if pipeline_log is not None and pipeline_log.is_file()
                else {}
            ),
        },
        "retention_policy": {
            "duplicate_alignment_resolution": "best per (qseqid, sseqid): bitscore desc, qcov desc, evalue asc, pident desc; exact ties retain first TSV record",
            "eligibility": "E-value <= evalue_max and qcovhsp / 100 >= min_query_coverage; no GO annotation filter",
            "ranking_and_top_k": "bitscore desc, qcov desc, evalue asc, sseqid asc; retain first top_k",
            "test_self_hit_removal": False,
            "prior_construction_removes_hits_after_retention": False,
            "identity_source": "DIAMOND outfmt pident, retained alignment only; unrounded percentage",
            "alignment_coordinates": "not available: the saved outfmt lacks qstart qend sstart send",
        },
        "retention_pipeline_audit": retention_audit,
        "search_and_retention_config": asdict(config),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "diamond_version": diamond_version,
        },
        "aspect_equivalence": {
            "raw_hits_are_shared": True,
            "retention_occurs_before_go_aspect_annotation_lookup": True,
            "result": "BP, MF, and CC use the same final retained hit membership under this pipeline; only GO-term projection differs.",
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(_output_readme(), encoding="utf-8")
    return {"summary": summary_rows, "validation": validation, "rows": rows}


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_id_list(path: Path, ids: Iterable[str]) -> None:
    path.write_text("".join(f"{protein_id}\n" for protein_id in ids), encoding="utf-8")


def _write_fasta(path: Path, ids: Iterable[str], sequences: Mapping[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for protein_id in ids:
            sequence = sequences[protein_id]
            handle.write(f">{protein_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(f"{sequence[start:start + 80]}\n")


def _read_id_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _artifact_record(path: Path) -> dict[str, str | int]:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def get_diamond_version(executable: str | Path | None) -> str | None:
    """Return DIAMOND's reported version when the executable is available."""
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [str(executable), "version"], check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _output_readme() -> str:
    return """# Retained-hit identity bins

`assignments.csv` is the sole membership source for gate and predictive-performance comparisons. Both `sequence_homology_confidence_gate` and `sequence_homology_internal_gate` must filter their test predictions by `test_protein_id` and `category` from this same table; neither evaluation may recompute bins or change homology priors.

The categories are mutually exclusive and exhaustive over the authoritative test FASTA. `max_retained_identity` is the unrounded DIAMOND `pident` of the maximum-identity alignment among the final retained hits only. Missing identity is represented as an empty CSV cell exclusively for `no_retained_hit` rows.

`selected_alignment_query_coverage_fraction` is the 0--1 value used by the prior pipeline (and `selected_alignment_query_coverage_percent` is provided for inspection). `provenance.json` identifies the raw shared DIAMOND records and exact filtering/ranking policy. The original result did not request alignment-coordinate fields, so their audit columns in `assignments.csv` are intentionally empty rather than reconstructed from a new search.
"""
