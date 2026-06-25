from __future__ import annotations
import csv
import json
import logging
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Mapping, Sequence
from reliability_aware.utils.config import PROJECT_ROOT
from reliability_aware.utils.diamond_executable import (
    diamond_resolution_error,
    resolve_diamond_executable,
)
import torch

logger = logging.getLogger(__name__)


def _default_diamond_executable() -> str:
    resolved = resolve_diamond_executable()
    return str(resolved) if resolved is not None else "diamond"


MANIFEST_REQUIRED_COLUMNS = {
    "shard_number",
    "local_seq_idx",
    "global_seq_idx",
    "label",
    "sequence_length",
    "truncated_length",
}

DIAMOND_OUTFMT_FIELDS = [
    "qseqid",
    "sseqid",
    "evalue",
    "bitscore",
    "qcovhsp",
    "qlen",
    "slen",
    "length",
    "pident",
]


@dataclass(frozen=True)
class DiamondSearchConfig:
    diamond_exe: str = field(default_factory=_default_diamond_executable)
    evalue_max: float = 1e-5
    min_query_coverage: float = 0.30
    max_target_seqs: int = 50
    top_k: int = 10
    sensitivity: str = (
        "sensitive"  # default, fast, mid-sensitive, sensitive, more-sensitive, very-sensitive, ultra-sensitive
    )
    iterate: bool = True
    threads: int | None = None


@dataclass(frozen=True)
class HomologyHit:
    qseqid: str
    sseqid: str
    evalue: float
    bitscore: float
    qcov: float  # stored on 0-1 scale
    qlen: int
    slen: int
    length: int
    pident: float


@dataclass(frozen=True)
class RetrievalStats:
    b_max: float
    cov_max: float
    n_hits: int
    has_hit: float

    @property
    def log1p_n_hits(self) -> float:
        return math.log1p(self.n_hits)

    def as_gate_features(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.b_max,
                self.cov_max,
                self.log1p_n_hits,
                self.has_hit,
            ],
            dtype=torch.float32,
        )


def read_fasta_as_dict(fasta_path: str | Path) -> dict[str, str]:
    """
    Read a FASTA file into a mapping from sequence ID to sequence.

    Important:
    This assumes the sequence ID is the first whitespace-delimited token
    after '>' and that it matches your manifest label / FASTA full_id.
    """
    fasta_path = Path(fasta_path).resolve()
    records: dict[str, str] = {}

    current_id: str | None = None
    chunks: list[str] = []

    with fasta_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        records[current_id] = "".join(chunks)

    return records


def write_fasta_from_ids(
    ids,
    sequence_lookup: Mapping[str, str],
    output_fasta_path: str | Path,
) -> Path:
    """
    Write a FASTA file in the exact label order requested by `ids`.
    """
    output_fasta_path = Path(output_fasta_path).resolve()
    output_fasta_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [seq_id for seq_id in ids if seq_id not in sequence_lookup]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"Missing {len(missing)} sequence IDs in FASTA lookup. Examples: {preview}"
        )

    with output_fasta_path.open("w") as handle:
        for seq_id in ids:
            handle.write(f">{seq_id}\n")
            handle.write(f"{sequence_lookup[seq_id]}\n")

    return output_fasta_path


def save_go_vocab(go_terms: Sequence[str], output_json_path: str | Path) -> Path:
    output_json_path = Path(output_json_path).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(list(go_terms), indent=2))
    return output_json_path


def build_diamond_database(
    training_fasta_path: str | Path,
    db_prefix: str | Path,
    config: DiamondSearchConfig,
) -> Path:
    """
    Build a DIAMOND database from training proteins only.
    """
    training_fasta_path = Path(training_fasta_path).resolve()
    db_prefix = Path(db_prefix).resolve()
    db_prefix.parent.mkdir(parents=True, exist_ok=True)

    diamond_executable = resolve_diamond_executable(explicit=config.diamond_exe)
    if diamond_executable is None:
        raise FileNotFoundError(diamond_resolution_error())

    cmd = [
        str(diamond_executable),
        "makedb",
        "--in",
        str(training_fasta_path),
        "--db",
        str(db_prefix),
    ]
    if config.threads is not None:
        cmd.extend(["--threads", str(config.threads)])

    logger.info("Running DIAMOND makedb")
    _run_subprocess(cmd)
    return db_prefix.with_suffix(".dmnd")


def run_diamond_blastp(
    query_fasta_path: str | Path,
    db_prefix: str | Path,
    output_tsv_path: str | Path,
    config: DiamondSearchConfig,
) -> Path:
    """
    Run DIAMOND blastp against a prebuilt training database.

    Notes:
    - We ask DIAMOND to emit qcovhsp directly.
    - Filtering by E-value and query coverage is still re-applied downstream
      from the parsed TSV so the branch logic stays explicit and reproducible.
    """
    query_fasta_path = Path(query_fasta_path).resolve()
    db_prefix = Path(db_prefix).resolve()
    output_tsv_path = Path(output_tsv_path).resolve()
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        config.diamond_exe,
        "blastp",
        "--db",
        str(db_prefix),
        "--query",
        str(query_fasta_path),
        "--out",
        str(output_tsv_path),
        "--outfmt",
        "6",
        *DIAMOND_OUTFMT_FIELDS,
        "--evalue",
        f"{config.evalue_max:g}",
        "--max-target-seqs",
        str(config.max_target_seqs),
    ]

    if config.sensitivity != "default":
        cmd.append(f"--{config.sensitivity}")

    if config.iterate:
        cmd.append("--iterate")

    if config.threads is not None:
        cmd.extend(["--threads", str(config.threads)])

    logger.info("Running DIAMOND blastp")
    _run_subprocess(cmd)
    return output_tsv_path


def build_aligned_homology_shards(
    manifest_path: str | Path,
    diamond_hits: str | Path,
    subject_go_index_json_path: str | Path,
    go_vocab_json_path: str | Path,
    output_dir: str | Path,
    config: DiamondSearchConfig,
    *,
    exclude_self_hits: bool = False,
    use_fp16: bool = False,
    keep_debug_hits: bool = True,
) -> Path:
    """
    Build homology shards aligned 1:1 with the ESM shards and local indices.

      homology_shard_{k:04d}.pt["priors"][i]
      corresponds to
      esm shard part_{k:04d}.pt["representations"][i]

    Each shard stores:
      - priors[i]: (num_go_terms,) tensor
      - gate_features[i]: tensor([b_max, cov_max, log1p_n_hits, has_hit])
      - stats[i]: dict with raw retrieval stats
      - debug_hits[i]: optional retained-hit summaries
      - labels[i]: manifest label
    """
    manifest_rows = _load_manifest_rows(manifest_path)
    hits_by_query = _parse_diamond_hits(diamond_hits)
    subject_to_go_indices = _load_subject_go_index(subject_go_index_json_path)
    go_vocab = _load_go_vocab(go_vocab_json_path)
    num_go_terms = len(go_vocab)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_shard: dict[int, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        rows_by_shard[row["shard_number"]].append(row)

    num_queries = 0
    num_with_hits = 0

    for shard_id in sorted(rows_by_shard):
        rows = sorted(rows_by_shard[shard_id], key=lambda r: r["local_seq_idx"])
        expected = list(range(len(rows)))
        observed = [row["local_seq_idx"] for row in rows]
        if observed != expected:
            raise ValueError(
                f"Shard {shard_id} local_seq_idx mismatch. "
                f"Expected {expected[:5]}..., got {observed[:5]}..."
            )

        priors: list[torch.Tensor] = []
        gate_features: list[torch.Tensor] = []
        stats_records: list[dict] = []
        debug_hits_records: list[list[dict] | None] = []
        labels: list[str] = []

        for row in rows:
            label = row["label"]
            query_hits = hits_by_query.get(label, [])
            retained = _retain_valid_hits(
                query_hits,
                config,
                exclude_self=exclude_self_hits,
                query_id=label,
            )
            prior, stats, debug_hits = _build_prior_from_hits(
                hits=retained,
                subject_to_go_indices=subject_to_go_indices,
                num_go_terms=num_go_terms,
            )

            if use_fp16:
                prior = prior.half()

            priors.append(prior)
            gate_features.append(stats.as_gate_features())
            stats_records.append(asdict(stats))
            debug_hits_records.append(debug_hits if keep_debug_hits else None)
            labels.append(label)

            num_queries += 1
            if stats.has_hit > 0:
                num_with_hits += 1

        shard_path = output_dir / f"homology_shard_{shard_id:04d}.pt"
        torch.save(
            {
                "priors": priors,
                "gate_features": gate_features,
                "stats": stats_records,
                "debug_hits": debug_hits_records,
                "labels": labels,
                "num_go_terms": num_go_terms,
            },
            shard_path,
        )

    metadata = {
        "num_queries": num_queries,
        "num_queries_with_hits": num_with_hits,
        "hit_rate": (num_with_hits / max(num_queries, 1)),
        "num_go_terms": num_go_terms,
        "config": asdict(config),
        "aligned_to": "ESM manifest shard_number/local_seq_idx order",
    }
    metadata_path = output_dir / "homology_shard_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata_path


def _load_manifest_rows(manifest_path: str | Path) -> list[dict]:
    """Load the ESM manifest in row order."""
    manifest_path = Path(manifest_path).resolve()
    rows: list[dict] = []

    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = MANIFEST_REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                {
                    "shard_number": int(row["shard_number"]),
                    "local_seq_idx": int(row["local_seq_idx"]),
                    "global_seq_idx": int(row["global_seq_idx"]),
                    "label": row["label"],
                    "sequence_length": int(row["sequence_length"]),
                    "truncated_length": int(row["truncated_length"]),
                }
            )

    return rows


def _load_subject_go_index(input_json_path: str | Path) -> dict[str, torch.Tensor]:
    """
    Load label -> GO index lists and convert them to CPU LongTensors for fast updates.
    """
    input_json_path = Path(input_json_path).resolve()
    raw = json.loads(input_json_path.read_text())
    return {
        label: torch.tensor(indices, dtype=torch.long) for label, indices in raw.items()
    }


def _load_go_vocab(input_json_path: str | Path) -> list[str]:
    input_json_path = Path(input_json_path).resolve()
    return list(json.loads(input_json_path.read_text()))


def _parse_diamond_hits(tsv_path: str | Path) -> dict[str, list[HomologyHit]]:
    """
    Parse DIAMOND TSV output and keep only the best row per (query, subject).

    This guards against repeated HSP-like rows for the same subject.
    """
    tsv_path = Path(tsv_path).resolve()
    best_by_pair: dict[tuple[str, str], HomologyHit] = {}

    with tsv_path.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue

            hit = HomologyHit(
                qseqid=row[0],
                sseqid=row[1],
                evalue=float(row[2]),
                bitscore=float(row[3]),
                qcov=float(row[4]) / 100.0,
                qlen=int(row[5]),
                slen=int(row[6]),
                length=int(row[7]),
                pident=float(row[8]),
            )

            key = (hit.qseqid, hit.sseqid)
            prev = best_by_pair.get(key)
            if prev is None or _hit_is_better(hit, prev):
                best_by_pair[key] = hit

    hits_by_query: dict[str, list[HomologyHit]] = defaultdict(list)
    for (_, _), hit in best_by_pair.items():
        hits_by_query[hit.qseqid].append(hit)

    for query_id, hits in hits_by_query.items():
        hits.sort(key=lambda h: (-h.bitscore, -h.qcov, h.evalue, h.sseqid))
        hits_by_query[query_id] = hits

    return dict(hits_by_query)


def _retain_valid_hits(
    hits: Sequence[HomologyHit],
    config: DiamondSearchConfig,
    *,
    exclude_self: bool = False,
    query_id: str | None = None,
) -> list[HomologyHit]:
    """
    Apply the fixed homology branch policy:
    1. filter by E-value and query coverage
    2. optionally remove self-hits
    3. sort by strongest alignment
    4. keep top K
    """
    valid: list[HomologyHit] = []
    for hit in hits:
        if exclude_self and query_id is not None and hit.sseqid == query_id:
            continue
        if hit.evalue > config.evalue_max:
            continue
        if hit.qcov < config.min_query_coverage:
            continue
        valid.append(hit)

    valid.sort(key=lambda h: (-h.bitscore, -h.qcov, h.evalue, h.sseqid))
    return valid[: config.top_k]


def _build_prior_from_hits(
    hits: Sequence[HomologyHit],
    subject_to_go_indices: Mapping[str, torch.Tensor],
    num_go_terms: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, RetrievalStats, list[dict]]:
    """
    Construct the homology prior probability vector p^(h).

    Weight:
        w_h = (bitscore_h * qcov_h) / sum_h' (bitscore_h' * qcov_h' + eps)

    Prior:
        p_k^(h) = sum_h w_h * 1[k in A(h)]

    Returns:
        prior: (num_go_terms,) float tensor on CPU
        stats: RetrievalStats for gate features
        retained_hits_debug: list of per-hit summaries including the final weight
    """
    prior = torch.zeros(num_go_terms, dtype=torch.float32)

    if not hits:
        zero_stats = RetrievalStats(
            b_max=0.0,
            cov_max=0.0,
            n_hits=0,
            has_hit=0.0,
        )
        return prior, zero_stats, []

    raw_weights = [max(hit.bitscore, 0.0) * max(hit.qcov, 0.0) for hit in hits]
    denom = float(sum(raw_weights)) + eps

    retained_hits_debug: list[dict] = []
    for hit, raw_weight in zip(hits, raw_weights):
        weight = raw_weight / denom
        go_indices = subject_to_go_indices.get(hit.sseqid)
        if go_indices is not None and go_indices.numel() > 0:
            prior[go_indices] += weight

        retained_hits_debug.append(
            {
                "sseqid": hit.sseqid,
                "evalue": hit.evalue,
                "bitscore": hit.bitscore,
                "qcov": hit.qcov,
                "weight": weight,
            }
        )

    prior.clamp_(0.0, 1.0)

    stats = RetrievalStats(
        b_max=max(hit.bitscore for hit in hits),
        cov_max=max(hit.qcov for hit in hits),
        n_hits=len(hits),
        has_hit=1.0,
    )
    return prior, stats, retained_hits_debug


def _hit_is_better(current: HomologyHit, previous: HomologyHit) -> bool:
    """
    Prefer:
      1. higher bitscore
      2. higher query coverage
      3. lower E-value
      4. higher percent identity
    """
    if current.bitscore != previous.bitscore:
        return current.bitscore > previous.bitscore
    if current.qcov != previous.qcov:
        return current.qcov > previous.qcov
    if current.evalue != previous.evalue:
        return current.evalue < previous.evalue
    return current.pident > previous.pident


def _run_subprocess(cmd: Sequence[str]) -> None:
    logger.info("Command: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Executable not found: {cmd[0]!r}. "
            "Install DIAMOND and ensure it is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"External command failed with exit code {exc.returncode}: {' '.join(cmd)}"
        ) from exc
