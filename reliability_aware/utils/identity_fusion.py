"""Build auditable top-five bitscore-weighted retained-hit identity sidecars."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from reliability_aware.utils.diamond_homology import _parse_diamond_hits, _retain_valid_hits, DiamondSearchConfig
from models.identity_fusion_data import IDENTITY_SIDECAR_SCHEMA

def _hit_row(hit, rank):
    if hit.nident is None:
        raise ValueError("Identity sidecars require exact nident; enrich DIAMOND output first")
    if not math.isfinite(hit.bitscore) or hit.bitscore <= 0:
        raise ValueError(f"{hit.qseqid}/{hit.sseqid}: selected bitscore must be finite and positive")
    denom = min(hit.qlen, hit.slen)
    if denom <= 0 or not 0 <= hit.nident <= denom:
        raise ValueError(f"{hit.qseqid}/{hit.sseqid}: invalid nident or sequence lengths")
    return {"target_id": hit.sseqid, "rank": rank, "bitscore": hit.bitscore, "nident": hit.nident, "qlen": hit.qlen, "slen": hit.slen, "identity_fraction": hit.nident / denom}

def build_identity_sidecar(*, query_ids, hits_tsv, output_path, config: DiamondSearchConfig, exclude_self_hits=False):
    """Use EIHR retained hits, then InterLabelGO implementation’s top-five weighted statistic."""
    parsed = _parse_diamond_hits(hits_tsv)
    records, audit = [], []
    for query_id in query_ids:
        retained = _retain_valid_hits(parsed.get(query_id, []), config, exclude_self=exclude_self_hits, query_id=query_id)
        selected = retained[:5]
        if not selected:
            records.append({"protein_id": query_id, "has_retained_hit": False, "identity_fraction": None, "retained_hit_count": 0, "number_used": 0})
            continue
        rows = [_hit_row(hit, rank + 1) for rank, hit in enumerate(selected)]
        total = sum(row["bitscore"] for row in rows)
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"{query_id}: invalid total selected bitscore")
        s = sum(row["bitscore"] * row["identity_fraction"] for row in rows) / total
        if not 0 <= s <= 1 or not math.isfinite(s):
            raise ValueError(f"{query_id}: computed identity outside [0,1]")
        records.append({"protein_id": query_id, "has_retained_hit": True, "identity_fraction": s, "retained_hit_count": len(retained), "number_used": len(rows)})
        audit.extend([{"protein_id": query_id, **row} for row in rows])
    output_path = Path(output_path)
    audit_path = output_path.with_name(output_path.stem + "_per_hit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
    payload = {"schema_version": IDENTITY_SIDECAR_SCHEMA, "formula_version": "top5_bitscore_weighted_nident_over_min_length_v1", "source_hits_sha256": hashlib.sha256(Path(hits_tsv).read_bytes()).hexdigest(), "exclude_self_hits": exclude_self_hits, "tie_policy": "EIHR retained ordering: -bitscore, -qcov, evalue, sseqid", "audit_path": str(audit_path), "records": records}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload
