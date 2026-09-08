"""Validated identity-sidecar loading for identity-conditioned fusion."""
from __future__ import annotations

import json
import math
from pathlib import Path
import torch

from models.sequence_homology_common import SequenceHomologyShardDataset, make_sequence_homology_collate_fn

IDENTITY_SIDECAR_SCHEMA = "eihr-retained-hit-top5-identity-v1"

def load_identity_sidecar(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != IDENTITY_SIDECAR_SCHEMA:
        raise ValueError(f"Unsupported identity sidecar schema in {path}")
    rows = raw.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: records must be a list")
    result = {}
    for row in rows:
        protein_id = row.get("protein_id")
        if not protein_id or protein_id in result:
            raise ValueError(f"{path}: missing or duplicate protein_id {protein_id!r}")
        count, used = row.get('retained_hit_count'), row.get('number_used')
        if type(count) is not int or type(used) is not int or not 0 <= count <= 10 or used != min(count, 5):
            raise ValueError(f'{path}: {protein_id} invalid retained-hit count/number_used')
        has_hit = row.get("has_retained_hit")
        s = row.get("identity_fraction")
        if not isinstance(has_hit, bool):
            raise ValueError(f"{path}: {protein_id} has_retained_hit must be boolean")
        if has_hit != (count > 0):
            raise ValueError(f'{path}: {protein_id} hit flag/count mismatch')
        if has_hit:
            if type(s) not in (float, int) or not 0 <= float(s) <= 1:
                raise ValueError(f"{path}: {protein_id} has hits but invalid identity fraction")
        elif s is not None:
            raise ValueError(f"{path}: {protein_id} no-hit identity must be null")
        result[protein_id] = row
    if raw.get('evidence_report'):
        from reliability_aware.utils.prediction_cache import sha256_file, canonical_ids_hash
        report_path = Path(raw['evidence_report'])
        if sha256_file(report_path) != raw.get('evidence_report_sha256'):
            raise ValueError(f'{path}: evidence report checksum mismatch')
        report = json.loads(report_path.read_text())
        if report.get('status') != 'matched':
            raise ValueError(f'{path}: retained evidence was not matched')
        for source in report['source_hashes'].values():
            if sha256_file(Path(source['path'])) != source['sha256']:
                raise ValueError(f'{path}: stale source hits/queries')
        if canonical_ids_hash(sorted(result)) != raw.get('query_ids_sha256'):
            raise ValueError(f'{path}: query cohort checksum mismatch')
        if sha256_file(Path(raw['audit_path'])) != raw.get('audit_sha256'):
            raise ValueError(f'{path}: per-hit audit checksum mismatch')
        if set(result) & set(raw['excluded_query_ids']):
            raise ValueError(f'{path}: excluded validation IDs remain in sidecar')
        if raw['excluded_query_ids'] != report['excluded_query_ids'] or raw['retention'] != report['retention']:
            raise ValueError(f'{path}: sidecar/evidence policy mismatch')
        if raw['exclude_self_hits'] != raw['retention']['exclude_self_hits']:
            raise ValueError(f'{path}: inconsistent self-hit policy')
        audit = json.loads(Path(raw['audit_path']).read_text())
        by_query = {}
        for hit in audit:
            q = hit['protein_id']
            if q not in result:
                raise ValueError(f'{path}: audit contains an unknown query {q}')
            n, qlen, slen, bits = hit['nident'], hit['qlen'], hit['slen'], hit['bitscore']
            if any(type(x) is not int for x in (n, qlen, slen)) or min(qlen, slen) <= 0 or not 0 <= n <= min(qlen, slen):
                raise ValueError(f'{path}: invalid exact count/lengths in audit')
            if not math.isfinite(bits) or bits <= 0 or not math.isclose(hit['identity_fraction'], n/min(qlen, slen), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'{path}: invalid bitscore/identity in audit')
            by_query.setdefault(q, []).append(hit)
        for q, row in result.items():
            hits = by_query.get(q, [])
            if len(hits) != row['number_used'] or sorted(h['rank'] for h in hits) != list(range(1, len(hits)+1)) or len({h['target_id'] for h in hits}) != len(hits):
                raise ValueError(f'{path}: {q} audit count/rank/target mismatch')
            if hits:
                total = sum(h['bitscore'] for h in hits)
                expected = sum(h['bitscore']*h['identity_fraction'] for h in hits) / total
                if not math.isfinite(total) or not math.isclose(row['identity_fraction'], expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f'{path}: {q} identity differs from its exact per-hit audit')

    return result


class IdentitySequenceHomologyShardDataset(SequenceHomologyShardDataset):
    def __init__(self, *args, identity_sidecar_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        records = load_identity_sidecar(identity_sidecar_path)
        if not json.loads(Path(identity_sidecar_path).read_text()).get('evidence_report'):
            raise ValueError('Identity dataset requires verified retained evidence; run fusion_baselines.py prepare')
        labels = [self.seq_ds.label_for_global_idx(i) for _, _, i in self.seq_ds.index]
        missing = [label for label in labels if label not in records]
        if missing:
            raise ValueError(f"Identity sidecar is missing {len(missing)} dataset IDs; first: {missing[0]}")
        self.identity_records = records

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        row = self.identity_records[item["label"]]
        shard_has_hit = bool(item['gate_features'][3] > 0)
        if shard_has_hit != row['has_retained_hit'] or not math.isclose(float(item['gate_features'][2]), math.log1p(row['retained_hit_count']), rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"{item['label']}: sidecar/shard retained-hit availability mismatch")
        item["identity_fraction"] = 0.0 if row["identity_fraction"] is None else float(row["identity_fraction"])
        item["has_retained_hit"] = bool(row["has_retained_hit"])
        return item


def make_identity_sequence_homology_collate_fn(label_to_indices, num_go_terms):
    base = make_sequence_homology_collate_fn(label_to_indices, num_go_terms)
    def collate(batch):
        out = base(batch)
        out["identity_fraction"] = torch.tensor([x["identity_fraction"] for x in batch], dtype=torch.float32)
        out["has_retained_hit"] = torch.tensor([x["has_retained_hit"] for x in batch], dtype=torch.bool)
        return out
    return collate
