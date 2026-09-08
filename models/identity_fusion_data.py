"""Validated identity-sidecar loading for identity-conditioned fusion."""
from __future__ import annotations

import json
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
        has_hit = row.get("has_retained_hit")
        s = row.get("identity_fraction")
        if not isinstance(has_hit, bool):
            raise ValueError(f"{path}: {protein_id} has_retained_hit must be boolean")
        if has_hit:
            if not isinstance(s, (float, int)) or not 0 <= float(s) <= 1:
                raise ValueError(f"{path}: {protein_id} has hits but invalid identity fraction")
        elif s is not None:
            raise ValueError(f"{path}: {protein_id} no-hit identity must be null")
        result[protein_id] = row
    return result


class IdentitySequenceHomologyShardDataset(SequenceHomologyShardDataset):
    def __init__(self, *args, identity_sidecar_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        records = load_identity_sidecar(identity_sidecar_path)
        labels = [self.seq_ds.label_for_global_idx(i) for i in self.seq_ds.global_indices]
        missing = [label for label in labels if label not in records]
        if missing:
            raise ValueError(f"Identity sidecar is missing {len(missing)} dataset IDs; first: {missing[0]}")
        self.identity_records = records

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        row = self.identity_records[item["label"]]
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
