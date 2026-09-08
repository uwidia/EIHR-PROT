"""Protocol and provenance used exclusively by the two new fusion baselines."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from reliability_aware.utils.prediction_cache import canonical_ids_hash, sha256_file

BASELINES = {"sequence_homology_fixed_fusion", "sequence_homology_identity_fusion"}
VALIDATION_EXCLUSIONS = {"2VAU-A", "5LSQ-A"}
IDENTITY_POLICY = {
    "formula_version": "top5_bitscore_weighted_nident_over_min_length_v1",
    "no_hit_policy": "missing identity; s=0; neural weight=1",
    "retention": {"evalue_max": 1e-5, "min_query_coverage": .30, "top_k": 10},
    "tie_policy": "-bitscore, -qcov, evalue, sseqid",
    "initialization": "fresh attention pooling and head; fixed ESM and homology inputs",
}


def load_resources(path):
    """Paths are repository-relative, as are existing configuration paths."""
    raw = yaml.safe_load(Path(path).read_text())
    if set(raw.get("validation_exclude_ids", [])) != VALIDATION_EXCLUSIONS:
        raise ValueError("This fusion protocol requires exactly 2VAU-A and 5LSQ-A validation exclusions")
    return raw


def validation_keep_ids(ablation, eligible_ids, excluded_ids):
    if ablation not in BASELINES:
        return eligible_ids
    if set(excluded_ids) != VALIDATION_EXCLUSIONS:
        raise ValueError("Invalid fusion validation exclusion policy")
    kept = set(eligible_ids) - set(excluded_ids)
    if not kept:
        raise ValueError("No eligible validation proteins remain")
    return kept


def seed_run(seed, train_loader=None, val_loader=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for loader in (train_loader, val_loader):
        sampler = getattr(loader, "batch_sampler", None)
        if sampler is not None:
            sampler.seed = seed
            sampler.set_epoch(0)


def resource_fingerprint(path):
    path = Path(path)
    if path.is_file():
        return {"path": str(path), "sha256": sha256_file(path)}
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file())
        if not files:
            raise ValueError(f"Empty resource directory: {path}")
        return {"path": str(path), "files": {p.name: sha256_file(p) for p in files}}
    raise FileNotFoundError(f"Missing fusion resource: {path}")


def protocol_metadata(*, ablation, aspect, go_terms, resources, hparams, train_ids, val_ids):
    sources = {"obo": resources["obo"], "annotations": resources["train_annotations"]}
    for split in ("pdb_train", "pdb_val"):
        item = resources[split]
        for field in ("fasta", "manifest", "esm_shards", "homology_shards", "original_hits"):
            sources[f"{split}/{field}"] = item[field].format(aspect=aspect)
        if ablation.endswith("identity_fusion"):
            sources[f"{split}/identity"] = item["identity"]
    return {
        "fusion_protocol_version": "validation_exclude_two_v1",
        "model_type": ablation, "go_aspect": aspect,
        "go_terms": list(go_terms), "go_terms_sha256": canonical_ids_hash(go_terms),
        "validation_exclude_ids": sorted(VALIDATION_EXCLUSIONS),
        "train_ids_sha256": canonical_ids_hash(sorted(train_ids)),
        "validation_ids_sha256": canonical_ids_hash(sorted(val_ids)),
        "train_n": len(train_ids), "validation_n": len(val_ids),
        "adaptation": (IDENTITY_POLICY if ablation.endswith('identity_fusion') else {
            "formula": "sigmoid(z) global homology weight; z initialized to zero",
            "no_hit_policy": "same global convex mixture with zero homology prior",
            "scalar_weight_decay": 0.0,
            "initialization": IDENTITY_POLICY['initialization'],
        }),
        "configuration": hparams,
        "resources": {k: resource_fingerprint(v) for k, v in sources.items()},
        "seed": int(hparams.get("seed", 42)),
        "reference_source_hashes": {name: resource_fingerprint(Path(resources['reference_source_dir']) / name)
                                    for name in ('predict.py', 'alignment_knn.py')},
        "seed_schedule": "search: seed + trial; final: seed + 10000 + candidate rank",
    }


def bind_run_directory(directory, metadata):
    """Never silently mix old cohorts, resources, or settings with a new run."""
    directory = Path(directory)
    target = directory / "protocol.json"
    normalized = json.loads(json.dumps(metadata, sort_keys=True))
    if target.exists():
        if json.loads(target.read_text()) != normalized:
            raise ValueError(f"Protocol mismatch in {directory}; select a new output directory")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Unversioned results already exist in {directory}; select a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, indent=2, sort_keys=True))


def selected_candidates(path, count, metadata):
    path = Path(path)
    saved = json.loads((path.parent / "protocol.json").read_text())
    if saved != json.loads(json.dumps(metadata)):
        raise ValueError("Search results use a different cohort, resources, model, or configuration")
    records = json.loads(path.read_text())
    if count <= 0 or not records:
        raise ValueError("Need positive top_k_params and nonempty search results")
    if any(not math.isfinite(float(r["score"])) for r in records):
        raise ValueError("Nonfinite search score")
    return [r["hparams"] for r in sorted(records, key=lambda r: r["score"], reverse=True)[:count]]


def fusion_parameters(model):
    if hasattr(model, "fusion_logit"):
        h = float(torch.sigmoid(model.fusion_logit.detach()).cpu())
        return {"alpha_neural": 1 - h, "alpha_homology": h}
    return {"identity_a": float(model.identity_a), "identity_k": float(model.identity_k)}


def validate_checkpoint(checkpoint, model, go_terms):
    if not checkpoint.get("fusion_protocol_version"):
        raise ValueError("Fusion checkpoint lacks cohort/protocol provenance; run the revised baseline training")
    if checkpoint.get("go_terms") != list(go_terms) or checkpoint.get("go_terms_sha256") != canonical_ids_hash(go_terms):
        raise ValueError("Fusion checkpoint GO vocabulary/order mismatch")
    parameters = fusion_parameters(model)
    for key, value in parameters.items():
        if not math.isclose(value, float(checkpoint["fusion_parameters"][key]), rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Tampered fusion parameter summary: {key}")
        if key.startswith("identity_") and not math.isclose(value, float(checkpoint["hparams"][key]), rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Identity hparams/state mismatch: {key}")
