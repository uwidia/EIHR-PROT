#!/usr/bin/env python3
"""Run retained-hit bootstrap analysis entirely from six full-output caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reliability_aware.utils.bootstrap_reporting import analyze_pair, load_assignments, write_workbook


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML mapping aspects to full and ablation cache paths")
    parser.add_argument("--assignments", type=Path, default=PROJECT_ROOT / "data/cleaned_dataset/retained_hit_identity_bins/assignments.csv")
    parser.add_argument("--obo-path", type=Path, default=PROJECT_ROOT / "data/HEAL_dataset/go-basic.obo")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/retained_hit_identity_bootstrap")
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.n_resamples <= 0 or args.batch_size <= 0:
        parser.error("--n-resamples and --batch-size must be positive")
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or "aspects" not in config:
        parser.error("config must contain an aspects mapping")
    assignments = load_assignments(args.assignments)
    all_records = {"counts": [], "gate": [], "gate_paired": [], "performance": [], "replicates": {}}
    for aspect in ("BP", "MF", "CC"):
        item = config["aspects"].get(aspect)
        if not isinstance(item, dict) or not {"full_cache", "ablation_cache"} <= set(item):
            parser.error(f"config missing full_cache/ablation_cache for {aspect}")
        result = analyze_pair(full_cache_path=(PROJECT_ROOT / item["full_cache"]).resolve() if not Path(item["full_cache"]).is_absolute() else Path(item["full_cache"]),
            ablation_cache_path=(PROJECT_ROOT / item["ablation_cache"]).resolve() if not Path(item["ablation_cache"]).is_absolute() else Path(item["ablation_cache"]),
            aspect=aspect, assignments=assignments, obo_path=args.obo_path, dataset=config.get("dataset", "PDB"), plans_dir=args.output_dir / "plans",
            n_resamples=args.n_resamples, seed=args.seed, batch_size=args.batch_size)
        for key in all_records:
            all_records[key].update(result[key]) if isinstance(all_records[key], dict) else all_records[key].extend(result[key])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "bootstrap_replicates.npz", **all_records.pop("replicates"))
    (args.output_dir / "bootstrap_summary.json").write_text(json.dumps(all_records, indent=2, sort_keys=True))
    workbook = write_workbook(all_records, args.output_dir / "retained_hit_identity_bootstrap_results.xlsx", {
        "dataset": config.get("dataset", "PDB"), "models": "full=sequence_homology_confidence_gate; ablation=sequence_homology_internal_gate",
        "cohort_policy": "assignment membership intersected with frozen annotation eligibility; no silent output omissions", "thresholds": "0.01..1.00 inclusive, score >= threshold",
        "IC source": "fixed propagated training annotations stored in caches", "B": args.n_resamples, "seed": args.seed,
        "percentile_method": "numpy.percentile(method='linear')", "direction": "all deltas are full minus ablation; Fmax/AUPR higher and Smin lower are favorable",
        "interpretation": "Intervals describe fixed-model test-protein sampling, not training variability; exploratory and unadjusted."})
    print(f"Saved workbook: {workbook}")


if __name__ == "__main__":
    main()
