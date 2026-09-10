"""Prepare or train the full confidence gate and four leave-one-signal-out models."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reliability_aware.utils.initialization import resolve_initialization_seed
from reliability_aware.utils.reproducibility import resolve_cpu_threads, training_environment

VARIANTS = {
    "full": None,
    "without_max_bitscore": "b_max",
    "without_max_query_coverage": "cov_max",
    "without_log_retained_hit_count": "log1p_n_hits",
    "without_hit_availability": "has_hit",
}


def prepare_suite(source_path, output_dir, seed, aspects=("BP", "MF", "CC"), *, cpu_threads=None):
    """Freeze selected parameters into reviewable configs without touching old runs."""
    source_path = Path(source_path).resolve()
    source_bytes = source_path.read_bytes()
    source = yaml.safe_load(source_bytes)
    seed = resolve_initialization_seed(seed)
    if seed is None:
        raise ValueError("A fixed integer or 'random' initialization seed is required")
    cpu_threads = resolve_cpu_threads(cpu_threads)
    aspects = list(aspects)
    if not aspects or len(set(aspects)) != len(aspects) or any(a not in ("BP", "MF", "CC") for a in aspects):
        raise ValueError("Choose distinct aspects from BP, MF, CC")
    # Validate before creating the output directory.
    for aspect in aspects:
        if not isinstance(source.get("promising_hparams", {}).get(aspect), dict):
            raise ValueError(f"Missing selected promising_hparams for {aspect}")
    for key in ("batch_size", "final_epochs", "patience"):
        if key not in source:
            raise ValueError(f"Missing training setting: {key}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config_dir = output_dir / "configs"
    config_dir.mkdir()
    jobs = []
    for variant, omitted in VARIANTS.items():
        config = deepcopy(source)
        config["use_search_results"] = False
        config["base_dir_final"] = str(output_dir / variant / "final")
        config["base_dir_search"] = str(output_dir / variant / "unused_search")
        config["promising_hparams"] = {
            aspect: {
                **source["promising_hparams"][aspect],
                "initialization_seed": seed,
                "training_seed": seed,
                "reproducible_training": True,
                "cpu_threads": cpu_threads,
                "omitted_gate_feature": omitted,
            }
            for aspect in aspects
        }
        config_path = config_dir / f"{variant}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        for aspect in aspects:
            command = [
                sys.executable, str(PROJECT_ROOT / "scripts/run_model_training.py"),
                "--ablation", "sequence_homology_confidence_gate",
                "--go_aspect", aspect, "--hparams", str(config_path),
                "--run_type", "full_training",
            ]
            jobs.append({"variant": variant, "aspect": aspect, "command": command,
                         "hparams": config["promising_hparams"][aspect],
                         "checkpoint": str(Path(config["base_dir_final"]) / aspect / "best_model.pt")})
    manifest = {
        "source_config": str(source_path),
        "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "initialization_seed": seed,
        "training_seed": seed,
        "reproducible_training": True,
        "cpu_threads": cpu_threads,
        "environment": training_environment(seed, cpu_threads),
        "seed_scope": "Initialization, Python/NumPy/PyTorch CPU and CUDA RNGs, epoch batch ordering, and loader workers",
        "homology_policy": "Existing training/validation homology shards and priors unchanged",
        "working_directory": str(PROJECT_ROOT),
        "jobs": jobs,
    }
    (output_dir / "suite.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hparams", type=Path, default=PROJECT_ROOT / "configs/sequence_homology_confidence_gate.yaml")
    parser.add_argument("--seed", default="42", help="Initialization and training seed: integer or random (one draw shared by all jobs).")
    parser.add_argument("--cpu-threads", type=int, help="CPU thread count shared by all variants (default: up to 8 available CPUs).")
    parser.add_argument("--aspects", nargs="+", choices=("BP", "MF", "CC"), default=["BP", "MF", "CC"])
    parser.add_argument("--output-dir", type=Path, required=True, help="New experiment directory; must not already exist.")
    parser.add_argument("--train", action="store_true", help="Train all prepared jobs sequentially. Otherwise only write configs and print commands.")
    args = parser.parse_args()
    try:
        seed = args.seed if args.seed == "random" else int(args.seed)
        manifest = prepare_suite(args.hparams, args.output_dir, seed, args.aspects, cpu_threads=args.cpu_threads)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Prepared {len(manifest['jobs'])} jobs with initialization_seed={manifest['initialization_seed']} and cpu_threads={manifest['cpu_threads']}", flush=True)
    for job in manifest["jobs"]:
        print(shlex.join(job["command"]), flush=True)
        if args.train:
            subprocess.run(job["command"], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
