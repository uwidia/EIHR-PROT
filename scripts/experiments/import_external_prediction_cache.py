#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reliability_aware.utils.prediction_import import import_long_csv
p = argparse.ArgumentParser(description="Import explicit external full prediction matrix.")
p.add_argument("--csv", type=Path, required=True); p.add_argument("--reference-cache", type=Path, required=True)
p.add_argument("--output-cache", type=Path, required=True); p.add_argument("--model-id", required=True); p.add_argument("--overwrite", action="store_true")
a = p.parse_args()
print(import_long_csv(csv_path=a.csv, reference_cache_path=a.reference_cache, output_path=a.output_cache, model_id=a.model_id, overwrite=a.overwrite))
