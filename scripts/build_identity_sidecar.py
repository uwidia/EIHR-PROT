#!/usr/bin/env python3
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from pathlib import Path
from reliability_aware.utils.diamond_homology import DiamondSearchConfig
from reliability_aware.utils.diamond_homology import read_fasta_as_dict
from reliability_aware.utils.identity_fusion import build_identity_sidecar

p = argparse.ArgumentParser(description="Build validated retained-hit top-five identity sidecar")
p.add_argument("--queries", type=Path, required=True)
p.add_argument("--hits", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--exclude-self", action="store_true")
args = p.parse_args()
cfg = DiamondSearchConfig(evalue_max=1e-5, min_query_coverage=.30, top_k=10)
args.output.parent.mkdir(parents=True, exist_ok=True)
build_identity_sidecar(query_ids=list(read_fasta_as_dict(args.queries)), hits_tsv=args.hits, output_path=args.output, config=cfg, exclude_self_hits=args.exclude_self)
