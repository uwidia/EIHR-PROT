#!/usr/bin/env python3
"""Run prediction or CAFA evaluation for the homology-only baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.homology_only_baseline import (
    HomologyShardDataset,
    make_homology_only_collate_fn,
)
from reliability_aware.utils.inference_utils import InferenceSpec, run_inference


SPEC = InferenceSpec(
    ablation="homology_only",
    dataset_kind="homology",
    dataset_cls=HomologyShardDataset,
    collate_factory=make_homology_only_collate_fn,
)


def main(argv: Sequence[str] | None = None) -> None:
    run_inference(
        spec=SPEC,
        description="Run prediction or CAFA evaluation for the homology-only baseline.",
        argv=argv,
    )


if __name__ == "__main__":
    main()
