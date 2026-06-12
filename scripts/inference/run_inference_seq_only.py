#!/usr/bin/env python3
"""Run prediction or CAFA evaluation for the sequence-only model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sequence_only_ablation import (
    SequenceOnlyESMShardDataset,
    build_seq_only_model,
    make_sequence_only_collate_fn,
)
from reliability_aware.utils.inference_utils import InferenceSpec, run_inference


SPEC = InferenceSpec(
    ablation="sequence_only",
    dataset_kind="sequence",
    dataset_cls=SequenceOnlyESMShardDataset,
    collate_factory=make_sequence_only_collate_fn,
    model_builder=build_seq_only_model,
)


def main(argv: Sequence[str] | None = None) -> None:
    run_inference(
        spec=SPEC,
        description="Run prediction or CAFA evaluation for a sequence-only checkpoint.",
        argv=argv,
    )


if __name__ == "__main__":
    main()
