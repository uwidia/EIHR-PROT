#!/usr/bin/env python3
"""Run CAFA evaluation for sequence+homology models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sequence_homology_ablation import (
    build_sequence_homology_confidence_gate_model,
    build_sequence_homology_internal_gate_model,
)
from models.sequence_homology_common import (
    SequenceHomologyShardDataset,
    make_sequence_homology_collate_fn,
)
from reliability_aware.utils.inference_utils import InferenceSpec, run_inference


SPECS = {
    "sequence_homology_internal_gate": InferenceSpec(
        ablation="sequence_homology_internal_gate",
        dataset_kind="sequence_homology",
        dataset_cls=SequenceHomologyShardDataset,
        collate_factory=make_sequence_homology_collate_fn,
        model_builder=build_sequence_homology_internal_gate_model,
    ),
    "sequence_homology_confidence_gate": InferenceSpec(
        ablation="sequence_homology_confidence_gate",
        dataset_kind="sequence_homology",
        dataset_cls=SequenceHomologyShardDataset,
        collate_factory=make_sequence_homology_collate_fn,
        model_builder=build_sequence_homology_confidence_gate_model,
    ),
}

GATE_SPECS = {
    "internal": "sequence_homology_internal_gate",
    "confidence": "sequence_homology_confidence_gate",
}


def main(argv: Sequence[str] | None = None) -> None:
    run_inference(
        spec=SPECS,
        description="Run prediction or CAFA evaluation for a sequence+homology checkpoint.",
        gate_specs=GATE_SPECS,
        default_gate="confidence",
        argv=argv,
    )


if __name__ == "__main__":
    main()
