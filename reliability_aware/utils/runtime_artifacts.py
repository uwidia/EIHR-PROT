from __future__ import annotations

from pathlib import Path
from typing import Iterable

ASPECTS = ("BP", "MF", "CC")
CHECKPOINT_TEMPLATE = (
    "runs/sequence_homology_confidence_gate/final/{aspect}/best_model.pt"
)

SHARED_RUNTIME_ARTIFACTS = (
    "diamond_db/train_db.fasta",
    "diamond_db/train_db.dmnd",
    "data/HEAL_dataset/go-basic.obo",
    "hashlist.txt",
)


def runtime_artifact_relative_paths(
    aspects: Iterable[str] = ASPECTS,
) -> tuple[str, ...]:
    paths = list(SHARED_RUNTIME_ARTIFACTS)
    for aspect in aspects:
        normalized = aspect.upper()
        if normalized not in ASPECTS:
            raise ValueError(f"Unsupported GO aspect: {aspect}")
        paths.extend(
            (
                f"diamond_db/{normalized}/go_vocab.json",
                f"diamond_db/{normalized}/subject_go_index.json",
                CHECKPOINT_TEMPLATE.format(aspect=normalized),
            )
        )
    return tuple(paths)


def missing_runtime_artifacts(
    project_root: Path,
    aspects: Iterable[str] = ASPECTS,
) -> list[str]:
    root = Path(project_root)
    return [
        relative_path
        for relative_path in runtime_artifact_relative_paths(aspects)
        if not (root / relative_path).is_file()
    ]
