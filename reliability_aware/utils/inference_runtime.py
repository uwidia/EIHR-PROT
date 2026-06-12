"""Model, dataset, prediction, and evaluation mechanics for inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import torch
from torch.utils.data import DataLoader

from reliability_aware.utils.cafa_metrics import evaluate_cafa
from reliability_aware.utils.go_term_extraction import (
    build_go_annotations_list,
    build_subject_go_index,
)
from reliability_aware.utils.model_training import (
    model_forward_from_batch,
    move_batch_to_device,
)
from reliability_aware.utils.parser import get_protein_info

DatasetKind = Literal["sequence", "sequence_homology", "homology"]


@dataclass(frozen=True)
class InferenceSpec:
    ablation: str
    dataset_kind: DatasetKind
    dataset_cls: type
    collate_factory: Callable[..., Callable]
    model_builder: Callable | None = None


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_torch_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load checkpoints across PyTorch versions, including weights_only versions."""
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    return checkpoint


def load_go_vocab(go_vocab_path: Path) -> list[str]:
    require_path(go_vocab_path, "GO vocabulary JSON")
    go_terms = json.loads(go_vocab_path.read_text())
    if not isinstance(go_terms, list) or not all(isinstance(x, str) for x in go_terms):
        raise ValueError(f"Invalid GO vocabulary file: {go_vocab_path}")
    if not go_terms:
        raise ValueError(f"Empty GO vocabulary: {go_vocab_path}")
    return go_terms


def fasta_ids(fasta_path: Path) -> list[str]:
    return [protein["full_id"] for protein in get_protein_info(fasta_path)]


def build_label_indices_for_split(
    *,
    fasta_path: Path,
    go_annotation_path: Path,
    obo_path: Path,
    go_aspect: str,
    go_term_to_idx: dict[str, int],
) -> tuple[dict[str, list[int]], set[str]]:
    """Build label-to-GO-index targets using the existing annotation utilities."""
    ids = fasta_ids(fasta_path)
    label_to_go_terms, _ = build_go_annotations_list(
        go_annotation_path=go_annotation_path,
        obo_path=obo_path,
        go_aspect=go_aspect,
        keep_ids=ids,
        remove_root_term=True,
        min_term_freq=None,
    )
    label_to_indices = build_subject_go_index(label_to_go_terms, go_term_to_idx)
    keep_ids = {label for label, idxs in label_to_indices.items() if idxs}
    return label_to_indices, keep_ids


def load_model_from_checkpoint(
    *,
    model_builder: Callable,
    checkpoint_path: Path,
    go_terms: list[str],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'model_state_dict'. Make sure you passed "
            "run_XXX/best_model.pt, not best_final_run.pt or final_meta.pt."
        )
    if "hparams" not in checkpoint or checkpoint["hparams"] is None:
        raise KeyError(
            "Checkpoint is missing 'hparams'. The model builder needs these "
            "to rebuild the architecture."
        )

    model, _optimizer = model_builder(checkpoint["hparams"], go_terms, device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model mismatch. Check that the inference script, "
            "--go_aspect, and --go_vocab_path match the exact training run."
        ) from exc

    model.eval()
    return model, checkpoint


def build_test_loader(
    *,
    spec: InferenceSpec,
    test_esm_shard_dir: Path,
    test_manifest_path: Path,
    test_homology_shard_dir: Path | None,
    test_label_to_indices: dict[str, list[int]] | None,
    test_keep_ids: set[str],
    go_terms: list[str],
    batch_size: int,
    num_workers: int,
    allow_unlabeled_predictions: bool,
    device: torch.device,
) -> tuple[Any, DataLoader]:
    """Build a test DataLoader using the model-specific dataset and collate factory."""
    dataset_kind = spec.dataset_kind
    keep_ids = (
        None
        if test_label_to_indices is None
        or (allow_unlabeled_predictions and dataset_kind == "sequence")
        else test_keep_ids
    )

    if dataset_kind == "sequence":
        dataset = spec.dataset_cls(
            shard_dir=test_esm_shard_dir,
            manifest_path=test_manifest_path,
            keep_ids=keep_ids,
        )
        collate_fn = spec.collate_factory(
            global_idx_to_label=dataset.global_idx_to_label,
            label_to_indices=test_label_to_indices,
            num_go_terms=len(go_terms),
        )
    elif dataset_kind == "sequence_homology":
        if test_homology_shard_dir is None:
            raise ValueError(
                "sequence+homology models require --test_homology_shard_dir"
            )
        dataset = spec.dataset_cls(
            esm_shard_dir=test_esm_shard_dir,
            homology_shard_dir=test_homology_shard_dir,
            manifest_path=test_manifest_path,
            keep_ids=keep_ids,
        )
        collate_fn = spec.collate_factory(
            label_to_indices=test_label_to_indices,
            num_go_terms=len(go_terms),
        )
    elif dataset_kind == "homology":
        if test_homology_shard_dir is None:
            raise ValueError("homology_only requires --test_homology_shard_dir")
        dataset = spec.dataset_cls(
            homology_shard_dir=test_homology_shard_dir,
            manifest_path=test_manifest_path,
            keep_ids=keep_ids,
        )
        collate_fn = spec.collate_factory(
            label_to_indices=test_label_to_indices,
            num_go_terms=len(go_terms),
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {dataset_kind}")

    if len(dataset) == 0:
        raise RuntimeError("The test dataset is empty after applying the filters.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return dataset, loader


@torch.no_grad()
def run_prediction_pass(
    *,
    model: torch.nn.Module | None,
    loader: Iterable[dict[str, Any]],
    go_terms: list[str],
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run one forward pass and collect probabilities, targets, and top-k rows."""
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_gate_weights: list[torch.Tensor] = []
    labels_all: list[str] = []
    global_indices_all: list[int] = []
    topk_rows: list[dict[str, Any]] = []
    k = min(top_k, len(go_terms))

    for batch in loader:
        labels = list(batch["labels"])
        global_indices = [int(x) for x in batch["global_indices"].tolist()]

        if model is None:
            probs = batch["probs"].detach().cpu().float()
            targets = batch.get("targets")
            if targets is not None:
                targets = targets.detach().cpu().float()
            neural_probs = None
            homology_scores = probs
            gate_weights = None
        else:
            batch = move_batch_to_device(batch, device)
            outputs = model_forward_from_batch(model, batch)
            probs = outputs["probs"].detach().cpu().float()
            targets = batch.get("targets")
            if targets is not None:
                targets = targets.detach().cpu().float()

            neural_probs = outputs.get("neural_probs")
            if neural_probs is not None:
                neural_probs = neural_probs.detach().cpu().float()
            homology_scores = outputs.get("homology_scores")
            if homology_scores is not None:
                homology_scores = homology_scores.detach().cpu().float()
            gate_weights = outputs.get("gate_weights")

        all_probs.append(probs)
        if targets is not None:
            all_targets.append(targets)
        labels_all.extend(labels)
        global_indices_all.extend(global_indices)
        if gate_weights is not None:
            gate_weights = gate_weights.detach().cpu().float()
            all_gate_weights.append(gate_weights)

        top_values, top_indices = probs.topk(k=k, dim=1)
        for row_idx, protein_id in enumerate(labels):
            neural_gate: float | str = ""
            homology_gate: float | str = ""
            if gate_weights is not None:
                neural_gate = float(gate_weights[row_idx, 0].item())
                homology_gate = float(gate_weights[row_idx, 1].item())

            for rank, (go_idx, score) in enumerate(
                zip(top_indices[row_idx].tolist(), top_values[row_idx].tolist()),
                start=1,
            ):
                row = {
                    "protein_id": protein_id,
                    "global_idx": global_indices[row_idx],
                    "rank": rank,
                    "go_term": go_terms[go_idx],
                    "probability": float(score),
                    "neural_probability": "",
                    "homology_probability": "",
                    "neural_gate": neural_gate,
                    "homology_gate": homology_gate,
                }
                if neural_probs is not None:
                    row["neural_probability"] = float(
                        neural_probs[row_idx, go_idx].item()
                    )
                if homology_scores is not None:
                    row["homology_probability"] = float(
                        homology_scores[row_idx, go_idx].item()
                    )
                topk_rows.append(row)

    result: dict[str, Any] = {
        "y_prob": torch.cat(all_probs, dim=0),
        "labels": labels_all,
        "global_indices": global_indices_all,
        "topk_rows": topk_rows,
    }
    if all_targets:
        result["y_true"] = torch.cat(all_targets, dim=0)
    if all_gate_weights:
        result["gate_weights"] = torch.cat(all_gate_weights, dim=0)
    return result


def compute_metrics(
    *,
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    go_terms: list[str],
    go_aspect: str,
    obo_path: Path,
    train_annotations: list[set[str]],
) -> dict[str, float | int | str]:
    """Compute the repository's supported CAFA metric suite."""
    metrics, _ = evaluate_cafa(
        y_true=y_true,
        y_prob=y_prob,
        go_terms=go_terms,
        go_aspect=go_aspect,
        obo_path=obo_path,
        train_annotations=train_annotations,
    )
    return metrics
