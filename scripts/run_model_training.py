"""
Examples:
  python scripts/run_model_training.py --ablation sequence_only --go_aspect BP --hparams configs/new_main_models/sequence_only_search.yaml --run_type randomized_search
  python scripts/run_model_training.py --ablation sequence_homology_confidence_gate --go_aspect BP --hparams configs/new_main_models/sequence_homology_confidence_gate_search.yaml --run_type randomized_search
  python scripts/run_model_training.py --ablation homology_only --go_aspect BP --hparams configs/new_main_models/homology_only_eval.yaml --run_type evaluate_only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml



ACTIVE_ABLATIONS = [
    "sequence_only",
    "homology_only",
    "sequence_homology_fixed",
    "sequence_homology_internal_gate",
    "sequence_homology_confidence_gate",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--go_aspect", type=str, required=True)
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument(
        "--ablation",
        type=str.lower,
        required=True,
        choices=ACTIVE_ABLATIONS,
    )
    parser.add_argument(
        "--run_type",
        type=str.lower,
        required=True,
        choices=["randomized_search", "full_training", "evaluate_only"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.hparams) as f:
        hparams = yaml.safe_load(f) or {}

    ablation = args.ablation.lower()
    run_type = args.run_type.lower()

    if ablation == "homology_only" and run_type != "evaluate_only":
        raise ValueError("homology_only is evaluation-only; use --run_type evaluate_only.")
    if ablation != "homology_only" and run_type == "evaluate_only":
        raise ValueError(
            "evaluate_only is currently supported only for homology_only. "
            "Use randomized_search or full_training for trainable models."
        )

    import torch

    from reliability_aware.utils import config
    from reliability_aware.utils.generic_loader import build_model_loaders
    from reliability_aware.utils.model_training import (
        build_go_annotation_data,
        fit_model,
        run_model_training,
    )
    import reliability_aware.utils.model_randomized_search as randomized_search

    from models.homology_only_baseline import run_homology_only_evaluation
    from models.sequence_homology_ablation import (
        build_sequence_homology_confidence_gate_model,
        build_sequence_homology_fixed_model,
        build_sequence_homology_internal_gate_model,
        run_one_batch_smoke_test_sequence_homology_confidence_gate,
        run_one_batch_smoke_test_sequence_homology_fixed,
        run_one_batch_smoke_test_sequence_homology_internal_gate,
    )
    from models.sequence_homology_common import (
        SequenceHomologyShardDataset,
        make_sequence_homology_collate_fn,
    )
    from models.sequence_only_ablation import (
        SequenceOnlyESMShardDataset,
        build_seq_only_model,
        make_sequence_only_collate_fn,
        run_one_batch_smoke_test_sequence_only,
    )

    config.setup_logging()
    logger = logging.getLogger(__name__)

    go_aspect = args.go_aspect.upper()
    batch_size = hparams.get("batch_size", 16)

    train_homology_shard_dir = (
        config.PROJECT_ROOT / "diamond_db" / go_aspect / "train_homology_shards"
    )
    val_homology_shard_dir = (
        config.PROJECT_ROOT / "diamond_db" / go_aspect / "val_homology_shards"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    go_data = build_go_annotation_data(
        train_dataset=config.train_dataset,
        val_dataset=config.val_dataset,
        go_annotation_path=config.go_annotation_path,
        obo_path=config.obo_path,
        go_aspect=go_aspect,
        device=device,
    )

    run_parameters = {
        "sequence_only": {
            "build_model_fn": build_seq_only_model,
            "dataset_cls": SequenceOnlyESMShardDataset,
            "dataset_kind": "sequence",
            "collate_factory": make_sequence_only_collate_fn,
            "filter_invalid_samples": False,
            "smoke_test_fn": run_one_batch_smoke_test_sequence_only,
        },
        "sequence_homology_fixed": {
            "build_model_fn": build_sequence_homology_fixed_model,
            "dataset_cls": SequenceHomologyShardDataset,
            "dataset_kind": "sequence_homology",
            "collate_factory": make_sequence_homology_collate_fn,
            "filter_invalid_samples": False,
            "smoke_test_fn": run_one_batch_smoke_test_sequence_homology_fixed,
        },
        "sequence_homology_internal_gate": {
            "build_model_fn": build_sequence_homology_internal_gate_model,
            "dataset_cls": SequenceHomologyShardDataset,
            "dataset_kind": "sequence_homology",
            "collate_factory": make_sequence_homology_collate_fn,
            "filter_invalid_samples": False,
            "smoke_test_fn": run_one_batch_smoke_test_sequence_homology_internal_gate,
        },
        "sequence_homology_confidence_gate": {
            "build_model_fn": build_sequence_homology_confidence_gate_model,
            "dataset_cls": SequenceHomologyShardDataset,
            "dataset_kind": "sequence_homology",
            "collate_factory": make_sequence_homology_collate_fn,
            "filter_invalid_samples": False,
            "smoke_test_fn": run_one_batch_smoke_test_sequence_homology_confidence_gate,
        },
    }

    if ablation == "homology_only":
        metrics = run_homology_only_evaluation(
            val_homology_shard_dir=val_homology_shard_dir,
            val_manifest_path=config.val_manifest_path,
            val_label_to_indices=go_data.val_label_to_indices,
            val_keep_ids_for_aspect=go_data.val_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            go_terms=go_data.go_terms,
            child_parent_pairs=go_data.child_parent_pairs,
            ic=go_data.ic,
            device=device,
            lambda_hier=float(hparams.get("lambda_hier", 0.0)),
            pos_weight_cap=float(hparams.get("pos_weight_cap", 20.0)),
            batch_size=batch_size,
            out_dir=hparams.get("base_dir_eval", "runs/homology_only"),
            use_wandb=hparams.get("use_wandb", False),
            wandb_project=hparams.get("wandb_project", "seq_homology_reliability_aware_pfp"),
            wandb_entity=hparams.get("wandb_entity"),
            wandb_mode=hparams.get("wandb_mode", "online"),
            wandb_run_name=hparams.get(
                "wandb_run_name",
                f"homology_only_{go_aspect.lower()}_evaluate_only",
            ),
        )
        logger.info("Homology-only evaluation metrics: %s", metrics)
        return

    loader_kwargs = dict(
        train_esm_shard_dir=config.train_esm_shard_dir,
        val_esm_shard_dir=config.val_esm_shard_dir,
        train_homology_shard_dir=train_homology_shard_dir,
        val_homology_shard_dir=val_homology_shard_dir,
        train_manifest_path=config.train_manifest_path,
        val_manifest_path=config.val_manifest_path,
        train_keep_ids_for_aspect=go_data.train_keep_ids,
        val_keep_ids_for_aspect=go_data.val_keep_ids,
        train_label_to_indices=go_data.train_label_to_indices,
        val_label_to_indices=go_data.val_label_to_indices,
        go_terms=go_data.go_terms,
        batch_size=batch_size,
    )

    model_specific_params = run_parameters[ablation]

    _, _, train_loader, val_loader = build_model_loaders(
        dataset_cls=model_specific_params["dataset_cls"],
        dataset_kind=model_specific_params["dataset_kind"],
        collate_factory=model_specific_params["collate_factory"],
        filter_invalid_samples=model_specific_params["filter_invalid_samples"],
        **loader_kwargs,
    )

    if run_type == "randomized_search":
        randomized_search.run_randomized_search(
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            go_terms=go_data.go_terms,
            child_parent_pairs=go_data.child_parent_pairs,
            ic=go_data.ic,
            search_space=hparams["search_space"],
            device=device,
            num_trials=hparams["num_trials"],
            trial_epochs=hparams["trial_epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            fit_function=fit_model,
            build_model_fn=model_specific_params["build_model_fn"],
            smoke_test_fn=model_specific_params["smoke_test_fn"],
            patience=hparams["patience"],
            base_dir=hparams["base_dir_search"],
            smoke_test=True,
            top_k_params=hparams["top_k_params"],
            use_wandb=hparams.get("use_wandb", False),
            wandb_project=hparams.get("wandb_project", "seq_homology_reliability_aware_pfp"),
            wandb_entity=hparams.get("wandb_entity"),
            wandb_mode=hparams.get("wandb_mode", "online"),
            ablation=ablation,
            run_type=run_type,
        )

    elif run_type == "full_training":
        run_model_training(
            promising_hparams=hparams["promising_hparams"],
            train_loader=train_loader,
            val_loader=val_loader,
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            go_terms=go_data.go_terms,
            child_parent_pairs=go_data.child_parent_pairs,
            ic=go_data.ic,
            build_model_fn=model_specific_params["build_model_fn"],
            fit_function=fit_model,
            device=device,
            final_epochs=hparams["final_epochs"],
            patience=hparams["patience"],
            base_dir=hparams["base_dir_final"],
            use_wandb=hparams.get("use_wandb", False),
            wandb_project=hparams.get("wandb_project", "seq_homology_reliability_aware_pfp"),
            wandb_entity=hparams.get("wandb_entity"),
            wandb_mode=hparams.get("wandb_mode", "online"),
            ablation=ablation,
            run_type=run_type,
        )


if __name__ == "__main__":
    main()
