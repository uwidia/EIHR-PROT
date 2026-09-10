"""
Examples:
  python scripts/run_model_training.py --ablation sequence_only --go_aspect BP --hparams configs/new_main_models/sequence_only_search.yaml --run_type randomized_search
  python scripts/run_model_training.py --ablation sequence_homology_confidence_gate --go_aspect BP --hparams configs/new_main_models/sequence_homology_confidence_gate_search.yaml --run_type randomized_search
  python scripts/run_model_training.py --ablation homology_only --go_aspect BP --hparams configs/new_main_models/homology_only_eval.yaml --run_type evaluate_only
"""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import yaml
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTIVE_ABLATIONS = [
    "sequence_only",
    "homology_only",
    "sequence_homology_internal_gate",
    "sequence_homology_confidence_gate",
    "sequence_homology_fixed_fusion",
    "sequence_homology_identity_fusion",
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
    parser.add_argument("--search-dir", type=Path, help="Existing fusion search directory containing aspect subdirectories.")
    parser.add_argument("--check-only", action="store_true", help="Validate fusion final-training inputs and candidates without training.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.hparams) as f:
        hparams = yaml.safe_load(f) or {}

    ablation = args.ablation.lower()
    run_type = args.run_type.lower()
    if (args.search_dir is not None or args.check_only) and (
            run_type != "full_training" or ablation not in {"sequence_homology_fixed_fusion", "sequence_homology_identity_fusion"}):
        raise ValueError("--search-dir and --check-only require fusion full_training")

    if ablation == "homology_only" and run_type != "evaluate_only":
        raise ValueError(
            "homology_only is evaluation-only; use --run_type evaluate_only."
        )
    if ablation != "homology_only" and run_type == "evaluate_only":
        raise ValueError(
            "evaluate_only is currently supported only for homology_only. "
            "Use randomized_search or full_training for trainable models."
        )

    # Apply process settings before torch/numpy imports or CUDA initialization.
    selected_hparams = hparams.get("promising_hparams", {}).get(args.go_aspect.upper(), {})
    if selected_hparams.get("reproducible_training", False):
        if ablation != "sequence_homology_confidence_gate" or run_type != "full_training" or hparams.get("use_search_results", False):
            raise ValueError("reproducible_training requires confidence-gate full_training with fixed promising_hparams")
        from reliability_aware.utils.reproducibility import prepare_training_process, configure_training, resolve_cpu_threads
        selected_hparams["cpu_threads"] = resolve_cpu_threads(selected_hparams.get("cpu_threads"))
        prepare_training_process(selected_hparams["training_seed"], selected_hparams["cpu_threads"])
        configure_training(selected_hparams["training_seed"], cpu_threads=selected_hparams["cpu_threads"])

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
        build_sequence_homology_internal_gate_model,
        run_one_batch_smoke_test_sequence_homology_confidence_gate,
        run_one_batch_smoke_test_sequence_homology_internal_gate,
    )
    from models.sequence_homology_fusion_baselines import (
        build_sequence_homology_fixed_fusion_model,
        build_sequence_homology_identity_fusion_model,
    )
    from models.identity_fusion_data import (
        IdentitySequenceHomologyShardDataset,
        make_identity_sequence_homology_collate_fn,
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

    from reliability_aware.utils.fusion_protocol import (
        BASELINES, load_resources, validation_keep_ids, protocol_metadata,
        bind_run_directory, selected_candidates,
    )
    is_fusion = ablation in BASELINES
    resources = load_resources(hparams['resources']) if is_fusion else None
    train_dataset, val_dataset = config.train_dataset, config.val_dataset
    train_manifest, val_manifest = config.train_manifest_path, config.val_manifest_path
    train_esm, val_esm = config.train_esm_shard_dir, config.val_esm_shard_dir
    annotation_path, obo_path = config.go_annotation_path, config.obo_path
    if is_fusion:
        train, val = resources['pdb_train'], resources['pdb_val']
        train_dataset, val_dataset = train['fasta'], val['fasta']
        train_manifest, val_manifest = train['manifest'], val['manifest']
        train_esm, val_esm = train['esm_shards'], val['esm_shards']
        train_homology_shard_dir = Path(train['homology_shards'].format(aspect=go_aspect))
        val_homology_shard_dir = Path(val['homology_shards'].format(aspect=go_aspect))
        annotation_path, obo_path = resources['train_annotations'], resources['obo']
        hparams['train_identity_sidecar_path'] = train['identity']
        hparams['val_identity_sidecar_path'] = val['identity']

    go_data = build_go_annotation_data(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        go_annotation_path=annotation_path,
        obo_path=obo_path,
        go_aspect=go_aspect,
        device=device,
    )

    baseline_options = {}
    final_candidates = None
    if is_fusion:
        go_data.val_keep_ids = validation_keep_ids(ablation, go_data.val_keep_ids, resources['validation_exclude_ids'])
        vocab = json.loads(Path(resources['go_vocab'].format(aspect=go_aspect)).read_text())
        if vocab != go_data.go_terms:
            raise ValueError('Training vocabulary/order differs from the existing homology vocabulary')
        from reliability_aware.utils.fusion_preparation import verify_frozen_priors
        for split in ('pdb_train', 'pdb_val'):
            verify_frozen_priors(resources, split, go_aspect)
            if ablation.endswith('identity_fusion'):
                from models.identity_fusion_data import load_identity_sidecar
                path = resources[split]['identity']
                load_identity_sidecar(path)
                payload = json.loads(Path(path).read_text())
                expected_excluded = sorted(resources['validation_exclude_ids']) if split == 'pdb_val' else []
                if payload.get('dataset_split') != split or payload.get('excluded_query_ids') != expected_excluded or payload.get('exclude_self_hits') != (split == 'pdb_train'):
                    raise ValueError(f'Identity sidecar split/exclusion/self-hit policy mismatch: {path}')
        metadata = protocol_metadata(ablation=ablation, aspect=go_aspect, go_terms=go_data.go_terms,
                                     resources=resources, hparams=hparams,
                                     train_ids=go_data.train_keep_ids, val_ids=go_data.val_keep_ids)
        if run_type == 'full_training':
            if not hparams.get('use_search_results', False):
                raise ValueError('Fusion baselines require use_search_results: true')
            from reliability_aware.utils.fusion_search_reuse import search_results_path, prepare_final_candidates
            selected_path = search_results_path(hparams, go_aspect, args.search_dir)
            final_candidates, source_search = prepare_final_candidates(
                selected_path, int(hparams['top_k_params']), metadata,
                allow_relocated=args.search_dir is not None)
            metadata['source_search'] = source_search
            report_path = Path(resources['prepared_dir']) / f'{ablation}_{go_aspect}_final_preflight.json'
            report_path.write_text(json.dumps({'status': 'matched', **source_search}, indent=2, sort_keys=True))
            logger.info('Final handoff %s: %d candidate(s) from %s; %s',
                        go_aspect, len(final_candidates), selected_path, source_search['comparison']['mode'])
            if args.check_only:
                print(f'{go_aspect}: final preflight matched; {len(final_candidates)} candidate(s); no training started', flush=True)
                return
        bind_run_directory(Path(hparams['base_dir_search' if run_type == 'randomized_search' else 'base_dir_final']) / go_aspect, metadata)
        baseline_options = {'seed': int(hparams.get('seed', 42)), 'checkpoint_extra': metadata}
        logger.info('Fusion validation excludes %s; %d eligible proteins remain for %s',
                    resources['validation_exclude_ids'], len(go_data.val_keep_ids), go_aspect)

    run_parameters = {
        "sequence_only": {
            "build_model_fn": build_seq_only_model,
            "dataset_cls": SequenceOnlyESMShardDataset,
            "dataset_kind": "sequence",
            "collate_factory": make_sequence_only_collate_fn,
            "filter_invalid_samples": False,
            "smoke_test_fn": run_one_batch_smoke_test_sequence_only,
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
        "sequence_homology_fixed_fusion": {
            "build_model_fn": build_sequence_homology_fixed_fusion_model,
            "dataset_cls": SequenceHomologyShardDataset,
            "dataset_kind": "sequence_homology",
            "collate_factory": make_sequence_homology_collate_fn,
            "filter_invalid_samples": False,
            "smoke_test_fn": None,
        },
        "sequence_homology_identity_fusion": {
            "build_model_fn": build_sequence_homology_identity_fusion_model,
            "dataset_cls": IdentitySequenceHomologyShardDataset,
            "dataset_kind": "identity_sequence_homology",
            "collate_factory": make_identity_sequence_homology_collate_fn,
            "filter_invalid_samples": False,
            "smoke_test_fn": None,
        },
    }

    if ablation == "homology_only":
        metrics = run_homology_only_evaluation(
            val_homology_shard_dir=val_homology_shard_dir,
            val_manifest_path=val_manifest,
            val_label_to_indices=go_data.val_label_to_indices,
            val_keep_ids_for_aspect=go_data.val_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            go_terms=go_data.go_terms,
            child_parent_pairs=go_data.child_parent_pairs,
            go_aspect=go_aspect,
            obo_path=obo_path,
            train_annotations=go_data.train_annotations,
            device=device,
            lambda_hier=float(hparams.get("lambda_hier", 0.0)),
            pos_weight_cap=float(hparams.get("pos_weight_cap", 20.0)),
            batch_size=batch_size,
            out_dir=Path(hparams["base_dir_eval"]) / go_aspect,
            use_wandb=hparams.get("use_wandb", False),
            wandb_project=hparams.get(
                "wandb_project", "seq_homology_reliability_aware_pfp"
            ),
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
        train_esm_shard_dir=train_esm,
        val_esm_shard_dir=val_esm,
        train_homology_shard_dir=train_homology_shard_dir,
        val_homology_shard_dir=val_homology_shard_dir,
        train_identity_sidecar_path=hparams.get("train_identity_sidecar_path"),
        val_identity_sidecar_path=hparams.get("val_identity_sidecar_path"),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
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
        **({"seed": baseline_options["seed"]} if is_fusion else {}),
    )

    if run_type == "randomized_search":
        randomized_search.run_randomized_search(
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            go_terms=go_data.go_terms,
            child_parent_pairs=go_data.child_parent_pairs,
            go_aspect=go_aspect,
            obo_path=obo_path,
            train_annotations=go_data.train_annotations,
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
            base_dir=Path(hparams["base_dir_search"]) / go_aspect,
            smoke_test=True,
            top_k_params=hparams["top_k_params"],
            use_wandb=hparams.get("use_wandb", False),
            wandb_project=hparams.get(
                "wandb_project", "seq_homology_reliability_aware_pfp"
            ),
            wandb_entity=hparams.get("wandb_entity"),
            wandb_mode=hparams.get("wandb_mode", "online"),
            ablation=ablation,
            run_type=run_type,
            **baseline_options,
        )

    elif run_type == "full_training":
        if final_candidates is not None:
            promising_hparams = final_candidates
        elif hparams.get("use_search_results", False):
            selected_path = hparams.get("search_results_path") or (Path(hparams["base_dir_search"]) / go_aspect / "search_results.json")
            selected_path = Path(selected_path)
            if not selected_path.exists():
                raise FileNotFoundError(f"Missing sorted search results: {selected_path}; run randomized_search first")
            promising_hparams = (selected_candidates(selected_path, int(hparams['top_k_params']), metadata)
                                 if is_fusion else [row["hparams"] for row in json.loads(selected_path.read_text())[: int(hparams["top_k_params"])]] )
            if not promising_hparams:
                raise ValueError("Search results contained no candidate hyperparameters")
        else:
            if is_fusion:
                raise ValueError('Fusion baselines require use_search_results: true')
            promising_hparams = [hparams["promising_hparams"][go_aspect]]
        run_model_training(
            promising_hparams=promising_hparams,
            train_loader=train_loader,
            val_loader=val_loader,
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            go_terms=go_data.go_terms,
            child_parent_pairs=go_data.child_parent_pairs,
            go_aspect=go_aspect,
            obo_path=obo_path,
            train_annotations=go_data.train_annotations,
            build_model_fn=model_specific_params["build_model_fn"],
            fit_function=fit_model,
            device=device,
            final_epochs=hparams["final_epochs"],
            patience=hparams["patience"],
            base_dir=Path(hparams["base_dir_final"]) / go_aspect,
            use_wandb=hparams.get("use_wandb", False),
            wandb_project=hparams.get(
                "wandb_project", "seq_homology_reliability_aware_pfp"
            ),
            wandb_entity=hparams.get("wandb_entity"),
            wandb_mode=hparams.get("wandb_mode", "online"),
            ablation=ablation,
            run_type=run_type,
            **baseline_options,
        )


if __name__ == "__main__":
    main()
