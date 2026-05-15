"""
Call with:
uv run python scripts/ablation_scripts/final_ablation_runs.py --ablation "specify ablation" --go_aspect "select go_aspect" \
    --hparams "path to specific config file with ablation hyperparameters"
"""

import torch
import argparse
import yaml
import logging
from utils import config
from utils.model_training import build_go_annotation_data, run_model_training
import utils.model_randomized_search as randomized_search
from utils.model_training import fit_model

from models.sequence_only_ablation import (
    run_one_batch_smoke_test_sequence_only,
    build_seq_only_model,
    build_sequence_only_loaders,
)
from models.reliability_aware_model import (
    run_one_batch_smoke_test,
    build_reliability_aware_model,
    build_reliability_aware_model_loaders,
)

from models.structure_only_ablation import (
    run_one_batch_smoke_test_structure_only,
    build_structure_only_model,
    build_structure_only_loaders,
)

from models.seq_structure_ablation import (
    run_one_batch_smoke_test_seq_structure,
    build_seq_structure_model,
    build_seq_structure_loaders,
)

from models.averaging_baseline_ablation import (
    run_one_batch_smoke_test_averaging,
    build_averaging_model,
    build_averaging_loaders,
)

from models.internal_gate_baseline_ablation import (
    run_one_batch_smoke_test_internal_gate,
    build_internal_gate_model,
    build_internal_gate_loaders,
)

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", type=str, required=True)
parser.add_argument("--go_aspect", type=str, required=True)
parser.add_argument("--hparams", type=str, required=True)
parser.add_argument("--run_type", type=str, required=True)
parser.add_argument(
    "--ablation",
    type=str.lower,
    required=True,
    choices=[
        "sequence_only",
        "structure_only",
        "seq_structure",
        "averaging_baseline",
        "internal_gate_baseline",
        "reliability_aware_model",
    ],
)

parser.add_argument(
    "--run_type",
    type=str.lower,
    required=True,
    choices=["randomized_search", "full_training"],
)
args = parser.parse_args()

with open(args.hparams) as f:
    hparams = yaml.safe_load(f)

config.setup_logging()
logger = logging.getLogger(__name__)

ablation = args.ablation.lower()
run_type = args.run_type.lower()

go_aspect = args.go_aspect.upper()
go_annotation_path = config.go_annotation_path
obo_path = config.obo_path
batch_size = hparams["batch_size"]

train_esm_shard_dir = config.train_esm_shard_dir
train_graph_shard_dir = config.train_graph_shard_dir
train_homology_shard_dir = config.train_homology_shard_dir
train_manifest_path = config.train_manifest_path
train_dataset = config.train_dataset

val_esm_shard_dir = config.val_esm_shard_dir
val_graph_shard_dir = config.val_graph_shard_dir
val_homology_shard_dir = config.val_homology_shard_dir
val_manifest_path = config.val_manifest_path
val_dataset = config.val_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

go_data = build_go_annotation_data(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    go_annotation_path=go_annotation_path,
    obo_path=obo_path,
    go_aspect=go_aspect,
    device=device,
)


run_parameters = {
    "sequence_only": {
        "build_model_fn": build_seq_only_model,
        "loader_fn": build_sequence_only_loaders,
        "smoke_test_fn": run_one_batch_smoke_test_sequence_only,
    },
    "structure_only": {
        "build_model_fn": build_structure_only_model,
        "loader_fn": build_structure_only_loaders,
        "smoke_test_fn": run_one_batch_smoke_test_structure_only,
    },
    "reliability_aware_model": {
        "build_model_fn": build_reliability_aware_model,
        "loader_fn": build_reliability_aware_model_loaders,
        "smoke_test_fn": run_one_batch_smoke_test,
    },
    "seq_structure": {
        "build_model_fn": build_seq_structure_model,
        "loader_fn": build_seq_structure_loaders,
        "smoke_test_fn": run_one_batch_smoke_test_seq_structure,
    },
    "averaging_baseline": {
        "build_model_fn": build_averaging_model,
        "loader_fn": build_averaging_loaders,
        "smoke_test_fn": run_one_batch_smoke_test_averaging,
    },
    "internal_gate_baseline": {
        "build_model_fn": build_internal_gate_model,
        "loader_fn": build_internal_gate_loaders,
        "smoke_test_fn": run_one_batch_smoke_test_internal_gate,
    },
}


def main():
    train_loader = None
    val_loader = None

    loader_kwargs = dict(
        train_esm_shard_dir=train_esm_shard_dir,
        val_esm_shard_dir=val_esm_shard_dir,
        train_graph_shard_dir=train_graph_shard_dir,
        val_graph_shard_dir=val_graph_shard_dir,
        train_homology_shard_dir=train_homology_shard_dir,
        val_homology_shard_dir=val_homology_shard_dir,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        train_keep_ids_for_aspect=go_data.train_keep_ids,
        val_keep_ids_for_aspect=go_data.val_keep_ids,
        train_label_to_indices=go_data.train_label_to_indices,
        val_label_to_indices=go_data.val_label_to_indices,
        go_terms=go_data.go_terms,
        batch_size=batch_size,
    )

    _, _, train_loader, val_loader = run_parameters[ablation]["loader_fn"](
        **loader_kwargs
    )

    if run_type == "randomized_search":

        num_trials = hparams["num_trials"]
        trial_epochs = hparams["trial_epochs"]
        top_k_params = hparams["top_k_params"]
        patience = hparams["patience"]
        base_dir_search = hparams["base_dir_search"]

        search_space = hparams["search_space"]

        randomized_search.run_randomized_search(
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            go_terms=go_data.go_terms,
            child_parent_pairs=go_data.child_parent_pairs,
            ic=go_data.ic,
            search_space=search_space,
            device=device,
            num_trials=num_trials,
            trial_epochs=trial_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            fit_function=fit_model,
            build_model_fn=run_parameters[ablation]["build_model_fn"],
            smoke_test_fn=run_parameters[ablation]["smoke_test_fn"],
            patience=patience,
            base_dir=base_dir_search,
            smoke_test=True,
            top_k_params=top_k_params,
        )

    elif run_type == "full_training":

        promising_hparams = hparams["promising_hparams"]
        final_epochs = hparams["final_epochs"]
        patience = hparams["patience"]
        base_dir_final = hparams["base_dir_final"]

        run_model_training(
            promising_hparams=promising_hparams,
            train_loader=train_loader,
            val_loader=val_loader,
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            go_terms=go_data.go_terms,
            child_parent_pairs=go_data.child_parent_pairs,
            ic=go_data.ic,
            build_model_fn=run_parameters[ablation]["build_model_fn"],
            fit_function=fit_model,
            device=device,
            final_epochs=final_epochs,
            patience=patience,
            base_dir=base_dir_final,
        )


if __name__ == "__main__":
    main()
