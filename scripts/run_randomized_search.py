"""
Executes randomized search across all ablations

Call with:
uv run python scripts/ablation_scripts/ablation_randomized_search.py --ablation "specify ablation" --go_aspect "select go_aspect" \
    --hparams "path to specific config file with ablation hyperparameters"
"""

import utils.model_randomized_search as randomized_search
import torch
from utils import config
import yaml
import logging
import argparse
from utils.model_training import build_go_annotation_data
from models.sequence_only_ablation import (
    build_seq_only_model,
    build_sequence_only_loaders,
    fit_sequence_only,
    run_one_batch_smoke_test_sequence_only,
)

from models.reliability_aware_model import (
    build_reliability_aware_model,
    build_reliability_aware_model_loaders,
    run_one_batch_smoke_test,
    fit_reliability_aware_model,
)

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", type=str, required=True)
parser.add_argument("--go_aspect", type=str)
parser.add_argument("--hparams", type=str, required=True)

args = parser.parse_args()


config.setup_logging()
logger = logging.getLogger(__name__)

ablation = args.ablation.lower()

if ablation not in ["sequence_only", "reliability_aware_model"]:
    raise ValueError(
        f"--ablation must be one of sequence_only, structure_only, ... , reliability_aware_model"
    )


# gene ontology parameters and paths
go_aspect = args.go_aspect.upper()
go_annotation_path = config.go_annotation_path
obo_path = config.obo_path
pdb_fasta_dir = config.cleaned_pdb_dir


# training dataset paths
train_graph_shard_dir = config.train_graph_shard_dir
train_esm_shard_dir = config.train_esm_shard_dir
train_homology_shard_dir = config.train_homology_shard_dir
train_manifest_path = config.train_manifest_path
train_dataset = config.train_dataset


# validation dataset paths
val_graph_shard_dir = config.val_graph_shard_dir
val_homology_shard_dir = config.val_homology_shard_dir
val_esm_shard_dir = config.val_esm_shard_dir
val_manifest_path = config.val_manifest_path
val_dataset = config.val_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args.hparams) as f:
    hparams = yaml.safe_load(f)

num_trials = hparams["num_trials"]
trial_epochs = hparams["trial_epochs"]
top_k_params = hparams["top_k_params"]
patience = hparams["patience"]
batch_size = hparams["batch_size"]
base_dir_search = hparams["base_dir_search"]

go_data = build_go_annotation_data(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    go_annotation_path=go_annotation_path,
    obo_path=obo_path,
    go_aspect=go_aspect,
    device=device,
)

search_space = hparams["search_space"]

run_parameters = {
    "sequence_only": {
        "fit_function": fit_sequence_only,
        "build_model_fn": build_seq_only_model,
        "smoke_test_fn": run_one_batch_smoke_test_sequence_only,
    },
    "reliability_aware_model": {
        "fit_function": fit_reliability_aware_model,
        "build_model_fn": build_reliability_aware_model,
        "smoke_test_fn": run_one_batch_smoke_test,
    },
}


def main():

    train_loader = None
    val_loader = None

    if ablation == "sequence_only":

        _, _, train_loader, val_loader = build_sequence_only_loaders(
            train_esm_shard_dir=train_esm_shard_dir,
            val_esm_shard_dir=val_esm_shard_dir,
            train_manifest_path=train_manifest_path,
            val_manifest_path=val_manifest_path,
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            val_keep_ids_for_aspect=go_data.val_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            val_label_to_indices=go_data.val_label_to_indices,
            go_terms=go_data.go_terms,
            batch_size=batch_size,
        )

    elif ablation == "reliability_aware_model":

        _, _, train_loader, val_loader = build_reliability_aware_model_loaders(
            train_esm_shard_dir=train_esm_shard_dir,
            val_esm_shard_dir=val_esm_shard_dir,
            train_graph_shard_dir=train_graph_shard_dir,
            val_graph_shard_dir=val_graph_shard_dir,
            train_manifest_path=train_manifest_path,
            val_manifest_path=val_manifest_path,
            train_homology_shard_dir=train_homology_shard_dir,
            val_homology_shard_dir=val_homology_shard_dir,
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            val_keep_ids_for_aspect=go_data.val_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            val_label_to_indices=go_data.val_label_to_indices,
            go_terms=go_data.go_terms,
            batch_size=batch_size,
        )

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
        fit_function=run_parameters[ablation]["fit_function"],
        build_model_fn=run_parameters[ablation]["build_model_fn"],
        smoke_test_fn=run_parameters[ablation]["smoke_test_fn"],
        patience=patience,
        base_dir=base_dir_search,
        smoke_test=True,
        top_k_params=top_k_params,
    )


if __name__ == "__main__":
    main()
