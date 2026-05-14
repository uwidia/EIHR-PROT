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
from models.sequence_only_ablation import (
    build_seq_only_model,
    build_sequence_only_loaders,
    fit_sequence_only,
)
from models.reliability_aware_model import (
    build_reliability_aware_model,
    build_reliability_aware_model_loaders,
    fit_reliability_aware_model,
)

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", type=str, required=True)
parser.add_argument("--go_aspect", type=str, required=True)
parser.add_argument("--hparams", type=str, required=True)
args = parser.parse_args()

with open(args.hparams) as f:
    hparams = yaml.safe_load(f)

config.setup_logging()
logger = logging.getLogger(__name__)

ablation = args.ablation.lower()

if ablation not in ["sequence_only", "reliability_aware_model"]:
    raise ValueError(
        f"--ablation must be one of sequence_only, reliability_aware_model"
    )

go_aspect = args.go_aspect.upper()
go_annotation_path = config.go_annotation_path
obo_path = config.obo_path

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

promising_hparams = hparams["promising_hparams"]
final_epochs = hparams["final_epochs"]
patience = hparams["patience"]
batch_size = hparams["batch_size"]
base_dir_final = hparams["base_dir_final"]

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
        "fit_function": fit_sequence_only,
        "build_model_fn": build_seq_only_model,
    },
    "reliability_aware_model": {
        "fit_function": fit_reliability_aware_model,
        "build_model_fn": build_reliability_aware_model,
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
        fit_function=run_parameters[ablation]["fit_function"],
        device=device,
        final_epochs=final_epochs,
        patience=patience,
        base_dir=base_dir_final,
    )


if __name__ == "__main__":
    main()
