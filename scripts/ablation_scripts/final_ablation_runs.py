"""
Call with:
uv run python scripts/ablation_scripts/final_ablation_runs.py --ablation "specify ablation" --paths "specify path config file" --go_aspect "select go_aspect" \
    --hparams "path to specific config file with ablation hyperparameters"
"""

from utils.model_training import run_seq_only_ablation_training
import torch
import argparse
from utils.model_training import build_go_annotation_data
from utils.config import setup_logging, PROJECT_ROOT
import yaml
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", type=str, required=True)
parser.add_argument("--paths", type=str, required=True)
parser.add_argument("--go_aspect", type=str)
parser.add_argument("--hparams", type=str, required=True)
args = parser.parse_args()


with open(args.paths) as f:
    paths = yaml.safe_load(f)

with open(args.hparams) as f:
    hparams = yaml.safe_load(f)


setup_logging()
logger = logging.getLogger(__name__)

ablation = args.ablation.lower()

if ablation not in ["sequence_only"]:
    raise ValueError(f"--ablation must be one of sequence_only, structure_only, ...")


# gene ontology parameters and paths
go_aspect = args.go_aspect.upper()
go_annotation_path = PROJECT_ROOT / paths["go_annotation_path"]
obo_path = PROJECT_ROOT / paths["obo_path"]
pdb_fasta_dir = PROJECT_ROOT / paths["cleaned_pdb_dir"]


# training dataset paths
train_graph_shard_dir = PROJECT_ROOT / paths["train_graph_shard_dir"]
train_esm_shard_dir = PROJECT_ROOT / paths["train_esm_shard_dir"]
train_homology_shard_dir = PROJECT_ROOT / paths["train_homology_shard_dir"]
train_manifest_path = PROJECT_ROOT / paths["train_manifest_path"]
train_dataset = PROJECT_ROOT / paths["train_dataset"]


# validation dataset paths
val_graph_shard_dir = PROJECT_ROOT / paths["val_graph_shard_dir"]
val_homology_shard_dir = PROJECT_ROOT / paths["val_homology_shard_dir"]
val_esm_shard_dir = PROJECT_ROOT / paths["val_esm_shard_dir"]
val_manifest_path = PROJECT_ROOT / paths["val_manifest_path"]
val_dataset = PROJECT_ROOT / paths["val_dataset"]


# get hyperparameters
promising_hparams = hparams["promising_hparams"]
final_epochs = hparams["final_epochs"]
patience = hparams["patience"]
batch_size = hparams["batch_size"]
base_dir_final = hparams["base_dir_final"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

go_data = build_go_annotation_data(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    go_annotation_path=go_annotation_path,
    obo_path=obo_path,
    go_aspect=go_aspect,
    device=device,
)


def main():

    if ablation == "sequence_only":
        run_seq_only_ablation_training(
            promising_hparams=promising_hparams,
            train_esm_shard_dir=train_esm_shard_dir,
            val_esm_shard_dir=val_esm_shard_dir,
            train_manifest_path=train_manifest_path,
            val_manifest_path=val_manifest_path,
            train_keep_ids_for_aspect=go_data.train_keep_ids,
            val_keep_ids_for_aspect=go_data.val_keep_ids,
            train_label_to_indices=go_data.train_label_to_indices,
            val_label_to_indices=go_data.val_label_to_indices,
            go_terms=go_data.go_terms,
            child_parent_pairs=go_data.child_parent_pairs,
            ic=go_data.ic,
            device=device,
            final_epochs=final_epochs,
            patience=patience,
            batch_size=batch_size,
            base_dir=base_dir_final,
        )


if __name__ == "__main__":
    main()
