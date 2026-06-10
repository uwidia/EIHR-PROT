from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]

diamond_directory = PROJECT_ROOT / "diamond_db"
diamond_executable_path = PROJECT_ROOT / "diamond"

esm_manifest_path = PROJECT_ROOT / "esm_manifests"
cleaned_pdb_dir = PROJECT_ROOT / "data/cleaned_dataset"
cleaned_pdb_train = PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_train.fasta"
cleaned_pdb_val = PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_val.fasta"

raw_pdb_train = (
    PROJECT_ROOT / "data/HEAL_dataset/nrPDB-GO_2019.06.18_train_sequences.fasta"
)

raw_pdb_val = PROJECT_ROOT / "data/HEAL_dataset/nrPDB-GO_2019.06.18_val_sequences.fasta"
raw_pdb_test = (
    PROJECT_ROOT / "data/HEAL_dataset/nrPDB-GO_2019.06.18_train_sequences.fasta"
)

structure_dir = PROJECT_ROOT / "structures/pdb"
graph_object_output_dir = PROJECT_ROOT / "graph_shards"
go_annotation_path = PROJECT_ROOT / "data/HEAL_dataset/nrPDB-GO_2019.06.18_annot.tsv"
obo_path = PROJECT_ROOT / "data/HEAL_dataset/go-basic.obo"
pdb_fasta_dir = PROJECT_ROOT / "data/cleaned_dataset"
retained_xray_ids_out_dir = PROJECT_ROOT / "data/retained_xray_ids"

# training dataset paths
train_graph_shard_dir = PROJECT_ROOT / "graph_shards/train"
train_esm_shard_dir = PROJECT_ROOT / "esm_embeddings/train"
train_dataset = PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_train.fasta"
train_manifest_path = PROJECT_ROOT / "esm_embeddings/train/train_manifest.csv"

# validation dataset paths
val_esm_shard_dir = PROJECT_ROOT / "esm_embeddings/val"
val_dataset = PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_val.fasta"
val_manifest_path = PROJECT_ROOT / "esm_embeddings/val/val_manifest.csv"


def setup_logging():
    logger = logging.getLogger()

    if logger.handlers:
        return logger

    handlers = [logging.StreamHandler(), logging.FileHandler("pipeline.log", mode="a")]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        handlers=handlers,
    )

    return logger
