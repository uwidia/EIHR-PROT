from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]

diamond_directory = PROJECT_ROOT / "diamond_db"
diamond_executable_candidates = [
    PROJECT_ROOT / "diamond.exe",
    PROJECT_ROOT / "diamond",
]
diamond_executable_path = next(
    (path for path in diamond_executable_candidates if path.exists()),
    diamond_executable_candidates[-1],
)

go_annotation_path = PROJECT_ROOT / "data/HEAL_dataset/nrPDB-GO_2019.06.18_annot.tsv"
obo_path = PROJECT_ROOT / "data/HEAL_dataset/go-basic.obo"

train_esm_shard_dir = PROJECT_ROOT / "esm_embeddings/train"
train_dataset = PROJECT_ROOT / "data/cleaned_dataset/cleaned_pdb_train.fasta"
train_manifest_path = PROJECT_ROOT / "esm_embeddings/train/train_manifest.csv"

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
