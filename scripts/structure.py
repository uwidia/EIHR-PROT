import reliability_aware.structure_processing as structure_processing
from pathlib import Path
from reliability_aware.utils import setup_logging
import reliability_aware.config as config

setup_logging()

def main():
    datasets = [
        ("af", "train"),
        ("af", "test"),
        ("af", "val"),
        ("pdb", "train"),
        ("pdb", "test"),
        ("pdb", "val")
    ]

    for dataset_type, split in datasets:   
        fasta_path = config.DATA_DIR / f"cleaned_dataset/{dataset_type}/cleaned_{dataset_type}_{split}.fasta" 
        structure_file = config.PROJECT_ROOT / f"structures/{dataset_type}/{dataset_type}_{split}"
        shard_build_dataset = structure_processing.create_shard_build_dataset(dataset_type, split, fasta_path, structure_file)
        structure_processing.build_graph_shards(shard_build_dataset, dataset_type, split)
        structure_processing.validate_random_samples(f"graph_shards/{dataset_type}_graph_shards/{split}")

if __name__ == "__main__":
    main()