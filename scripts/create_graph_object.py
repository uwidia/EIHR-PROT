from reliability_aware.graph_shard_builder import build_aligned_graph_shards
from pathlib import Path
import reliability_aware.config as config
from reliability_aware.parser import get_protein_info
from reliability_aware.graph_shard_builder import build_aligned_graph_shards, process_single_entry


config.setup_logging()

def main():
    datasets = [
        ("af", "train"),
        ("af", "test"),
        ("af", "val"),
        ("pdb", "train"),
        ("pdb", "test"),
        ("pdb", "val")
    ]

    ESM_MANIFEST_PATH = config.PROJECT_ROOT / "esm_manifests"

    for dataset_type, split in datasets:   
        fasta_path = config.DATA_DIR / f"cleaned_dataset/{dataset_type}/cleaned_{dataset_type}_{split}.fasta" 
        structure_file = config.PROJECT_ROOT / f"structures/{dataset_type}/{dataset_type}_{split}"
        shard_build_dataset = build_aligned_graph_shards(
            manifest_path = ESM_MANIFEST_PATH / f"{dataset_type}_{split}_manifest.csv" ,
            fasta_path = fasta_path,
            structure_dir = structure_file,
            output_dir = config.PROJECT_ROOT / f"new_graph_shards/{dataset_type}_graph_shards/{split}",
            get_protein_info_fn = get_protein_info,
            process_single_entry_fn = process_single_entry,
            cutoff = 10.0,
        )

if __name__ == "__main__":
    main()
