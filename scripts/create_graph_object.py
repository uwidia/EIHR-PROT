from reliability_aware.graph_shard_builder import build_aligned_graph_shards
from pathlib import Path
import reliability_aware.config as config
from reliability_aware.parser import get_protein_info
from reliability_aware.graph_shard_builder import build_aligned_graph_shards, process_single_entry


config.setup_logging()

def main():
    ESM_MANIFEST_PATH = config.PROJECT_ROOT / "esm_manifests"
    CLEANED_PDB_DIR = config.DATA_DIR / f"cleaned_dataset/pdb"
    STRUCTURE_DIR = config.PROJECT_ROOT / "structures/pdb"
    OUTPUT_DIR = config.PROJECT_ROOT / "graph_shards"

    for split in ["train", "test", "val"]:   
        fasta_path = CLEANED_PDB_DIR / f"cleaned_pdb_{split}.fasta" 
        structure_file = STRUCTURE_DIR / f"pdb_{split}"
        shard_build_dataset = build_aligned_graph_shards(
            manifest_path = ESM_MANIFEST_PATH / f"pdb_{split}_manifest.csv" ,
            fasta_path = fasta_path,
            structure_dir = structure_file,
            output_dir = OUTPUT_DIR / f"pdb_{split}",
            get_protein_info_fn = get_protein_info,
            process_single_entry_fn = process_single_entry,
            cutoff = 10.0,
        )

if __name__ == "__main__":
    main()
