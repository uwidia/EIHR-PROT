"""
Call with uv run python scripts/create_graph_object.py --paths "specify path to paths config file"
"""

from utils.graph_shard_builder import build_aligned_graph_shards
from pathlib import Path
from utils.config import setup_logging, PROJECT_ROOT
from utils.parser import get_protein_info
from utils.graph_shard_builder import build_aligned_graph_shards, process_single_entry
import yaml
import argparse

setup_logging()

parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=str, required=True)
args = parser.parse_args()


with open(args.paths) as f:
    paths = yaml.safe_load(f)

esm_manifest_path = PROJECT_ROOT / paths["esm_manifest_path"]
cleaned_pdb_dir = PROJECT_ROOT / paths["cleaned_pdb_dir"]
structure_dir = PROJECT_ROOT / paths["structure_dir"]
output_dir = PROJECT_ROOT / paths["graph_object_output_dir"]


def main():
    for split in ["train", "test", "val"]:
        fasta_path = cleaned_pdb_dir / f"cleaned_pdb_{split}.fasta"
        structure_file = structure_dir / f"pdb_{split}"
        build_aligned_graph_shards(
            manifest_path=esm_manifest_path / f"pdb_{split}_manifest.csv",
            fasta_path=fasta_path,
            structure_dir=structure_file,
            output_dir=output_dir / f"pdb_{split}",
            get_protein_info_fn=get_protein_info,
            process_single_entry_fn=process_single_entry,
            cutoff=10.0,
        )


if __name__ == "__main__":
    main()
