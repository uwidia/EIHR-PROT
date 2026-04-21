import reliability_aware.preprocessing as preprocessing
import logging
from reliability_aware.config import setup_logging
logger = logging.getLogger(__name__)

setup_logging()

HEAL_DIR = DATA_DIR / "HEAL_dataset"
STRUCTURE_DIR = config.PROJECT_ROOT / "structures/pdb"
OUTPUT_FILE = 
RETAINED_XRAY_IDS = DATA_DIR / "retained_xray_ids_pdb"


pipeline_results = {}

def main():
    for split in ["train", "test", "val"]:
        fasta_file_path = HEAL_DIR / f"nrPDB-GO_2019.06.18_{split}_sequences.fasta"
        output_file = f"cleaned_pdb_{split}"
        structure_dir = STRUCTURE_DIR / f"pdb_{split}"

        if split not in pipeline_results:
            pipeline_results[split] = {}

        #Download PDBset files
        structure_dir.mkdir(parents = True, exist_ok = True)

        download_fn = preprocessing.download_one_pdb_structure

        logger.info(f"Downloading structures for {dataset_type}_{split}....")
        download_result = preprocessing.download_multiple_structures_fast(
            fasta_file_path, structure_dir, download_fn
        )

        pipeline_results[split]["download"] = download_result

        #Count x-ray derived structures
        """
        The count step below has been commented out due to how long it takes to execute. 
        You can uncomment it if you wish to verify counts yourself
        """
        # counts_result = preprocessing.count_methods_per_split(split, save_dir)
        # pipeline_results[dataset_type][split]["counts"] = counts_result

        #Remove all non-xray derived structure files from each split
        deleted, failed, missing, retained = preprocessing.delete_non_xray_structures(split, structure_dir)

        filter_result = {
            "deleted": deleted,
            "failed": failed,
            "missing": missing,
            "retained": retained
        }

        pipeline_results[split]["xray_structure_filter"] = filter_result
        logger.info(f"Non-xray sequences deleted. IDs of retained protein sequences saved to {RETAINED_XRAY_IDS}_{split}.txt")

        # Create new fasta files per split with xray-derived PDBset sequences only
        retained_xray_ids = f"{RETAINED_XRAY_IDS}_{split}.txt" #file already created by delete_non_xray_structures function

        preprocessing.filter_xray_struct(
            retained_xray_ids,
            fasta_file_path,
            output_file
        )

        pipeline_results[split]["xray_fasta_sequence_filter"] = {
            "input_fasta": str(fasta_file_path),
            "xray_ids": str(retained_xray_ids),
            "output_name": output_file
        }

if __name__ == "__main__":
    main()
