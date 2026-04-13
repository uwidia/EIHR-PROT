#Import libraries
import reliability_aware.config as config
import reliability_aware.preprocessing as preprocessing
import logging
from reliability_aware.utils import setup_logging
logger = logging.getLogger(__name__)

setup_logging()

datasets = [
    ("pdb", "train", config.RAW_PDB_TRAIN_DATA),
    ("pdb", "test", config.RAW_PDB_TEST_DATA),
    ("pdb", "val", config.RAW_PDB_VAL_DATA),
    ("af", "train", config.RAW_AF_TRAIN_DATA),
    ("af", "test", config.RAW_AF_TEST_DATA),
    ("af", "val", config.RAW_AF_VAL_DATA),
]

pipeline_results = {}

def main():
    for dataset_type, split, fasta_file_path in datasets:
        output_file = f"cleaned_{dataset_type}_{split}"
        structure_dir = config.PROJECT_ROOT / f"structures/{dataset_type}/{dataset_type}_{split}"

        if dataset_type not in pipeline_results:
            pipeline_results[dataset_type] = {}


        pipeline_results[dataset_type][split] = {}

        #Download AFSet and PDBset files
        
        structure_dir.mkdir(parents = True, exist_ok = True)

        download_fn = preprocessing.download_one_af_structure if dataset_type == "af" else preprocessing.download_one_pdb_structure


        logger.info(f"Downloading structures for {dataset_type}_{split}....")
        download_result = preprocessing.download_multiple_structures_fast(
            fasta_file_path, structure_dir, download_fn
        )

        pipeline_results[dataset_type][split]["download"] = download_result

        # Handle PDBSet-specific concerns 
        if dataset_type == "pdb":

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

            pipeline_results[dataset_type][split]["xray_structure_filter"] = filter_result
            logger.info(f"Non-xray sequences deleted. IDs of retained protein sequences saved to {config.RETAINED_XRAY_IDS}_{split}.txt")

            # Create new fasta files per split with xray-derived PDBset sequences only
            retained_xray_ids = f"{config.RETAINED_XRAY_IDS}_{split}.txt" #file already created by delete_non_xray_structures function

            preprocessing.filter_xray_struct(
                retained_xray_ids,
                fasta_file_path,
                output_file
            )

            pipeline_results[dataset_type][split]["xray_fasta_sequence_filter"] = {
                "input_fasta": str(fasta_file_path),
                "xray_ids": str(retained_xray_ids),
                "output_name": output_file
            }


        #Handle AFset specific concerns
        if dataset_type == "af":
            preprocessing.clean_af_fasta_file_header(
                fasta_file_path,
                output_file
            )

            pipeline_results[dataset_type][split]["clean_af_fasta_file_header"] = {
                "input_fasta": str(fasta_file_path),
                "output_name": output_file
            }
if __name__ == "__main__":
    main()
