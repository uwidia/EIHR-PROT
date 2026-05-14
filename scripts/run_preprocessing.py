import utils.preprocessing as preprocessing
import logging
import utils.config as config

logger = logging.getLogger(__name__)

config.setup_logging()


raw_dataset_dir = config.PROJECT_ROOT / "data/HEAL_dataset"
structure_file_dir = config.PROJECT_ROOT / "structures"
output_file_dir = config.PROJECT_ROOT / "data/cleaned_dataset"
retained_xray_ids_dir = config.retained_xray_ids_out_dir
pipeline_results = {}


def main():
    for split in ["train", "test", "val"]:
        fasta_file_path = (
            raw_dataset_dir / f"nrPDB-GO_2019.06.18_{split}_sequences.fasta"
        )
        output_file = output_file_dir / f"cleaned_pdb_{split}"
        structure_file_dir_per_split = structure_file_dir / f"pdb_{split}"

        if split not in pipeline_results:
            pipeline_results[split] = {}

        # Download PDBset files
        structure_file_dir_per_split.mkdir(parents=True, exist_ok=True)

        download_fn = preprocessing.download_one_pdb_structure

        logger.info(f"Downloading structures for pdb_{split}....")
        download_result = preprocessing.download_multiple_structures_fast(
            fasta_file_path, structure_file_dir_per_split, download_fn
        )

        pipeline_results[split]["download"] = download_result

        # Count x-ray derived structures
        """
        The count step below has been commented out due to how long it takes to execute. 
        You can uncomment it if you wish to verify counts yourself
        """
        # counts_result = preprocessing.count_methods_per_split(split, save_dir)
        # pipeline_results[dataset_type][split]["counts"] = counts_result

        # Remove all non-xray derived structure files from each split
        deleted, failed, missing, retained = preprocessing.delete_non_xray_structures(
            split, structure_file_dir_per_split
        )

        filter_result = {
            "deleted": deleted,
            "failed": failed,
            "missing": missing,
            "retained": retained,
        }

        pipeline_results[split]["xray_structure_filter"] = filter_result

        # Create new fasta files per split with xray-derived PDBset sequences only
        retained_xray_ids_path = (
            config.PROJECT_ROOT / f"data/retained_xray_ids_pdb_{split}.txt"
        )  # file already created by delete_non_xray_structures function

        logger.info(
            f"Non-xray sequences deleted. IDs of retained protein sequences saved to {retained_xray_ids_path}.txt"
        )

        preprocessing.filter_xray_struct(
            retained_xray_ids_path, fasta_file_path, output_file
        )

        pipeline_results[split]["xray_fasta_sequence_filter"] = {
            "input_fasta": str(fasta_file_path),
            "xray_ids": str(retained_xray_ids_path),
            "output_name": output_file,
        }


if __name__ == "__main__":
    main()
