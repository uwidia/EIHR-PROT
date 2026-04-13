#Import libraries
from pathlib import Path
import preprocessing
from parser import get_protein_info
import logging

logger = logging.getLogger(__name__)
from utils import setup_logging

setup_logging()

#Save paths for AFSet and PDBset
af_train_data_path = Path("dataset/nrAF-Model-GO_train_sequences.fasta")
af_test_data_path = Path("dataset/nrAF-Model-GO_test_sequences.fasta")
af_val_data_path = Path("dataset/nrAF-Model-GO_val_sequences.fasta")

pdb_train_data_path = Path("dataset/nrPDB-GO_2019.06.18_train_sequences.fasta")
pdb_test_data_path = Path("dataset/nrPDB-GO_2019.06.18_test_sequences.fasta")
pdb_val_data_path = Path("dataset/nrPDB-GO_2019.06.18_val_sequences.fasta")


datasets = [
    ("pdb", "train", pdb_train_data_path),
    ("pdb", "test", pdb_test_data_path),
    ("pdb", "val", pdb_val_data_path),
    ("af", "train", af_train_data_path),
    ("af", "test", af_test_data_path),
    ("af", "val", af_val_data_path),
]

pipeline_results = {}

def main():
    for dataset_type, split, fasta_path in datasets:
        if dataset_type not in pipeline_results:
            pipeline_results[dataset_type] = {}


        # pipeline_results[dataset_type][split] = {}

        # #Download AFSet and PDBset files
        # save_dir = f"structures/{dataset_type}/{dataset_type}_{split}"

        # download_fn = preprocessing.download_one_af_structure if dataset_type == "af" else preprocessing.download_one_pdb_structure


        # print(f"Downloading structures for {dataset_type}_{split}....")
        # download_result = preprocessing.download_multiple_structures_fast(
        #     fasta_path, save_dir, download_fn
        # )

        # pipeline_results[dataset_type][split]["download"] = download_result

        # Handle PDBSet-specific concerns 
        if dataset_type == "pdb":

            #Count x-ray derived structures
            """
            The count step below has been commented out due to how long it takes to execute. 
            You can uncomment it if you wish to verify counts yourself
            """

            # counts_result = preprocessing.count_methods_per_split(split)
            # pipeline_results[dataset_type][split]["counts"] = counts_result

            #Remove all non-xray derived structure files from each split
            # deleted, failed, missing, retained = preprocessing.delete_non_xray_structures(split)

            # filter_result = {
            #     "deleted": deleted,
            #     "failed": failed,
            #     "missing": missing,
            #     "retained": retained
            # }

            # pipeline_results[dataset_type][split]["xray_structure_filter"] = filter_result

            # Create new fasta files per split with xray-derived PDBset sequences only
            split_xray_ids_path = Path(f"retained_xray_ids_pdb_{split}.txt")
            split_fasta_path = Path(f"dataset/nrPDB-GO_2019.06.18_{split}_sequences.fasta")

            output_filename = f"filtered_pdb_{split}"
            

            preprocessing.filter_xray_struct(
                split_xray_ids_path,
                split_fasta_path,
                output_filename
            )

            # pipeline_results[dataset_type][split]["xray_fasta_sequence_filter"] = {
            #     "input_fasta": str(split_fasta_path),
            #     "xray_ids": str(split_xray_ids_path),
            #     "output_name": output_filename
            # }


        #Handle AFset specific concerns
        if dataset_type == "af":

            split_fasta_path = Path(f"dataset/nrAF-Model-GO_{split}_sequences.fasta")
            output_filename = f"cleaned_af_{split}"
            preprocessing.clean_af_fasta_file_header(
                split_fasta_path,
                output_filename
            )

            # pipeline_results[dataset_type][split]["clean_af_fasta_file_header"] = {
            #     "input_fasta": str(split_fasta_path),
            #     "output_name": output_filename
            # }
if __name__ == "__main__":
    main()
