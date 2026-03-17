#Import libraries
from pathlib import Path
import preprocessing

#Save paths for AFSet and PDBset
af_train_data_path = Path("dataset/nrAF-Model-GO_train_sequences.fasta")
af_test_data_path = Path("dataset/nrAF-Model-GO_test_sequences.fasta")
af_val_data_path = Path("dataset/nrAF-Model-GO_val_sequences.fasta")

pdb_train_data_path = Path("dataset/nrPDB-GO_2019.06.18_train_sequences.fasta")
pdb_test_data_path = Path("dataset/nrPDB-GO_2019.06.18_test_sequences.fasta")
pdb_val_data_path = Path("dataset/nrPDB-GO_2019.06.18_val_sequences.fasta")

#Create Directory for saving retrieved cif files 
def create_cif_file_download_path(dataset_type, split_name): 
    return Path(f"structures/{dataset_type}/{split_name}")


datasets = [
    ("pdb", "train", pdb_train_data_path, preprocessing._download_one_pdb_structure),
    ("pdb", "test", pdb_test_data_path, preprocessing._download_one_pdb_structure),
    ("pdb", "val", pdb_val_data_path, preprocessing._download_one_pdb_structure),
    ("af", "train", af_train_data_path, preprocessing._download_one_af_structure),
    ("af", "test", af_test_data_path, preprocessing._download_one_af_structure),
    ("af", "val", af_val_data_path, preprocessing._download_one_af_structure),
]

pipeline_results = {}

for dataset_type, split, fasta_path, download_fn in datasets:
    if dataset_type not in pipeline_results:
        pipeline_results[dataset_type] = {}

    pipeline_results[dataset_type][split] = {}

    #Download AFSet and PDBset files
    save_dir = create_cif_file_download_path(dataset_type, f"{dataset_type}_{split}")

    download_result = preprocessing.download_multiple_structures_fast(
        fasta_path, save_dir, download_fn
    )

    pipeline_results[dataset_type][split]["download"] = download_result

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
        deleted, failed, missing, retained = preprocessing.delete_non_xray_structures(split)

        filter_result = {
            "deleted": deleted,
            "failed": failed,
            "missing": missing,
            "retained": retained
        }

        pipeline_results[dataset_type][split]["xray_structure_filter"] = filter_result

        # Create new fasta files per split with xray-derived PDBset sequences only
        split_xray_ids_path = Path(f"retained_xray_ids_pdb_{split}.txt")
        split_fasta_path = Path(f"dataset/nrPDB-GO_2019.06.18_{split}_sequences.fasta")

        fasta_output_name = f"filtered_{split}"

        preprocessing.filter_xray_struct(
            split_xray_ids_path,
            split_fasta_path,
            fasta_output_name
        )

        pipeline_results[dataset_type][split]["xray_fasta_sequence_filter"] = {
            "input_fasta": str(split_fasta_path),
            "xray_ids": str(split_xray_ids_path),
            "output_name": fasta_output_name
        }


