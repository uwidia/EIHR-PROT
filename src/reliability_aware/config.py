from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT/ "data"
RAW_AF_TRAIN_DATA = DATA_DIR / "HEAL_dataset/nrAF-Model-GO_train_sequences.fasta"
RAW_AF_TEST_DATA = DATA_DIR / "HEAL_dataset/nrAF-Model-GO_test_sequences.fasta"
RAW_AF_VAL_DATA = DATA_DIR / "HEAL_dataset/nrAF-Model-GO_val_sequences.fasta"
RAW_PDB_TRAIN_DATA = DATA_DIR / "HEAL_dataset/nrPDB-GO_2019.06.18_train_sequences.fasta"
RAW_PDB_TEST_DATA = DATA_DIR / "HEAL_dataset/nrPDB-GO_2019.06.18_test_sequences.fasta"
RAW_PDB_VAL_DATA = DATA_DIR / "HEAL_dataset/nrPDB-GO_2019.06.18_val_sequences.fasta"
RETAINED_XRAY_IDS = DATA_DIR / "retained_xray_ids_pdb"
ALPHAFOLD_API_ENDPOINT = "https://alphafold.ebi.ac.uk/api/prediction/"
print(PROJECT_ROOT)