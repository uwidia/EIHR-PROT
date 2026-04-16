from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT/ "data"

RAW_AF_TRAIN_DATA = DATA_DIR / "HEAL_dataset/nrAF-Model-GO_train_sequences.fasta"
RAW_AF_TEST_DATA = DATA_DIR / "HEAL_dataset/nrAF-Model-GO_test_sequences.fasta"
RAW_AF_VAL_DATA = DATA_DIR / "HEAL_dataset/nrAF-Model-GO_val_sequences.fasta"

RAW_PDB_TRAIN_DATA = DATA_DIR / "HEAL_dataset/nrPDB-GO_2019.06.18_train_sequences.fasta"
RAW_PDB_TEST_DATA = DATA_DIR / "HEAL_dataset/nrPDB-GO_2019.06.18_test_sequences.fasta"
RAW_PDB_VAL_DATA = DATA_DIR / "HEAL_dataset/nrPDB-GO_2019.06.18_val_sequences.fasta"

CLEAN_AF_TRAIN_DATA = DATA_DIR / "cleaned_dataset/af/cleaned_af_train.fasta"
CLEAN_AF_TEST_DATA = DATA_DIR / "cleaned_dataset/af/cleaned_af_test.fasta"
CLEAN_AF_VAL_DATA = DATA_DIR / "cleaned_dataset/af/cleaned_af_val.fasta"

RETAINED_XRAY_IDS = DATA_DIR / "retained_xray_ids_pdb"

def setup_logging():
    logger = logging.getLogger()  
    
    
    if logger.handlers:
        return logger
    
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="a")
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        handlers=handlers
    )
    
    return logger