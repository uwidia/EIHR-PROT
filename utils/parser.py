import gemmi
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_protein_info(file_path: Path):
    """
    Retrieves protein information - ids and sequence - for each protein in a fasta file.
    Return:
        List of dictionaries with id and sequence information per protein entry
    """
    with open(file_path, "r") as file:
        fasta_file = file.read()

    protein_info = []
    per_protein_info = fasta_file.split(">")[1:]

    for protein in per_protein_info:
        lines = protein.splitlines()
        header = lines[0].strip()
        sequence = "".join(lines[1:])
        full_id = header.split()[0]

        parts = full_id.split("-")
        entry_id = parts[0]

        chain = parts[1] if len(parts) > 1 else None

        protein_i = {
            "entry_id": entry_id,
            "full_id": full_id,
            "chain": chain,
            "sequence": sequence,
        }
        protein_info.append(protein_i)

    return protein_info


# Parse cif file and retreive method
def get_method(cif_path: Path):
    """
    Identify method used for protein structure determination in cif file
    Args:
        cif_path(Path): Path to protein structure (.cif) file
    Returns:
        String with single method or multiple methods concatenated with ";"
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()
    methods = block.find_values("_exptl.method")
    method = "; ".join(methods) if methods else "UNKNOWN"

    return method


# Parse retained_ids for xray-derived pdb structures
def get_xray_ids(file_path: Path):
    """
    Creates set containing xray-derived-protein-structure ids from text file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        retained_ids_txt = file.read()
        xray_ids = retained_ids_txt.splitlines()
        xray_ids = set(xray_ids)
    return xray_ids


def get_dataset_hashes(file_path: Path):
    """
    Retrieves hashes from text file with valid hashes for preprocessed datasets.
    Returns:
        List containing valid hashes
    """
    if not file_path.exists():
        logger.warning(f"Hash file not found: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as file:
        hashlist_txt = file.read()
        hashlist = [line.strip() for line in hashlist_txt.splitlines() if line.strip()]
    return hashlist
