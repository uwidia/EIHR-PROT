from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from utils.parser import get_protein_info, get_method, get_xray_ids
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import hashlib
import logging
import utils.config as config

logger = logging.getLogger(__name__)

_thread_local = local()

raw_pdb_paths = {
    "train": config.raw_pdb_train,
    "test": config.raw_pdb_test,
    "val": config.raw_pdb_val,
}


def _make_session():
    """
    Create a configured requests session with retry and connection pooling.

    Returns:
        requests.Session: Session with retry policy and default headers.
    """
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=100,
        pool_maxsize=100,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "prot-structure-downloader/1.0"})

    return session


def _get_session():
    """
    Get a thread-local HTTP session, creating one if needed.

    Returns:
        requests.Session: Per-thread cached session instance.
    """
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _make_session()
    return _thread_local.session


# Download Multiple Cif Files Using Protein IDs (for PDBset and AFSet)
def download_multiple_structures_fast(
    fasta_path: Path,
    save_dir: str | Path,
    structure_downloader,
    timeout: tuple = (5, 20),
    max_workers: int = 8,
    log_failures: bool = True,
):
    """
    Download multiple protein structure CIF files in parallel.

    Reads protein IDs from a FASTA file, skips already-downloaded files,
    and uses a provided downloader function to fetch missing structures.

    Args:
        fasta_path (Path): Path to FASTA file containing protein entries.
        save_dir (str): Directory to save downloaded CIF files.
        structure_downloader (callable): Function to download a single structure.
            Must return (protein_id, status, error_message).
        timeout (tuple, optional): Request timeout (connect, read).
        max_workers (int, optional): Number of parallel download threads.
        log_failures (bool, optional): Whether to save sample failures in log.

    Returns:
        dict: Summary with keys:
            - "downloaded": set of successfully downloaded IDs
            - "skipped": set of already existing IDs
            - "failed": set of failed IDs
            - "failure_reasons": dict mapping ID → error message
    """

    protein_ids = get_protein_info(fasta_path)

    # Deduplicate IDs and normalize case
    protein_ids = sorted({protein["entry_id"] for protein in protein_ids})
    save_dir = Path(save_dir)

    downloaded = set()
    skipped = set()
    failed = set()
    failure_reasons = {}

    # Only submit jobs for files that do not already exist
    to_download = []
    for protein_id in protein_ids:
        file_path = save_dir / f"{protein_id}.cif"
        if file_path.exists():
            skipped.add(protein_id)
        else:
            to_download.append(protein_id)

    logger.info(
        f"Total unique structures: {len(protein_ids)} | Already present: {len(skipped)} | Need to download: {len(to_download)}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                structure_downloader, protein_id, save_dir, timeout
            ): protein_id
            for protein_id in to_download
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading CIFs"
        ):
            protein_id, status, err = future.result()

            if status == "downloaded":
                downloaded.add(protein_id)
            else:
                failed.add(protein_id)
                failure_reasons[protein_id] = err

    logger.info(
        f"\nDownloaded: {len(downloaded)} | Skipped existing: {len(skipped)} | Failed: {len(failed)} "
    )

    if failed and log_failures:
        for protein_id in list(sorted(failed))[:20]:  # logs first 20 failures only
            logger.error(f"{protein_id}: {failure_reasons[protein_id]}")

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "failure_reasons": failure_reasons,
    }


# Download a single cif file (PDB)
def download_one_pdb_structure(pdb_id: str, save_dir: str | Path, timeout: tuple):
    """
    Download a protein structure CIF file from the RCSB PDB.

    Args:
        pdb_id (str): PDB identifier.
        save_dir (str): Directory to save the CIF file.
        timeout (tuple): Request timeout (connect, read).

    Returns:
        tuple: (pdb_id, status, error_message)
            status ∈ {"downloaded", "skipped", "failed"}.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{pdb_id}.cif"

    if file_path.exists():
        return pdb_id, "skipped", None

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    session = _get_session()

    try:
        r = session.get(url, timeout=timeout)

        if r.status_code == 200:
            file_path.write_bytes(r.content)
            return pdb_id, "downloaded", None

        return pdb_id, "failed", f"HTTP {r.status_code}"

    except requests.RequestException as e:
        return pdb_id, "failed", str(e)


# Count the number of methods for each protein entry in a split (i.e. train, test, or val)
def count_methods_per_split(pdb_split: str, structure_dir: Path):
    """
    Count structure determination methods for a dataset split.

    Iterates over protein entries, reads corresponding CIF files,
    and aggregates method frequencies. Uses caching to avoid
    reprocessing duplicate entries.

    Args:
        pdb_split (str): Dataset split name ("train", "val", or "test").
        structure_dir (Path): Path to directory containing structure files (.cif)

    Returns:
        tuple:
            - dict: method → count
            - set: skipped protein IDs (missing CIF files)
            - dict: entry_id → error message for failed parses

    Raises:
        ValueError: If pdb_split is not one of {"train", "val", "test"}.
    """
    cached_method_info = {}
    skipped = set()
    errors = {}
    accepted_values = {"train", "val", "test"}
    if pdb_split not in accepted_values:
        raise ValueError("pdb_split should be one of: train, val, or test")

    methods_freq_count = defaultdict(int)
    protein_id_path = raw_pdb_paths[pdb_split]

    protein_entries = get_protein_info(protein_id_path)

    for protein in tqdm(protein_entries, desc="Methods counted"):

        structure_file = structure_dir / f"{protein['entry_id']}.cif"
        entry_id = protein["entry_id"]

        if structure_file.exists():
            if entry_id not in cached_method_info:
                try:
                    cached_method_info[entry_id] = get_method(structure_file)
                except Exception as err:
                    errors[entry_id] = f"There was an error with {entry_id}: {err}"
                    continue
                method = cached_method_info[entry_id]
                methods_freq_count[method] += 1
                continue
            else:
                method = cached_method_info[entry_id]
                methods_freq_count[method] += 1

        else:
            skipped.add(entry_id)

    return methods_freq_count, skipped, errors


def delete_non_xray_structures(pdb_split: str, structure_dir: Path):
    """
    Remove non–X-ray diffraction CIF files for a dataset split.

    Iterates over protein IDs, deletes CIF files whose method is not
    "X-RAY DIFFRACTION", and records outcomes.

    Args:
        pdb_split (str): Dataset split ("train", "val", or "test").
        structure_dir (Path): Path to the directory containing structure files (.cif)

    Returns:
        tuple:
            - set: deleted IDs
            - dict: ID → error message for failures
            - set: IDs with missing CIF files
            - set: retained (X-ray) IDs

    Raises:
        ValueError: If pdb_split is invalid.
    """
    accepted_values = ["train", "val", "test"]
    if pdb_split not in accepted_values:
        raise ValueError("pdb_split should be one of: train, val, or test")

    protein_id_path = raw_pdb_paths[pdb_split]
    protein_entries = get_protein_info(protein_id_path)
    unique_ids = {protein["entry_id"] for protein in protein_entries}

    deleted = set()
    failed = {}
    structure_files_not_found = set()
    retained = set()
    for unique_id in tqdm(
        unique_ids, desc=f"Deleting non-xray structures from pdb_{pdb_split}"
    ):
        structure_file = structure_dir / f"{unique_id}.cif"

        if not structure_file.exists():
            structure_files_not_found.add(unique_id)
            continue

        try:
            method = get_method(structure_file)
        except Exception as err:
            failed[unique_id] = f"There was an error parsing the cif file {err}"
            continue

        try:
            if "X-RAY DIFFRACTION" not in method:
                structure_file.unlink()
                deleted.add(unique_id)
        except Exception as err:
            failed[unique_id] = f"There was an error deleting the file {err}"
            continue
        else:
            retained.add(unique_id)

    output_file = config.PROJECT_ROOT / f"data/retained_xray_ids_pdb_{pdb_split}.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        for retained_id in retained:
            file.write(f"{retained_id}\n")

    logger.info(
        f"""Deleted files: {len(deleted)} | Missing structure files: {len(structure_files_not_found)} 
    | Failed to process: {len(failed)} | Retained files: {len(retained)} | Output File: {output_file}"""
    )

    return deleted, failed, structure_files_not_found, retained


# Filter fasta file to extract x-ray protein-ids and sequences
def filter_xray_struct(
    valid_xray_ids_file_path: Path, split_fasta_path: Path, output_filename: str | Path
):
    """
    Filter FASTA entries to include only proteins with valid X-ray structures.

    Writes a new FASTA file and stores its hash.

    Args:
        valid_xray_ids_file_path (Path): File containing valid X-ray IDs.
        split_fasta_path (Path): Input FASTA file.
        output_filename (str): Name of output FASTA file (without extension).

    Returns:
        str: "completed" on success.
    """
    protein_entries = get_protein_info(split_fasta_path)
    records = []
    valid_xray_ids = get_xray_ids(valid_xray_ids_file_path)
    for protein in protein_entries:
        if protein["entry_id"] in valid_xray_ids:
            if not (protein["sequence"].isalpha() and protein["full_id"]):
                continue
            record = SeqRecord(
                Seq(protein["sequence"]), id=protein["full_id"], description=""
            )
            records.append(record)

    save_dir = config.PROJECT_ROOT / "data/cleaned_dataset"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{output_filename}.fasta"

    SeqIO.write(records, save_path, "fasta")
    save_hash(save_path)
    logger.info(f"Filtering completed successfully. File saved to {save_path}")
    return "completed"


def save_hash(
    fasta_path: Path, hashlist_file: Path = config.PROJECT_ROOT / "hashlist.txt"
):
    """
    Compute and append the SHA-256 hash of a file to 'hashlist.txt'.

    Args:
        fasta_path (Path): File to hash.
        hashlist_file (Path): File containing list of valid hashes for datasets
    """
    file_hash = _sha256_file(fasta_path)
    with open(hashlist_file, "a") as file:
        file.write(f"{file_hash}\n")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024):
    """
    Compute SHA-256 hash of a file.

    Args:
        path (Path): File to hash.
        chunk_size (int, optional): Size of chunks to read.

    Returns:
        str: Hex digest of the file.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
