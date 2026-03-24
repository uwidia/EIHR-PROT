from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from parser import get_protein_info, get_method, get_xray_ids
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import hashlib

_thread_local = local()


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

#Download a single cif file (AlphaFOld)
def download_one_af_structure(protein_id: str, save_dir: str, timeout: tuple):
    """
    Download a protein structure CIF file from AlphaFold.

    Args:
        protein_id (str): UniProt protein identifier.
        save_dir (str): Directory to save the CIF file.
        timeout (tuple): Request timeout (connect, read).

    Returns:
        tuple: (protein_id, status, error_message)
            status ∈ {"downloaded", "skipped", "failed"}.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{protein_id}.cif"

    if file_path.exists():
        return protein_id, "skipped", None

    api_endpoint = "https://alphafold.ebi.ac.uk/api/prediction/"
    url = f"{api_endpoint}{protein_id}"
    session = _get_session()

    try:
        r = session.get(url, timeout=timeout)

        if r.status_code != 200:
            return protein_id, "failed", f"HTTP {r.status_code}"

        result = r.json()
        if not result or "cifUrl" not in result[0]:
            return protein_id, "failed", "Invalid JSON response or missing cifUrl for protein"
        cif_url = result[0]["cifUrl"]
        cif_r = session.get(cif_url, timeout = timeout)

        if cif_r.status_code != 200:
            return protein_id, "failed", f"HTTP {cif_r.status_code}"

        file_path.write_bytes(cif_r.content)
        return protein_id, "downloaded", None

    except requests.RequestException as e:
        return protein_id, "failed", str(e)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        return protein_id, "failed", f"Bad API response {e}"


#Download a single cif file (PDB)
def download_one_pdb_structure(pdb_id: str, save_dir: str, timeout: tuple):
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


#Download Multiple Cif Files Using Protein IDs (for PDBset and AFSet)
def download_multiple_structures_fast(
    fasta_path: Path,
    save_dir: str,
    structure_downloader,
    timeout: tuple =(5, 20),   
    max_workers: int =8,
    show_failures: bool =True):

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
        show_failures (bool, optional): Whether to print sample failures.

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

    print(f"Total unique structures: {len(protein_ids)}")
    print(f"Already present: {len(skipped)}")
    print(f"Need to download: {len(to_download)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(structure_downloader, protein_id, save_dir, timeout): protein_id
            for protein_id in to_download
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading CIFs"):
            protein_id, status, err = future.result()

            if status == "downloaded":
                downloaded.add(protein_id)
            else:
                failed.add(protein_id)
                failure_reasons[protein_id] = err

    print(f"\nFinished.")
    print(f"Downloaded: {len(downloaded)}")
    print(f"Skipped existing: {len(skipped)}")
    print(f"Failed: {len(failed)}")

    if failed and show_failures:
        print("\nSample failures:")
        for protein_id in list(sorted(failed))[:20]:
            print(f"  {protein_id}: {failure_reasons[protein_id]}")

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "failure_reasons": failure_reasons,
    }


#Count the number of methods for each protein entry in a split (i.e. train, test, or val)
def count_methods_per_split(pdb_split: str):
    """
    Count structure determination methods for a dataset split.

    Iterates over protein entries, reads corresponding CIF files,
    and aggregates method frequencies. Uses caching to avoid
    reprocessing duplicate entries.

    Args:
        pdb_split (str): Dataset split name ("train", "val", or "test").

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

    protein_id_path = Path(f"dataset/nrPDB-GO_2019.06.18_{pdb_split}_sequences.fasta")
    protein_entries = get_protein_info(protein_id_path)

    for protein in tqdm(protein_entries, desc = "Methods counted"):
        

        cif_path = Path(f"structures/pdb/pdb_{pdb_split}/{protein['entry_id']}.cif")
        entry_id = protein["entry_id"]

        if cif_path.exists():
            if entry_id not in cached_method_info:
                try: 
                    cached_method_info[entry_id] = get_method(cif_path)
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

def sha256_file(path: Path, chunk_size: int = 1024 * 1024):
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

def save_hash(fasta_path: Path):
     """
    Compute and append the SHA-256 hash of a file to 'hashlist.txt'.

    Args:
        fasta_path (Path): File to hash.
    """
    file_hash = sha256_file(fasta_path)
    with open("hashlist.txt", "a") as file:
        file.write(f"{file_hash}\n")

def delete_non_xray_structures(pdb_split: str): 
    """
    Remove non–X-ray diffraction CIF files for a dataset split.

    Iterates over protein IDs, deletes CIF files whose method is not
    "X-RAY DIFFRACTION", and records outcomes.

    Args:
        pdb_split (str): Dataset split ("train", "val", or "test").

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

    protein_id_path = Path(f"dataset/nrPDB-GO_2019.06.18_{pdb_split}_sequences.fasta")
    protein_entries = get_protein_info(protein_id_path)
    unique_ids = {protein["entry_id"] for protein in protein_entries}

    deleted = set()
    failed = {}
    cif_files_not_found = set()
    retained = set()
    for unique_id in tqdm(unique_ids, desc = f"Deleting non-xray structures from pdb_{pdb_split}"):

        cif_path = Path(f"structures/pdb/pdb_{pdb_split}/{unique_id}.cif")
    

        if not cif_path.exists():
            cif_files_not_found.add(unique_id)
            continue
            
        try:
            method = get_method(cif_path)
        except Exception as err:
            failed[unique_id] = f"There was an error parsing the cif file {err}"
            continue
        
        try:
            if "X-RAY DIFFRACTION" not in method:
                cif_path.unlink()
                deleted.add(unique_id)
        except Exception as err:
            failed[unique_id] = f"There was an error deleting the file {err}"
            continue
        else:
            retained.add(unique_id)
        
    output_dir = Path(f"retained_xray_ids_pdb_{pdb_split}.txt")
    with open(output_dir, "w", encoding= "utf-8") as file:
        for retained_id in retained:
            file.write(f"{retained_id}\n")
            
    print(f"{len(deleted)} files were successfully deleted.")
    print(f"{len(cif_files_not_found)} cif files weren't found in the pdb {pdb_split} directory")
    print(f"{len(failed)} cif files couldn't be processed")
    print(f"{len(retained)} xray-derived cif files left in the directory.Ids saved to {output_dir}")

    return deleted, failed, cif_files_not_found, retained


#Filter fasta file to extract x-ray protein-ids and sequences
def filter_xray_struct(valid_xray_ids_file_path: Path, split_fasta_path: Path, output_filename: str):
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
            record = SeqRecord(Seq(protein["sequence"]), id=protein["full_id"], description = "")
            records.append(record)

    save_dir = Path("./xray_filtered_pdb")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{output_filename}.fasta"


    SeqIO.write(records, save_path, "fasta")
    save_hash(save_path)
    print(f"Filtering completed successfully. File saved to {save_path}")
    return "completed"


#Remove extra characters (nrAF-) in fasta file header. filter_xray_struct has already handled this for pdb files
def clean_af_fasta_file_header(split_fasta_path: Path, output_filename: str):
    """
    Clean FASTA headers and rewrite sequences to a new file.

    Removes unwanted prefixes and preserves sequence data.

    Args:
        split_fasta_path (Path): Input FASTA file.
        output_filename (str): Name of output FASTA file (without extension).

    Returns:
        str: "completed" on success.
    """
    protein_entries = get_protein_info(split_fasta_path)
    records = []
    for protein in protein_entries:
        record = SeqRecord(Seq(protein["sequence"]), id=protein["full_id"], description = "")
        records.append(record)
    
    save_dir = Path("./clean_af_fasta_files")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{output_filename}.fasta"


    SeqIO.write(records, save_path, "fasta")
    save_hash(save_path)
    print(f"Header cleaning completed successfully. File saved to {save_path}")
    return "completed"


    """
Suggestions for later:
- Handle partial downloads
- Set up download resumption logic
- Manage broken or damaged files???
- standardize naming + return with dataclasses instead of tuples/dicts
    """