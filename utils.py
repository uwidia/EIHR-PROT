#import libraries
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm

#Parse Fasta Files and Extract Protein IDs
def get_protein_ids(file_path):
    with open(file_path, "r") as file:
        fasta_file = file.read()

    protein_ids = []
    per_protein_info = fasta_file.split(">")[1:]

    for protein in per_protein_info:
        lines = protein.splitlines()
        header = lines[0].strip()
        full_id = header.split()[0].upper()

        parts = full_id.split("-")
        entry_id = parts[0]

        chain = parts[1] if len(parts) > 1 else None

        protein_i = {
            "entry_id": entry_id, 
            "full_id" : full_id, 
            "chain": chain
            }
        protein_ids.append(protein_i)

    return protein_ids

#Parse Fasta files and Extract Protein Sequences
def get_protein_sequences(file_path):

    with open(file_path, "r") as file:
        fasta_file = file.read()

    sequences = []
    per_protein_info = fasta_file.split(">")[1:]
    for protein in per_protein_info:
        lines = protein.splitlines()
        sequence ="".join(lines[1:])
        sequences.append(sequence)
    return sequences


#Sessions stuff
_thread_local = local()


def _make_session():
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
    session.headers.update({"User-Agent": "af-downloader/1.0"})

    return session


def _get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _make_session()
    return _thread_local.session

#Download a single cif file (AlphaFOld)
def _download_one_af_structure(protein_id, save_dir, timeout):
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
        if not result or ["cifUrl"] not in result[0]:
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
def _download_one_pdb_structure(pdb_id, save_dir, timeout):
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
    fasta_path,
    save_dir,
    structure_downloader,
    timeout=(5, 20),   # (connect timeout, read timeout)
    max_workers=16,
    show_failures=True):
    protein_ids = get_protein_ids(fasta_path)

    # Deduplicate IDs and normalize case
    protein_ids = sorted({protein["entry_id"].lower() for protein in protein_ids})

    
    save_dir.mkdir(parents=True, exist_ok=True)

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