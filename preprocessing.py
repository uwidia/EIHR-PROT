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
    session.headers.update({"User-Agent": "prot-structure-downloader/1.0"})

    return session


def _get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _make_session()
    return _thread_local.session

#Download a single cif file (AlphaFOld)
def download_one_af_structure(protein_id, save_dir, timeout):
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
def download_one_pdb_structure(pdb_id, save_dir, timeout):
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
    fasta_path,
    save_dir,
    structure_downloader,
    timeout=(5, 20),   
    max_workers=8,
    show_failures=True):
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
def count_methods_per_split(pdb_split):
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

#Delete any cif files that the method wasn't "X-RAY DIFFRACTION" from local storage
def delete_non_xray_structures(pdb_split): 
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
def filter_xray_struct(valid_xray_ids_file_path, split_fasta_path, output_filename):
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
    print(f"Filtering completed successfully. File saved to {save_path}")
    return "completed"


#Remove extra characters (nrAF-) in fasta file header. filter_xray_struct has already handled this for pdb files
def clean_af_fasta_file_header(split_fasta_path, output_filename):
    protein_entries = get_protein_info(split_fasta_path)
    records = []
    for protein in protein_entries:
        record = SeqRecord(Seq(protein["sequence"]), id=protein["full_id"], description = "")
        records.append(record)
    
    save_dir = Path("./clean_af_fasta_files")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{output_filename}.fasta"


    SeqIO.write(records, save_path, "fasta")
    print(f"Header cleaning completed successfully. File saved to {save_path}")
    return "completed"


    