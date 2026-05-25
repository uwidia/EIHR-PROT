from pathlib import Path
import hashlib
import logging
import reliability_aware.utils.config as config

logger = logging.getLogger(__name__)


def save_hash(
    fasta_path: Path, hashlist_file: Path = config.PROJECT_ROOT / "hashlist.txt"
):
    """
    Compute and append the SHA-256 hash of a file to 'hashlist.txt'.

    Args:
        fasta_path (Path): File to hash.
        hashlist_file (Path): File containing list of valid hashes for datasets
    """
    file_hash = sha256_file(fasta_path)
    with open(hashlist_file, "a") as file:
        file.write(f"{file_hash}\n")


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
