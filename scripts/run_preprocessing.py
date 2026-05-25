from utils.preprocessing import save_hash
import logging
import utils.config as config
from utils.parser import get_protein_info
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import re

logger = logging.getLogger(__name__)

config.setup_logging()


raw_dataset_dir = config.PROJECT_ROOT / "data/HEAL_dataset"
output_file_dir = config.PROJECT_ROOT / "data/cleaned_dataset"
pipeline_results = {}


def main():
    for split in ["train", "test", "val"]:
        fasta_file_path = (
            raw_dataset_dir / f"nrPDB-GO_2019.06.18_{split}_sequences.fasta"
        )

        output_file = output_file_dir / f"cleaned_pdb_{split}"

        protein_entries = get_protein_info(fasta_file_path)
        records = []
        removed_entries = 0

        # check fasta entries for missing or invalid headers and non-alphabet characters in sequence
        for protein in protein_entries:
            if not (
                protein["sequence"].isalpha()
                and re.fullmatch(r"[A-Za-z0-9-]+", protein["full_id"])
            ):
                removed_entries += 1
                continue
            record = SeqRecord(
                Seq(protein["sequence"]), id=protein["full_id"], description=""
            )
            records.append(record)

        output_file_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_file_dir / f"{output_file}.fasta"

        SeqIO.write(records, save_path, "fasta")
        save_hash(save_path)
        if removed_entries > 0:
            logger.info(
                f"""{removed_entries} entries were removed from the data due to missing headers or inclusion of invalid characters in sequence information"""
            )
        logger.info(
            f"Split:{split} | Previous count: {len(protein_entries)} | Current count: {len(records)} | Removed entries: {removed_entries}"
        )
    logger.info(f"Cleaning completed successfully. Files saved to {output_file_dir}")
    return "completed"


if __name__ == "__main__":
    main()
