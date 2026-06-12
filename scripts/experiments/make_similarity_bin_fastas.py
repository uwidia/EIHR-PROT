#!/usr/bin/env python3
"""Create test FASTA files for HEAL/nrPDB sequence-similarity bins.

Default behavior is cumulative threshold mode:
- 30 -> all proteins with <30% == 1
- 40 -> all proteins with <40% == 1
- 50 -> all proteins with <50% == 1
- 70 -> all proteins with <70% == 1
- 95 -> all proteins with <95% == 1

This matches the format of nrPDB-GO_2019.06.18_test.csv, where the threshold
columns are cumulative subsets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BIN_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("30", "<30%"),
    ("40", "<40%"),
    ("50", "<50%"),
    ("70", "<70%"),
    ("95", "<95%"),
)


def parse_fasta(fasta_path: Path) -> Dict[str, str]:
    """Return {protein_id: sequence} from a FASTA file.

    The protein ID is taken as the first whitespace-separated token after `>`.
    For this dataset, headers look like `>5LA7-A`, so the ID is `5LA7-A`.
    """
    records: Dict[str, str] = {}
    current_id: str | None = None
    seq_parts: List[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(seq_parts)

                current_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)

    if current_id is not None:
        records[current_id] = "".join(seq_parts)

    return records


def read_similarity_bins(csv_path: Path) -> Dict[str, List[str]]:
    """Read the HEAL similarity CSV and return cumulative bin memberships."""
    bins: Dict[str, List[str]] = {bin_name: [] for bin_name, _ in BIN_COLUMNS}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"PDB-chain", *(column for _, column in BIN_COLUMNS)}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"CSV is missing required column(s): {sorted(missing_columns)}"
            )

        for row in reader:
            protein_id = row["PDB-chain"].strip()
            for bin_name, column in BIN_COLUMNS:
                if row[column].strip() == "1":
                    bins[bin_name].append(protein_id)

    return bins


def write_fasta(
    records: Dict[str, str], protein_ids: Iterable[str], output_path: Path
) -> int:
    """Write selected protein IDs to a FASTA file and return record count."""
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for protein_id in protein_ids:
            sequence = records[protein_id]
            handle.write(f">{protein_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
            count += 1
    return count


def create_similarity_bin_fastas(
    csv_path: Path,
    fasta_path: Path,
    output_dir: Path,
) -> Dict[str, int]:
    """Create one FASTA file per similarity threshold bin.

    Output files are named:
    - 30_similarity_test_sequence.fasta
    - 40_similarity_test_sequence.fasta
    - 50_similarity_test_sequence.fasta
    - 70_similarity_test_sequence.fasta
    - 95_similarity_test_sequence.fasta
    """
    records = parse_fasta(fasta_path)
    bins = read_similarity_bins(csv_path)

    csv_ids = {protein_id for ids in bins.values() for protein_id in ids}
    missing_ids = sorted(csv_ids - set(records))
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            f"{len(missing_ids)} protein ID(s) from the CSV were not found in the FASTA. "
            f"Examples: {preview}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    for bin_name, protein_ids in bins.items():
        output_path = output_dir / f"{bin_name}_similarity_test_sequence.fasta"
        counts[bin_name] = write_fasta(records, protein_ids, output_path)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create FASTA files for 30/40/50/70/95 similarity test bins."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/HEAL_dataset/nrPDB-GO_2019.06.18_test.csv"),
        help="Path to test dataset csv with sequence similarity binning information",
    )
    parser.add_argument(
        "--fasta-path",
        type=Path,
        default=Path("data/cleaned_dataset/cleaned_pdb_test.fasta"),
        help="Path to fasta file with protein sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cleaned_dataset/similarity_bins"),
        help="Directory where the bin FASTA files will be written",
    )
    args = parser.parse_args()

    counts = create_similarity_bin_fastas(
        csv_path=args.csv_path,
        fasta_path=args.fasta_path,
        output_dir=args.output_dir,
    )

    print(f"Saved FASTA files to: {args.output_dir}")
    for bin_name in ("30", "40", "50", "70", "95"):
        print(f"{bin_name}% similarity bin: {counts[bin_name]} sequences")


if __name__ == "__main__":
    main()
