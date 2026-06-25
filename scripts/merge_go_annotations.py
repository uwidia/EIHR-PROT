#!/usr/bin/env python3
"""Merge two GO annotation TSV files with identical structure."""

from __future__ import annotations

import argparse
from pathlib import Path


def merge_go_annotations(
    file1_path: Path,
    file2_path: Path,
    output_path: Path,
) -> None:
    """
    Merge two GO annotation TSV files.
    
    Assumes both files have identical header structure:
    - GO-terms and GO-names for MF, BP, CC
    - Data rows with Protein ID, GO-terms (MF, BP, CC)
    
    Args:
        file1_path: Path to first annotation file (PDB)
        file2_path: Path to second annotation file (Swiss-Model)
        output_path: Path to output merged file
    """
    print(f"Reading {file1_path.name}...")
    with open(file1_path) as f:
        lines1 = f.readlines()
    
    print(f"Reading {file2_path.name}...")
    with open(file2_path) as f:
        lines2 = f.readlines()
    
    # Find where data starts (after header sections)
    # Header ends at the line with column names (starts with ### PROTEIN-ID or PDB-chain or SWISS-MODEL-chain)
    def find_data_start(lines: list[str]) -> int:
        for i, line in enumerate(lines):
            if line.strip().startswith("###") and ("chain" in line.lower() or "protein" in line.lower()):
                return i
        raise ValueError("Could not find data section in file")
    
    header_end1 = find_data_start(lines1)
    header_end2 = find_data_start(lines2)
    
    # Extract header and data
    header = lines1[:header_end1 + 1]  # Include the header line itself
    data1 = lines1[header_end1 + 1:]
    data2 = lines2[header_end2 + 1:]
    
    # Combine data and remove duplicates (by protein ID)
    seen_proteins = set()
    merged_data = []
    
    for line in data1 + data2:
        line = line.rstrip('\n')
        if not line.strip():
            continue
        
        # Extract protein ID (first column)
        protein_id = line.split('\t')[0].strip()
        
        if protein_id not in seen_proteins:
            seen_proteins.add(protein_id)
            merged_data.append(line)
    
    # Write merged file
    print(f"Writing merged file to {output_path.name}...")
    with open(output_path, 'w') as f:
        # Write header
        for line in header:
            f.write(line)
        
        # Write merged data
        for line in merged_data:
            f.write(line + '\n')
    
    print(f"\n✓ Merge complete!")
    print(f"  File 1 ({file1_path.name}): {len(data1)} proteins")
    print(f"  File 2 ({file2_path.name}): {len(data2)} proteins")
    print(f"  Output: {len(merged_data)} unique proteins")
    print(f"  Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two GO annotation TSV files with identical structure."
    )
    parser.add_argument(
        "--file1",
        type=Path,
        default=Path("data/HEAL_dataset/nrPDB-GO_2019.06.18_annot.tsv"),
        help="First annotation file (default: nrPDB)",
    )
    parser.add_argument(
        "--file2",
        type=Path,
        default=Path("data/HEAL_dataset/nrSwiss-Model-GO_annot.tsv"),
        help="Second annotation file (default: nrSwiss-Model)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/HEAL_dataset/merged_GO_annot.tsv"),
        help="Output merged file",
    )
    
    args = parser.parse_args()
    
    merge_go_annotations(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()
