"""Generate original nine-column hits; use fusion_baselines.py enrich for nident."""

import argparse
import logging
from pathlib import Path
import sys
PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from reliability_aware.utils.config import (
    PROJECT_ROOT,
    setup_logging,
    diamond_directory,
)
from reliability_aware.utils.diamond_homology import (
    DiamondSearchConfig,
    build_diamond_database,
    read_fasta_as_dict,
    run_diamond_blastp,
    write_fasta_from_ids,
)
from reliability_aware.utils.parser import get_protein_info

logger = logging.getLogger(__name__)


def file_exists(path: Path) -> bool:
    """Return True only when the file exists and is not empty."""
    return path.exists() and path.stat().st_size > 0


def build_config(threads: int, *, iterate: bool = True) -> DiamondSearchConfig:
    """Keep DIAMOND search settings in one place."""
    return DiamondSearchConfig(
        evalue_max=1e-5,
        min_query_coverage=0.30,
        max_target_seqs=50,
        top_k=10,
        sensitivity="sensitive",
        iterate=iterate,
        threads=threads,
    )


def maybe_build_database(
    train_fasta: Path,
    db_prefix: Path,
    cfg: DiamondSearchConfig,
    *,
    force: bool,
) -> None:
    """Build the DIAMOND database unless it already exists."""
    db_path = db_prefix.with_suffix(".dmnd")

    # The database is shared across BP/MF/CC because it only stores sequences.
    if file_exists(db_path) and not force:
        logger.info("Skipping existing DIAMOND database: %s", db_path)
        return

    build_diamond_database(train_fasta, db_prefix, cfg)


def maybe_run_search(
    query_fasta: Path,
    db_prefix: Path,
    output_tsv: Path,
    cfg: DiamondSearchConfig,
    *,
    force: bool,
) -> None:
    """Run DIAMOND blastp unless the requested hit file already exists."""
    if file_exists(output_tsv) and not force:
        logger.info("Skipping existing DIAMOND hits: %s", output_tsv)
        return

    run_diamond_blastp(query_fasta, db_prefix, output_tsv, cfg)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate DIAMOND database and hit files even if they already exist.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of CPU threads to pass to DIAMOND.",
    )
    parser.add_argument(
        "--no-iterate",
        action="store_true",
        help="Disable iterative search. Use a new output directory and compare retained evidence before reusing priors.",
    )
    parser.add_argument(
        "--train_dataset",
        type=Path,
        default=PROJECT_ROOT / "data/cleaned_dataset" / "cleaned_pdb_train.fasta",
        help="Path to the training FASTA used to build the DIAMOND database.",
    )
    parser.add_argument(
        "--val_dataset",
        type=Path,
        default=PROJECT_ROOT / "data/cleaned_dataset" / "cleaned_pdb_val.fasta",
        help="Path to the validation FASTA used to generate validation search queries.",
    )
    parser.add_argument("--af_test_dataset", type=Path, default=PROJECT_ROOT / "data/cleaned_dataset" / "cleaned_af_test.fasta", help="AF test FASTA; queried against the PDB-training DB only.")
    parser.add_argument("--output_dir", type=Path, default=diamond_directory, help="Original nine-column hit directory (default: diamond_db). For enriched hits use fusion_baselines.py enrich.")
    parser.add_argument(
        "--test_dataset",
        type=Path,
        default=PROJECT_ROOT / "data/cleaned_dataset" / "cleaned_pdb_test.fasta",
        help="Path to the test FASTA used to generate test search queries.",
    )
    args = parser.parse_args()

    cfg = build_config(args.threads, iterate=not args.no_iterate)

    cleaned_dir = PROJECT_ROOT / "data/cleaned_dataset"
    diamond_dir = args.output_dir

    train_dataset = args.train_dataset
    val_dataset = args.val_dataset
    test_dataset = args.test_dataset
    af_test_dataset = args.af_test_dataset

    diamond_dir.mkdir(parents=True, exist_ok=True)

    train_sequences = read_fasta_as_dict(train_dataset)
    val_sequences = read_fasta_as_dict(val_dataset)
    test_sequences = read_fasta_as_dict(test_dataset)
    af_test_sequences = read_fasta_as_dict(af_test_dataset)

    # Extract split IDs from the FASTA headers while preserving dataset order.
    train_ids = [protein["full_id"] for protein in get_protein_info(train_dataset)]
    val_ids = [protein["full_id"] for protein in get_protein_info(val_dataset)]
    test_ids = [protein["full_id"] for protein in get_protein_info(test_dataset)]
    af_test_ids = [protein["full_id"] for protein in get_protein_info(af_test_dataset)]

    # Write split-specific FASTA files used by DIAMOND.
    # The training FASTA is used both as the database source and as training queries.
    train_fasta = write_fasta_from_ids(
        train_ids,
        train_sequences,
        diamond_dir / "train_db.fasta",
    )
    val_fasta = write_fasta_from_ids(
        val_ids,
        val_sequences,
        diamond_dir / "val_queries.fasta",
    )
    test_fasta = write_fasta_from_ids(
        test_ids,
        test_sequences,
        diamond_dir / "test_queries.fasta",
    )

    af_test_fasta = write_fasta_from_ids(af_test_ids, af_test_sequences, diamond_dir / "af_test_queries.fasta")

    db_prefix = diamond_dir / "train_db"

    # Build one training-only database. Validation/test must search against training only.
    maybe_build_database(train_fasta, db_prefix, cfg, force=args.force)

    # These raw DIAMOND hit files are shared across BP/MF/CC.
    # GO-aspect-specific processing happens later in build_homology_shards.py.
    maybe_run_search(
        train_fasta,
        db_prefix,
        diamond_dir / "train_hits.tsv",
        cfg,
        force=args.force,
    )
    maybe_run_search(
        val_fasta,
        db_prefix,
        diamond_dir / "val_hits.tsv",
        cfg,
        force=args.force,
    )
    maybe_run_search(
        test_fasta,
        db_prefix,
        diamond_dir / "test_hits.tsv",
        cfg,
        force=args.force,
    )

    maybe_run_search(af_test_fasta, db_prefix, diamond_dir / "af_test_hits.tsv", cfg, force=args.force)
    logger.info("DIAMOND preparation complete.")


if __name__ == "__main__":
    main()
