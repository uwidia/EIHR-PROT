import argparse
import logging
from pathlib import Path

from reliability_aware.utils.config import PROJECT_ROOT, setup_logging, diamond_directory
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


def build_config(threads: int) -> DiamondSearchConfig:
    """Keep DIAMOND search settings in one place."""
    return DiamondSearchConfig(
        evalue_max=1e-5,
        min_query_coverage=0.30,
        max_target_seqs=50,
        top_k=10,
        sensitivity="sensitive",
        iterate=True,
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

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    cfg = build_config(args.threads)

    
    cleaned_dir = PROJECT_ROOT / "data/cleaned_dataset"
    diamond_dir = diamond_directory

    train_dataset = cleaned_dir / "cleaned_pdb_train.fasta"
    val_dataset = cleaned_dir / "cleaned_pdb_val.fasta"
    test_dataset = cleaned_dir / "cleaned_pdb_test.fasta"

    diamond_dir.mkdir(parents=True, exist_ok=True)

    
    train_sequences = read_fasta_as_dict(train_dataset)
    val_sequences = read_fasta_as_dict(val_dataset)
    test_sequences = read_fasta_as_dict(test_dataset)

    # Extract split IDs from the FASTA headers while preserving dataset order.
    train_ids = [protein["full_id"] for protein in get_protein_info(train_dataset)]
    val_ids = [protein["full_id"] for protein in get_protein_info(val_dataset)]
    test_ids = [protein["full_id"] for protein in get_protein_info(test_dataset)]

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

    logger.info("DIAMOND preparation complete.")


if __name__ == "__main__":
    main()
