"""
Build GO-aspect-specific homology shards from shared DIAMOND hits.

Run once per GO aspect:

    uv run python scripts/build_homology_shards.py --go_aspect BP
    uv run python scripts/build_homology_shards.py --go_aspect MF
    uv run python scripts/build_homology_shards.py --go_aspect CC

Force regeneration of existing shard outputs:

    uv run python scripts/build_homology_shards.py --go_aspect BP --force
"""

import argparse
import logging
from pathlib import Path

from reliability_aware.utils.config import PROJECT_ROOT, setup_logging, diamond_directory
from reliability_aware.utils.diamond_homology import DiamondSearchConfig, build_aligned_homology_shards
from reliability_aware.utils.go_term_extraction import (
    build_go_annotations_list,
    build_subject_go_index,
    save_go_vocab,
    save_subject_go_index,
)
from reliability_aware.utils.parser import get_protein_info

logger = logging.getLogger(__name__)


VALID_GO_ASPECTS = {"BP", "MF", "CC"}


def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def build_config(threads: int) -> DiamondSearchConfig:
    """
    Build the homology filtering config.

    Build_aligned_homology_shards re-applies E-value, coverage, and top-k filtering.
    """
    return DiamondSearchConfig(
        evalue_max=1e-5,
        min_query_coverage=0.30,
        max_target_seqs=50,
        top_k=10,
        sensitivity="sensitive",
        iterate=True,
        threads=threads,
    )


def require_file(path: Path, message: str) -> None:
    if not file_exists(path):
        raise FileNotFoundError(f"{message}: {path}")


def maybe_build_split_shards(
    *,
    split: str,
    manifest_path: Path,
    diamond_hits: Path,
    subject_go_index_path: Path,
    go_vocab_path: Path,
    output_dir: Path,
    cfg: DiamondSearchConfig,
    exclude_self_hits: bool,
    force: bool,
) -> None:
    """Build homology shards for one dataset split unless they already exist."""
    metadata_path = output_dir / "homology_shard_metadata.json"

  
    if file_exists(metadata_path) and not force:
        logger.info("Skipping existing %s homology shards: %s", split, output_dir)
        return

    
    build_aligned_homology_shards(
        manifest_path=manifest_path,
        diamond_hits=diamond_hits,
        subject_go_index_json_path=subject_go_index_path,
        go_vocab_json_path=go_vocab_path,
        output_dir=output_dir,
        config=cfg,
        exclude_self_hits=exclude_self_hits,
        use_fp16=True,
        keep_debug_hits=True,
    )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--go_aspect",
        type=str,
        required=True,
        choices=sorted(VALID_GO_ASPECTS),
        help="GO aspect to process: BP, MF, or CC.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate GO vocab, subject index, and homology shards.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Thread count stored in the shared DIAMOND config.",
    )
    args = parser.parse_args()

    go_aspect = args.go_aspect.upper()
    cfg = build_config(args.threads)

    cleaned_dir = PROJECT_ROOT / "data/cleaned_dataset"
    heal_dir = PROJECT_ROOT / "data/HEAL_dataset"
    manifest_dir = PROJECT_ROOT / "esm_embeddings"
    diamond_dir = diamond_directory
    aspect_dir = diamond_dir / go_aspect

    train_dataset = cleaned_dir / "cleaned_pdb_train.fasta"
    go_annotation_path = heal_dir / "nrPDB-GO_2019.06.18_annot.tsv"
    obo_path = heal_dir / "go-basic.obo"

    train_hits = diamond_dir / "train_hits.tsv"
    val_hits = diamond_dir / "val_hits.tsv"
    test_hits = diamond_dir / "test_hits.tsv"

    # These files should already exist from prepare_diamond_hits.py.
    require_file(
        train_hits, "Missing train DIAMOND hits. Run prepare_diamond_hits.py first"
    )
    require_file(
        val_hits, "Missing validation DIAMOND hits. Run prepare_diamond_hits.py first"
    )
    require_file(
        test_hits, "Missing test DIAMOND hits. Run prepare_diamond_hits.py first"
    )

    aspect_dir.mkdir(parents=True, exist_ok=True)

    # Build the vocabulary using training annotations only to avoid val/test label leakage.
    train_ids = [protein["full_id"] for protein in get_protein_info(train_dataset)]

    label_to_go_terms, go_terms = build_go_annotations_list(
        go_annotation_path=go_annotation_path,
        obo_path=obo_path,
        go_aspect=go_aspect,
        keep_ids=train_ids,
        remove_root_term=True,
        min_term_freq=None,
    )

    # Map each GO term to a model output index.
    go_term_to_idx = {go: i for i, go in enumerate(go_terms)}
    logger.info("Built %s GO vocabulary with %d terms.", go_aspect, len(go_terms))

    # Save aspect-specific files; BP/MF/CC must not overwrite each other.
    subject_go_index = build_subject_go_index(label_to_go_terms, go_term_to_idx)
    subject_go_index_path = aspect_dir / "subject_go_index.json"
    go_vocab_path = aspect_dir / "go_vocab.json"

    save_subject_go_index(subject_go_index, subject_go_index_path)
    save_go_vocab(go_terms, go_vocab_path)

    # Convert shared sequence hits into GO-aspect-specific prior vectors.
    maybe_build_split_shards(
        split="train",
        manifest_path=manifest_dir / "train/pdb_train_manifest.csv",
        diamond_hits=train_hits,
        subject_go_index_path=subject_go_index_path,
        go_vocab_path=go_vocab_path,
        output_dir=aspect_dir / "train_homology_shards",
        cfg=cfg,
        exclude_self_hits=True,
        force=args.force,
    )
    maybe_build_split_shards(
        split="val",
        manifest_path=manifest_dir / "val/pdb_val_manifest.csv",
        diamond_hits=val_hits,
        subject_go_index_path=subject_go_index_path,
        go_vocab_path=go_vocab_path,
        output_dir=aspect_dir / "val_homology_shards",
        cfg=cfg,
        exclude_self_hits=False,
        force=args.force,
    )
    maybe_build_split_shards(
        split="test",
        manifest_path=manifest_dir / "test/pdb_test_manifest.csv",
        diamond_hits=test_hits,
        subject_go_index_path=subject_go_index_path,
        go_vocab_path=go_vocab_path,
        output_dir=aspect_dir / "test_homology_shards",
        cfg=cfg,
        exclude_self_hits=False,
        force=args.force,
    )

    logger.info("Finished building %s homology shards.", go_aspect)


if __name__ == "__main__":
    main()
