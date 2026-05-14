"""
Call with:
uv python run get_go_terms.py --go_aspect "specify go_aspect" --paths "specify path to config path file"
"""

import argparse
import yaml
from utils.go_term_extraction import (
    build_go_annotations_list,
    build_subject_go_index,
    save_subject_go_index,
    save_go_vocab,
)
from utils.parser import get_protein_info
from utils.diamond_homology import (
    DiamondSearchConfig,
    read_fasta_as_dict,
    write_fasta_from_ids,
    build_diamond_database,
    run_diamond_blastp,
    build_aligned_homology_shards,
)
from utils.config import setup_logging, PROJECT_ROOT

setup_logging()


parser = argparse.ArgumentParser()
parser.add_argument("--go_aspect", type=str, required=True)
parser.add_argument("--paths", type=str, required=True)
args = parser.parse_args()

with open(args.paths) as f:
    paths = yaml.safe_load(f)

pdb_fasta_dir = PROJECT_ROOT / paths["cleaned_pdb_dir"]
train_dataset = PROJECT_ROOT / paths["cleaned_pdb_train"]
val_dataset = PROJECT_ROOT / paths["cleaned_pdb_val"]
go_annotation_path = paths["go_annotation_path"]
obo_path = paths["obo_path"]


protein_info = get_protein_info(train_dataset)
train_ids = {protein["full_id"] for protein in protein_info}

GO_ASPECT = args.go_aspect.upper()


# Build FASTA for the training database only
train_sequences = read_fasta_as_dict(train_dataset)
val_sequences = read_fasta_as_dict(val_dataset)
train_ids = train_ids  # labels/full_ids from your training split only
train_fasta = write_fasta_from_ids(train_ids, train_sequences, "diamond/train_db.fasta")

# Build GO vocabulary + training subject annotation index
bp_label_to_go_terms, bp_go_terms = build_go_annotations_list(
    go_annotation_path=go_annotation_path,
    obo_path=obo_path,
    go_aspect="BP",
    keep_ids=train_ids,
    remove_root_term=True,
    min_term_freq=None,
)

bp_go_term_to_idx = {go: i for i, go in enumerate(bp_go_terms)}

print(len(bp_go_terms))


subject_index = build_subject_go_index(bp_label_to_go_terms, bp_go_term_to_idx)
save_subject_go_index(subject_index, "diamond/subject_go_index.json")
save_go_vocab(bp_go_terms, "diamond/go_vocab.json")

# Build the DIAMOND database from training proteins only
cfg = DiamondSearchConfig(
    evalue_max=1e-5,
    min_query_coverage=0.30,
    max_target_seqs=50,
    top_k=10,
    sensitivity="sensitive",
    iterate=True,
    threads=16,
)
build_diamond_database(train_fasta, "diamond/train_db", cfg)

# Write query FASTA for one split and run DIAMOND
val_protein_info = get_protein_info(val_dataset)
val_ids = [
    protein["full_id"] for protein in val_protein_info
]  # validation labels/full_ids
val_fasta = write_fasta_from_ids(val_ids, val_sequences, "diamond/val_queries.fasta")
run_diamond_blastp(val_fasta, "diamond/train_db", "diamond/val_hits.tsv", cfg)

# Build aligned homology shards that match the ESM manifest
build_aligned_homology_shards(
    manifest_path="esm_manifests/pdb_val_manifest.csv",
    diamond_hits="diamond/val_hits.tsv",
    subject_go_index_json_path="diamond/subject_go_index.json",
    go_vocab_json_path="diamond/go_vocab.json",
    output_dir="diamond/val_homology_shards",
    config=cfg,
    exclude_self_hits=False,
    use_fp16=True,
    keep_debug_hits=True,
)


run_diamond_blastp(val_fasta, "diamond/train_db", "diamond/train_hits.tsv", cfg)
build_aligned_homology_shards(
    manifest_path="esm_manifests/pdb_train_manifest.csv",
    diamond_hits="diamond/train_hits.tsv",
    subject_go_index_json_path="diamond/subject_go_index.json",
    go_vocab_json_path="diamond/go_vocab.json",
    output_dir="diamond/train_homology_shards",
    config=cfg,
    exclude_self_hits=True,  # Exclude self-hits to avoid trivial label leakage from exact self-matches
    use_fp16=True,
    keep_debug_hits=True,
)
