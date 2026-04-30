from pathlib import Path
from reliability_aware.config import DATA_DIR
from reliability_aware.go_term_extraction import ( 
    build_go_annotations_list, 
    build_subject_go_index, 
    save_subject_go_index, 
    save_go_vocab
    )
from reliability_aware.parser import get_protein_info
from reliability_aware.diamond_homology import (
    DiamondSearchConfig,
    read_fasta_as_dict,
    write_fasta_from_ids,
    build_diamond_database,
    run_diamond_blastp,
    build_aligned_homology_shards,
)

pdb_fasta_dir = DATA_DIR / "cleaned_dataset/pdb"

train = pdb_fasta_dir / "cleaned_pdb_train.fasta"
val = pdb_fasta_dir / "cleaned_pdb_val.fasta"

protein_info = get_protein_info(train)

tsv_path = DATA_DIR / "HEAL_dataset/nrPDB-GO_2019.06.18_annot.tsv"
obo_path = DATA_DIR / "HEAL_dataset/go-basic.obo"

train_ids = {
   protein["full_id"] for protein in protein_info
}
GO_ASPECT = "CC"




# Build FASTA for the training database only
train_sequences = read_fasta_as_dict(train)
val_sequences = read_fasta_as_dict(val)
train_ids = train_ids  # labels/full_ids from your training split only
train_fasta = write_fasta_from_ids(train_ids, train_sequences, "diamond/train_db.fasta")

# Build GO vocabulary + training subject annotation index
bp_label_to_go_terms, bp_go_terms = build_go_annotations_list(
    tsv_path=tsv_path,
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
val_protein_info = get_protein_info(val)
val_ids = [protein["full_id"] for protein in val_protein_info ]  # validation labels/full_ids
val_fasta = write_fasta_from_ids(val_ids, val_sequences, "diamond/val_queries.fasta")
run_diamond_blastp(val_fasta, "diamond/train_db", "diamond/val_hits.tsv", cfg)

# Build aligned homology shards that match the ESM manifest
build_aligned_homology_shards(
    manifest_path="esm_manifests/pdb_val_manifest.csv",
    diamond_tsv_path="diamond/val_hits.tsv",
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
    diamond_tsv_path="diamond/train_hits.tsv",
    subject_go_index_json_path="diamond/subject_go_index.json",
    go_vocab_json_path="diamond/go_vocab.json",
    output_dir="diamond/train_homology_shards",
    config=cfg,
    exclude_self_hits=True, # Exclude self-hits to avoid trivial label leakage from exact self-matches
    use_fp16=True,
    keep_debug_hits=True,
)
