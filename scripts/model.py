#SCRIPT FOR muLtimodal version
import torch
from torch.utils.data import DataLoader
from reliability_aware.parser import get_protein_info
from reliability_aware.go_term_extraction import build_subject_go_index, build_go_annotations_list, build_child_parent_idx_pairs
from reliability_aware.losses import compute_pos_weight_from_label_indices, run_one_batch_smoke_test
import reliability_aware.reliability_aware_model as ra_model
from reliability_aware.shard_handling import ESMGraphHomologyShardDataset
from reliability_aware.pool_embeddings import GATBranch, ESMSequenceBranch
from reliability_aware.config import DATA_DIR, PROJECT_ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GO_ASPECT = "BP"

TSV_PATH = DATA_DIR / "HEAL_dataset/nrPDB-GO_2019.06.18_annot.tsv"
OBO_PATH = DATA_DIR / "HEAL_dataset/go-basic.obo"
PDB_FASTA_DIR = DATA_DIR / "cleaned_dataset/pdb"

#training dataset paths
TRAIN_GRAPH_SHARD_DIR = PROJECT_ROOT / "graph_shards/train"
TRAIN_HOMOLOGY_SHARD_DIR = PROJECT_ROOT / "diamond/train_homology_shards"
TRAIN_ESM_SHARD_DIR = PROJECT_ROOT / "esm_embeddings/pdb/pdb_train" 
TRAIN_MANIFEST_PATH = PROJECT_ROOT / "esm_manifests/pdb_train_manifest.csv"
TRAIN_DATA = PDB_FASTA_DIR / "cleaned_pdb_train.fasta"

#validation dataset paths
VAL_GRAPH_SHARD_DIR = PROJECT_ROOT / "graph_shards/val"
VAL_HOMOLOGY_SHARD_DIR = PROJECT_ROOT / "diamond/val_homology_shards"
VAL_ESM_SHARD_DIR = PROJECT_ROOT / "esm_embeddings/pdb/pdb_val"
VAL_MANIFEST_PATH = PROJECT_ROOT / "esm_manifests/pdb_val_manifest.csv"
VAL_DATA = PDB_FASTA_DIR / "cleaned_pdb_val.fasta"



train_protein_info = get_protein_info(TRAIN_DATA)
val_protein_info = get_protein_info(VAL_DATA)

train_ids = {
   protein["full_id"] for protein in train_protein_info
}

val_ids = {
    protein["full_id"] for protein in val_protein_info
}

train_label_to_go_terms, go_terms = build_go_annotations_list(
    tsv_path=TSV_PATH,
    obo_path=OBO_PATH,
    go_aspect=GO_ASPECT,
    keep_ids=train_ids,
    remove_root_term=True,
    min_term_freq=None,
)

val_label_to_go_terms, _ = build_go_annotations_list(
    tsv_path=TSV_PATH,
    obo_path=OBO_PATH,
    go_aspect=GO_ASPECT,
    keep_ids=val_ids,
    remove_root_term=True,
    min_term_freq=None,
)


child_parent_pairs = build_child_parent_idx_pairs(
    obo_path=OBO_PATH,
    go_terms=go_terms,
)


go_term_to_idx = {go: i for i, go in enumerate(go_terms)}

train_label_to_indices = build_subject_go_index(
    train_label_to_go_terms,
    go_term_to_idx,
)

val_label_to_indices = build_subject_go_index(
    val_label_to_go_terms,
    go_term_to_idx,
)


train_keep_ids_for_aspect = {
    label for label, idxs in train_label_to_indices.items()
    if len(idxs) > 0
}

val_keep_ids_for_aspect = {
    label for label, idxs in val_label_to_indices.items()
    if len(idxs) > 0
}

pos_weight = compute_pos_weight_from_label_indices(
    label_to_indices=train_label_to_indices,
    num_go_terms=len(go_terms),
    train_ids=train_keep_ids_for_aspect,
    cap=20.0,
)


train_dataset = ESMGraphHomologyShardDataset(
    esm_shard_dir=TRAIN_ESM_SHARD_DIR,
    graph_shard_dir=TRAIN_GRAPH_SHARD_DIR,
    homology_shard_dir=TRAIN_HOMOLOGY_SHARD_DIR,
    manifest_path=TRAIN_MANIFEST_PATH,
    require_graph=True,
    keep_ids=train_keep_ids_for_aspect,
)

val_dataset = ESMGraphHomologyShardDataset(
    esm_shard_dir=VAL_ESM_SHARD_DIR,
    graph_shard_dir=VAL_GRAPH_SHARD_DIR,
    homology_shard_dir=VAL_HOMOLOGY_SHARD_DIR,
    manifest_path=VAL_MANIFEST_PATH,
    require_graph=True,
    keep_ids=val_keep_ids_for_aspect,
)

train_batch_sampler = ra_model.HybridBatchSampler(
    dataset=train_dataset,
    batch_size=16,
    active_shards=3,
    lookahead_factor=2,
    drop_last=True,
    seed=42,
)

val_batch_sampler = ra_model.HybridBatchSampler(
    dataset=val_dataset,
    batch_size=16,
    active_shards=3,
    lookahead_factor=2,
    drop_last=True,
    seed=42,
)


train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=ra_model.multimodal_collate_fn_generator(label_to_indices=train_label_to_indices, num_go_terms=len(go_terms)),
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_batch_sampler,
    collate_fn=ra_model.multimodal_collate_fn_generator(label_to_indices=val_label_to_indices, num_go_terms=len(go_terms)),
    num_workers=0,
    pin_memory=True,
)


seq_branch = ESMSequenceBranch()

gat_branch = GATBranch(
    esm_dim=1280,
    hidden_dim=256,
    heads=4,
    dropout=0.1,
    edge_dim=5,
    out_dim=1280,
    use_confidence_as_node_feature=True,
)

num_go_terms = len(go_terms) 

model = ra_model.ReliabilityAwareProteinFunctionModel(
    seq_branch=seq_branch,
    gat_branch=gat_branch,
    num_go_terms=num_go_terms,
    fusion_hidden_dim=1024,
    fusion_out_dim=512,
    gate_q_dim=8,
    gate_hidden_dim=32,
    dropout=0.2,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)


lambda_hier = 0.01

run_one_batch_smoke_test(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    pos_weight=pos_weight,
    child_parent_pairs=child_parent_pairs,
    lambda_hier=lambda_hier,
    device=device,
)


ic = ra_model.compute_ic_from_label_indices(
    label_to_indices=train_label_to_indices,
    num_go_terms=len(go_terms),
    train_ids=train_keep_ids_for_aspect,
).to(device)


#Run Model
history = ra_model.fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    pos_weight=pos_weight.to(device),
    child_parent_pairs=child_parent_pairs.to(device),
    ic=ic,
    device=device,
    lambda_hier=lambda_hier,
    num_epochs=100,
    patience=10,
    out_dir="runs/bp_run_01",
)