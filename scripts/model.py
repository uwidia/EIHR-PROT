#SCRIPT FOR muLtimodal version
import torch
from torch.utils.data import DataLoader
from reliability_aware.parser import get_protein_info
from reliability_aware.go_term_extraction import build_subject_go_index, build_go_annotations_list, build_child_parent_idx_pairs
from reliability_aware.losses import compute_pos_weight_from_label_indices, run_one_batch_smoke_test
from reliability_aware.reliability_aware_model import multimodal_collate_fn_generator, ReliabilityAwareProteinFunctionModel, HybridBatchSampler
from reliability_aware.shard_handling import ESMGraphHomologyShardDataset
from reliability_aware.pool_embeddings import GATBranch, ESMSequenceBranch
from reliability_aware.config import DATA_DIR, PROJECT_ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GO_ASPECT = "BP"

TSV_PATH = DATA_DIR / "HEAL_dataset/nrPDB-GO_2019.06.18_annot.tsv"
OBO_PATH = DATA_DIR / "HEAL_dataset/go-basic.obo"
GRAPH_SHARD_DIR = PROJECT_ROOT / "graph_shards/train"
HOMOLOGY_SHARD_DIR = PROJECT_ROOT / "diamond/train_homology_shards"
PDB_FASTA_DIR = DATA_DIR / "cleaned_dataset/pdb"
ESM_SHARD_DIR = PROJECT_ROOT / "esm_embeddings/pdb/pdb_train" #use directory from the runpod network volume
MANIFEST_PATH = PROJECT_ROOT / "esm_manifests/pdb_train_manifest.csv"

train_data = PDB_FASTA_DIR / "cleaned_pdb_train.fasta"

protein_info = get_protein_info(train_data)

train_ids = {
   protein["full_id"] for protein in protein_info
}

label_to_go_terms, go_terms = build_go_annotations_list(
    tsv_path=TSV_PATH,
    obo_path=OBO_PATH,
    go_aspect=GO_ASPECT,
    keep_ids=train_ids,
    remove_root_term=True,
    min_term_freq=None,
)

child_parent_pairs = build_child_parent_idx_pairs(
    obo_path=OBO_PATH,
    go_terms=go_terms,
)


go_term_to_idx = {go: i for i, go in enumerate(go_terms)}

label_to_indices = build_subject_go_index(
    label_to_go_terms,
    go_term_to_idx,
)

keep_ids_for_aspect = {
    label for label, idxs in label_to_indices.items()
    if len(idxs) > 0
}

pos_weight = compute_pos_weight_from_label_indices(
    label_to_indices=label_to_indices,
    num_go_terms=len(go_terms),
    train_ids=keep_ids_for_aspect,
    cap=20.0,
)


train_dataset = ESMGraphHomologyShardDataset(
    esm_shard_dir=ESM_SHARD_DIR,
    graph_shard_dir=GRAPH_SHARD_DIR,
    homology_shard_dir=HOMOLOGY_SHARD_DIR,
    manifest_path=MANIFEST_PATH,
    require_graph=True,
    keep_ids=keep_ids_for_aspect,
)

batch_sampler = HybridBatchSampler(
    dataset=train_dataset,
    batch_size=16,
    active_shards=3,
    lookahead_factor=2,
    drop_last=True,
    seed=42,
)

loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=multimodal_collate_fn_generator(label_to_indices=label_to_indices, num_go_terms=len(go_terms)),
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

model = ReliabilityAwareProteinFunctionModel(
    seq_branch=seq_branch,
    gat_branch=gat_branch,
    num_classes=num_go_terms,
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
    train_loader=loader,
    optimizer=optimizer,
    pos_weight=pos_weight,
    child_parent_pairs=child_parent_pairs,
    lambda_hier=lambda_hier,
    device=device,
)

#EXAMPLE TRAINING ONE EPOCH
# lambda_heir = 0.01
# for padded, mask, graph_batch, homology_prior, gate_features, targets, global_indices, labels in loader:
#     outputs = model(
#         padded=padded.to(device),
#         mask=mask.to(device),
#         graph_batch=graph_batch.to(device),
#         homology_scores=homology_prior.to(device),   # really prior probs
#         gate_features=gate_features.to(device),      # full q
#         targets = targets.to(device)
#     )
#     bce_loss = weighted_bce_probs(
#     outputs["fused_probs"],
#     targets,
#     pos_weight,
# )

#     hier_loss = hierarchy_loss(
#       outputs["fused_probs"],
#       child_parent_pairs,
#       )

# loss = bce_loss + lambda_hier * hier_loss
