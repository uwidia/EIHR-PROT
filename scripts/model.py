#SCRIPT FOR muLtimodal version
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from reliability_aware.parser import get_protein_info
from reliability_aware.go_term_extraction import build_subject_go_index, build_go_annotations_list, build_child_parent_idx_pairs
from reliability_aware.losses import compute_pos_weight_from_label_indices, run_one_batch_smoke_test
import reliability_aware.reliability_aware_model as ra_model
from reliability_aware.shard_handling import ESMGraphHomologyShardDataset
from reliability_aware.pool_embeddings import GATBranch, ESMSequenceBranch
from reliability_aware.config import DATA_DIR, PROJECT_ROOT
from copy import deepcopy
import json
import random
import logging
import reliability_aware.config as config


config.setup_logging()
logger = logging.getLogger(__name__)


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_dataset(dataset, name):
    valid_indices = []

    for i in range(len(dataset)):
        try:
            sample = dataset[i]

            if sample["graph"] is not None:
                valid_indices.append(i)

        except Exception as e:
            logger.info(f"Skipping sample {i}: {e}")
    print(f"Filtering complete for {name}")

    return torch.utils.data.Subset(dataset, valid_indices)


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


print("Filtering train dataset...")
train_dataset._filter_invalid_samples()

print("Filtering validation dataset...")
val_dataset._filter_invalid_samples()




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

ic = ra_model.compute_ic_from_label_indices(
    label_to_indices=train_label_to_indices,
    num_go_terms=len(go_terms),
    train_ids=train_keep_ids_for_aspect,
).to(device)


num_go_terms = len(go_terms) 

# lambda_hier = 0.01


# #Randomized Search
# SEARCH_SPACE = {
#     "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4],
#     "fusion_hidden_dim": [512, 768, 1024, 1536],
#     "lambda_hier": [0.001, 0.005, 0.01, 0.05, 0.1],
#     "pos_weight_cap": [5.0, 10.0, 20.0, 30.0],
#     "dropout": [0.1, 0.2, 0.3, 0.4],
# }

# def sample_hparams():
#     return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}

# best_score = -1.0
# best_record = None
# num_trials = 20
# counter = 1

# for trial in range(num_trials):
    
#     h = sample_hparams()

#     pos_weight = compute_pos_weight_from_label_indices(
#         label_to_indices=train_label_to_indices,
#         num_go_terms=len(go_terms),
#         train_ids=train_keep_ids_for_aspect,
#         cap=h["pos_weight_cap"],
#     )

#     seq_branch = ESMSequenceBranch(attn_dropout=h["dropout"])
#     gat_branch = GATBranch(dropout=h["dropout"])

#     model = ra_model.ReliabilityAwareProteinFunctionModel(
#         seq_branch=seq_branch,
#         gat_branch=gat_branch,
#         num_go_terms=len(go_terms),
#         fusion_hidden_dim=h["fusion_hidden_dim"],
#         dropout=h["dropout"],
#     ).to(device)

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=h["learning_rate"],
#         weight_decay=1e-4,
#     )
    
#     if counter == 1:
#         run_one_batch_smoke_test(
#         model=model,
#         train_loader=train_loader,
#         optimizer=optimizer,
#         pos_weight=pos_weight,
#         child_parent_pairs=child_parent_pairs,
#         lambda_hier=lambda_hier,
#         device=device,
#         )
#         print("Smoke test passed.... Now running training")
#     counter += 1
    
#     trial_dir = Path(f"runs/search/trial_{trial:03d}")

#     history = ra_model.fit(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         optimizer=optimizer,
#         pos_weight=pos_weight.to(device),
#         child_parent_pairs=child_parent_pairs.to(device),
#         ic=ic,
#         device=device,
#         lambda_hier=h["lambda_hier"],
#         num_epochs=6,
#         patience=15,
#         out_dir=trial_dir,
#         hparams=deepcopy(h),
#     )

#     score = max(history["val_Fmax"])

#     record = {
#         "trial": trial,
#         "score": score,
#         "metrics": {
#             "Fmax": history["val_Fmax"],
#             "Smin": history["val_Smin"],
#             "AUPR": history["val_AUPR"],
#         },
#         "hparams": deepcopy(h),
#     }

#     torch.save(record, trial_dir / "best_meta.pt")

#     if score > best_score:
#         best_score = score
#         best_record = record
#         torch.save(record, "runs/search/best_meta.pt")

# print(best_record)



#MODEL RUN
promising_hparams = [
    {
        "learning_rate": 0.0001,
        "fusion_hidden_dim": 768,
        "lambda_hier": 0.1,
        "pos_weight_cap": 20.0,
        "dropout": 0.4,
    },
    {
        "learning_rate": 0.0003,
        "fusion_hidden_dim": 512,
        "lambda_hier": 0.01,
        "pos_weight_cap": 5.0,
        "dropout": 0.1,
    },
    {
        "learning_rate": 0.0003,
        "fusion_hidden_dim": 1024,
        "lambda_hier": 0.1,
        "pos_weight_cap": 10.0,
        "dropout": 0.4,
    },
]

results = []
best_score = -1.0
best_run = None

for run_id, h in enumerate(promising_hparams):
    run_dir = Path(f"runs/final_runs/run_{run_id:03d}")
    run_dir.mkdir(parents=True, exist_ok=True)

    pos_weight = compute_pos_weight_from_label_indices(
        label_to_indices=train_label_to_indices,
        num_go_terms=len(go_terms),
        train_ids=train_keep_ids_for_aspect,
        cap=h["pos_weight_cap"],
    )

    seq_branch = ESMSequenceBranch(attn_dropout=h["dropout"])
    gat_branch = GATBranch(dropout=h["dropout"])

    model = ra_model.ReliabilityAwareProteinFunctionModel(
        seq_branch=seq_branch,
        gat_branch=gat_branch,
        num_go_terms=len(go_terms),
        fusion_hidden_dim=h["fusion_hidden_dim"],
        dropout=h["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=h["learning_rate"],
        weight_decay=1e-4,
    )

    history = ra_model.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        pos_weight=pos_weight.to(device),
        child_parent_pairs=child_parent_pairs.to(device),
        ic=ic,
        device=device,
        lambda_hier=h["lambda_hier"],
        num_epochs=50,
        patience=10,
        out_dir=run_dir,
        hparams=deepcopy(h),
    )

    score = max(history["val_Fmax"])
    record = {
        "run_id": run_id,
        "score": score,
        "history": history,
        "hparams": deepcopy(h),
    }

    torch.save(record, run_dir / "final_meta.pt")
    results.append(record)

    if score > best_score:
        best_score = score
        best_run = record
        torch.save(record, "runs/final_runs/best_final_run.pt")

print("Best run:")
print(best_run)