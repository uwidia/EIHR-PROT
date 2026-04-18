# Reliability-Aware Protein Function Prediction

A multimodal protein function (GO term) prediction  project that combines sequence representations, structure-derived graph representations, with homology-information and weights modality contributions via a reliability-aware gate.

---

## Overview

This project is built around the simple idea that not all biological evidence is equally trustworthy.
Sequence embeddings from large protein language models are useful, but they do not explicitly encode structural uncertainty. Structural information can add important signal, but its reliability varies across proteins and even across residues. 

This predictive model accounts for this signal variability by using sequence and structure features along with a homology prior, while explicitly modeling structural confidence during graph construction, pooling, and downstream decision-making. 

It is designed based on the hypothesis that incorporating a reliability-aware gate that down-weights unreliable structural regions based on a structure confidence proxy instead of blindly trusting every residue-level structural feature will lead to more accurate and interpretable GO term predictions. 
The graph construction code already computes residue confidence proxies, edge reliability weights, and graph-level summary statistics for this purpose.

---

## Current Status

NOTE:⚠️ **This repository is still under active development.**

The preprocessing and representation-building stages are already implemented. That includes structure download and cleanup, cleaned FASTA generation, per-residue ESM embedding extraction into shards, manifest creation, graph shard construction aligned to the ESM shards, and sequence-side scalar attention pooling. 

The full end-to-end multimodal predictor is **not finished yet**. The graph encoder, fusion block, homology prior integration, reliability gate, and final GO classifier are still in progress. The current codebase should be read as a research pipeline under construction rather than a finished training framework.

---

## Features

### Completed

- **Dataset preprocessing pipeline**
  - handles AF and PDB train/test/val splits
  - downloads structure CIF files in parallel
  - cleans AlphaFold FASTA headers
  - filters PDB structures to X-ray-derived entries only
  - regenerates cleaned FASTA files
  - stores FASTA hashes for reproducibility checks

- **ESM embedding extraction**
  - uses frozen `esm2_t33_650M_UR50D`
  - extracts per-residue embeddings
  - saves embeddings in shard files
  - creates a manifest with shard ID, local index, global index, label, and truncation metadata
  - writes `run_metadata.json` for reproducibility

- **Sequence branch**
  - scalar attention pooling over residue embeddings
  - pooled sequence representation from variable-length residue embeddings

- **Structure graph construction**
  - parses CIF structures
  - builds residue graphs with a distance cutoff
  - computes node-level confidence values
  - computes confidence-derived edge weights
  - stores graph-level coverage and confidence statistics
  - truncates graphs to stay aligned with truncated ESM embeddings 

- **Shard-aware loading utilities**
  - shard dataset abstraction
  - cache-aware loading
  - custom hybrid batch sampler to reduce shard I/O bottlenecks
  - length-aware candidate batching within an active shard pool 

---

### Planned
- GAT-based structure encoder
- sequence + structure fusion via concatenation + MLP
- incorporation of homology prior branch
- reliability gate over fused embeddings and homology logits
- final GO term classifier
- end-to-end training and evaluation scripts
- baseline comparisons and ablations
- Tests and documentation

---
## Environment

### Requirements

* **Python 3.11**
* **CUDA 11.8 / cu118**
* **uv** for dependency management and reproducible environments

This project uses uv for environment management and to ensure reproducibility. I recommend you install `uv`, install Python 3.11, sync the locked environment, and run project scripts with `uv run`.

---

## Installation

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Python 3.11

```bash
uv python install 3.11
```

### 3. Clone the repository

```bash
git clone https://github.com/uwidia/Reliability-Aware-PFP.git
cd Reliability-Aware-PFP
```

### 4. Sync the environment from `uv.lock`

```bash
uv sync
```

### 5. Run everything through `uv`

```bash
uv run python --version
```

---

## How to Run

The final script for inference is still being developed. However, if you wish to reproduce the current runnable pipeline, you can do so by executing the following scripts in this order:

1. preprocessing
2. ESM embedding extraction
3. graph shard creation

That order is not optional. Graph shards depend on the ESM manifest, and the graph builder assumes the alignment produced by the embedding stage.  

---

### Step 1: Preprocess structures and FASTA files

```bash
uv run python run_preprocessing.py
```

This script:

* iterates over AF and PDB train/test/val splits
* downloads missing CIF files
* filters PDB structures to X-ray-only entries
* creates cleaned FASTA files for downstream steps 

---

### Step 2: Extract ESM embeddings

Example:

```bash
uv run python get_embeddings.py \
  --fasta data/cleaned_dataset/af/cleaned_af_train.fasta \
  --outdir esm_shards/af/train \
  --valid_hashes hashlist.txt \
  --manifest_filename af_train_manifest \
  --model esm2_t33_650M_UR50D \
  --toks_per_batch 4096 \
  --truncation_seq_length 1022 \
  --shard_size 1000 \
  --use_fp16 \
  --deterministic \
  --device cuda
```

This stage:

* loads frozen ESM-2
* extracts per-residue embeddings
* writes shard files
* creates a manifest CSV
* writes run metadata for reproducibility 

Repeat this step for each dataset split you need.

---

### Step 3: Build graph shards aligned to the ESM shards

```bash
uv run python create_graph_object.py
```

This stage:

* reads cleaned FASTA files
* reads ESM manifests
* parses CIF structure files
* builds residue graphs
* computes structural confidence features
* saves aligned graph shards for multimodal loading  

---

## Model Design

### Sequence branch

The sequence branch takes per-residue ESM embeddings and produces a pooled protein-level representation using scalar attention pooling. This is already implemented as `ScalarAttentionPooling` and `ESMSequenceBranch`. 

### Structure branch

The structure branch is designed around residue-level protein graphs. Graph nodes correspond to residues, while edges reflect structural proximity. The graph-building utilities already compute residue confidence and edge reliability information, but the full GAT encoder is still being finalized.  

### Reliability-aware prediction

The final intended model will fuse sequence and structure representations, combine them with homology evidence, and use a reliability gate before final GO prediction. That full stage is planned, but not yet fully implemented in the current repo.

---

## Current Outputs

Depending on the stage you run, the repository currently produces:

* cleaned FASTA files
* downloaded CIF files
* ESM shard files
* manifest CSV files
* graph shard files
* graph shard metadata
* run metadata JSON files  

---
## Tech Stack

* Python 3.11
* uv
* PyTorch
* ESM-2
* PyTorch Geometric
* NumPy
* Gemmi
* Parasail
* Biopython
* Requests
* tqdm
* CSV/JSON-based manifest and metadata tracking   
---

## Notes

This is a research project in progress. My current emphasis is on building a reliable data and representation pipeline first, then layering the full multimodal predictor on top of it.


