# EIHR-PROT (Explicitly Incorporated Homology Reliability Signals for Protein Function Prediction)

EIHR-PROT is a multimodal protein function (GO term) predictor that combines sequence representations with homology-information. The model receives explicit homology signals per protein and weights the contribution of sequence and homology branches via a gating mechanism to provide  GO term predictions across all aspects.

While several protein function prediction methods combining sequence representations and homology priors exist, EIHR-PROT improves performance by directly incorporating measures of homology confidence, such as maximum alignment bit score and query coverage, and number of hits to improve performance across standard CAFA metrics (Fmax, AUPR, and Smin).

EIHR-PROT also adds an added layer of interpretability by displaying how much each branch contributes to the final prediction for each protein entry.

The web version of this model is available [here](https://protein-function-predictor-ten.vercel.app/)

![Model Architecture](images/Emmanuel's%20First%20Illustration%20(2)-images-0.jpg)

---

## Environment

### Requirements

* **Python 3.11**
* **uv** for dependency management and reproducible environments
* **DIAMOND v2.1.24** for building database and performing quick sequence alignments
* Optional: an NVIDIA driver compatible with the PyTorch CUDA 12.8 build

This project uses uv for environment management. PyTorch is provided through
mutually exclusive `cpu` and `cu128` extras so the same lockfile supports both
CPU-only and NVIDIA GPU systems. However, we strongly recommend that you use the cu128 extra for optimal performance.

---

## Installation

To run EIHR-PROT on your protein sequence dataset or reproduce the model's training steps, perform the following installation steps.

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

### 3. Clone the git repository

```bash
git clone https://github.com/uwidia/EIHR-PROT.git
cd EIHR-PROT
```

### 4. Sync the environment from `uv.lock`

Choose one PyTorch variant.

CPU-only:

```bash
uv sync --extra cpu
```

NVIDIA GPU using the PyTorch CUDA 12.8 build:

```bash
uv sync --extra cu128
```

Do not enable both extras at the same time.

### 5. Verify the installation

For CPU:

```bash
uv run --extra cpu python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For NVIDIA GPU:

```bash
uv run --extra cu128 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Use the same selected extra when running project commands, for example
`uv run --extra cpu python scripts/get_embeddings.py --help`.

### 6. DIAMOND Installation
To run DIAMOND and obtain homology priors, [download the compatible DIAMONDv2.1.24 release](https://github.com/bbuchfink/diamond/releases) for your operating system.

On Linux, make the downloaded executable runnable and configure it before
running any homology commands. `uv` installs Python dependencies only; it does
not install the external `diamond` command.

```bash
chmod +x path_to_diamond_executable

# Option A: use a DIAMOND binary placed in the repository root.
# Run this from the repository root; repeat it in each new shell session.
export PATH="$PWD:$PATH"

# Option B: use a binary stored elsewhere.
# export DIAMOND_EXECUTABLE="/absolute/path/to/diamond"

# Confirm that the command is discoverable before continuing.
command -v diamond
diamond version
```

On Windows, place `diamond.exe` in the
project root and unblock it if Windows marked the download as untrusted, and set
the executable path for the current PowerShell session:

```powershell
Unblock-File .\diamond.exe
$env:DIAMOND_EXECUTABLE = (Resolve-Path .\diamond.exe)
```

If `diamond version` fails, it is not available in the current shell even if
the executable is present in the project directory. Re-run the `export PATH`
command above, or set `DIAMOND_EXECUTABLE` to the executable's absolute path.

---

## Getting Started (How to Run EIHR-PROT on your Protein Sequence Dataset)

EIHR-PROT was designed to predict protein function for single or multiple protein sequences in a fasta file. However, note that the model performs homology search against an already defined database obtained from PDB sequences used in other protein function prediction papers such as [DeepFRI](https://github.com/flatironinstitute/DeepFRI/tree/master/preprocessing/data), [HEAL](https://github.com/ZhonghuiGu/HEAL/tree/main/data), and [MAEF-GO](https://github.com/nebstudio/MAEF-GO/tree/main/data).

To modify the homology database, you will need to re-run the model training step (see: Training and Reproducibility section)

### Download the model checkpoint
Before running inference, download the confidence-gate checkpoints for all three
GO aspects [here](https://zenodo.org/records/20661853?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJhZTdmYWMyLWUyMjEtNGRkYi1hZmIyLTZiOWY4ZWNkZmRiZiIsImRhdGEiOnt9LCJyYW5kb20iOiJhZWU5MWY0ZjZlNTcyNDQ5M2Y3MmIxZWQ5MTc1NjBkYyJ9.tfDdXzsfDpw8eMGG6ERVaznDp4ixj7FSYX_-ac1lSKLg75yecloqhH9muImJp7F1f-bbGX-yMukYrxO2dNxBNg).

Ensure model checkpoints are saved to your Downloads folder, then run these commands.

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
tar -xzf "$HOME\Downloads\sequence_homology_confidence_gate.tar.gz" -C runs

Test-Path .\runs\sequence_homology_confidence_gate\final\BP\best_model.pt
Test-Path .\runs\sequence_homology_confidence_gate\final\MF\best_model.pt
Test-Path .\runs\sequence_homology_confidence_gate\final\CC\best_model.pt
```

On Linux:

```bash
mkdir -p runs
tar -xzf ~/Downloads/sequence_homology_confidence_gate.tar.gz -C runs

test -f runs/sequence_homology_confidence_gate/final/BP/best_model.pt
test -f runs/sequence_homology_confidence_gate/final/MF/best_model.pt
test -f runs/sequence_homology_confidence_gate/final/CC/best_model.pt
```
Alternatively, you can manually copy the model checkpoints to the project root's run folder and extract them with `tar -xzf sequence_homology_confidence_gate.tar.gz`

### Step 1: Extract ESM embeddings

This extracts frozen, pre-trained embeddings from `esm2_t33_650M_UR50D` (650M parameters) from [ESM-2](https://github.com/facebookresearch/ESM).

Sequences longer than 1022 amino acids are automatically truncated. For
inference, only the test/query embedding shards and manifest are required.

Choose the same PyTorch extra used during installation. Use `cpu` on a
CPU-only system or `cu128` on a system with a compatible NVIDIA driver.
**Warning:** extracting embeddings with the 650M-parameter ESM-2 model on a
CPU is extremely slow. A compatible NVIDIA GPU is strongly recommended for
this step.

For a custom dataset, use the *same query FASTA* in this embedding command
and in Step 2's `prepare_diamond_hits.py --test_dataset` command. The ESM
manifest and the DIAMOND hit file are matched by protein ID; using different
FASTA files can produce missing or mismatched homology features.

On Windows PowerShell:

```powershell
$extra = "cu128" # To run on a cpu instead, change to "cpu"

uv run --extra $extra python scripts/get_embeddings.py `
  --split test `
  --fasta_file data/cleaned_dataset/cleaned_pdb_test.fasta `
  --outdir esm_embeddings/test `
  --deterministic
```

On Linux:

```bash
EXTRA=cu128 # To run on a cpu instead, change to "cpu"

uv run --extra "$EXTRA" python scripts/get_embeddings.py \
  --split test \
  --fasta_file data/cleaned_dataset/cleaned_pdb_test.fasta \
  --outdir esm_embeddings/test \
  --deterministic
```

### Step 2: Prepare DIAMOND database and build homology shards

Before this step, ensure you've downloaded a compatible DIAMOND executable and configured access to it as described in the earlier installation step.

The first command builds a DIAMOND database from the training split and generates the shared hit files. Existing non-empty DIAMOND hit files (`train_hits.tsv`, `val_hits.tsv`, and `test_hits.tsv`) are reused; pass `--force` to regenerate the database and all hit files.

The `--splits test` commands convert only the query dataset's hits into the homology priors needed for inference.

For a custom query FASTA, replace `data/cleaned_dataset/cleaned_pdb_test.fasta`
in Step 1 with your file and pass that exact same path as `--test_dataset` below.
If you change the query FASTA after a previous run, include `--force` so that
`test_hits.tsv` is not reused for the old queries.

On Windows PowerShell:

```powershell
$extra = "cu128" # Change to "cpu" if that is your selected environment

uv run --extra $extra python scripts/prepare_diamond_hits.py `
  --test_dataset data/cleaned_dataset/cleaned_pdb_test.fasta

uv run --extra $extra python scripts/build_homology_shards.py --go_aspect BP --splits test
uv run --extra $extra python scripts/build_homology_shards.py --go_aspect MF --splits test
uv run --extra $extra python scripts/build_homology_shards.py --go_aspect CC --splits test
```

On Linux:

```bash
EXTRA=cu128 # Change to cpu if that is your selected environment

uv run --extra "$EXTRA" python scripts/prepare_diamond_hits.py \
  --test_dataset data/cleaned_dataset/cleaned_pdb_test.fasta

uv run --extra "$EXTRA" python scripts/build_homology_shards.py --go_aspect BP --splits test
uv run --extra "$EXTRA" python scripts/build_homology_shards.py --go_aspect MF --splits test
uv run --extra "$EXTRA" python scripts/build_homology_shards.py --go_aspect CC --splits test
```

### Step 3: Make Predictions

Each GO aspect uses a separate checkpoint, vocabulary, homology-shard directory, and output directory.

On Windows PowerShell:

```powershell
$extra = "cu128" # To run on a cpu instead, change to "cpu"

uv run --extra $extra python scripts/inference/run_inference_seq_hom.py `
  --mode predict `
  --go_aspect BP `
  --checkpoint runs/sequence_homology_confidence_gate/final/BP/best_model.pt `
  --outdir runs/inference/sequence_homology_confidence_gate/BP

uv run --extra $extra python scripts/inference/run_inference_seq_hom.py `
  --mode predict `
  --go_aspect MF `
  --checkpoint runs/sequence_homology_confidence_gate/final/MF/best_model.pt `
  --outdir runs/inference/sequence_homology_confidence_gate/MF

uv run --extra $extra python scripts/inference/run_inference_seq_hom.py `
  --mode predict `
  --go_aspect CC `
  --checkpoint runs/sequence_homology_confidence_gate/final/CC/best_model.pt `
  --outdir runs/inference/sequence_homology_confidence_gate/CC
```

On Linux:

```bash
EXTRA=cu128 # To run on a cpu instead, change to "cpu"

uv run --extra "$EXTRA" python scripts/inference/run_inference_seq_hom.py \
  --mode predict \
  --go_aspect BP \
  --checkpoint runs/sequence_homology_confidence_gate/final/BP/best_model.pt \
  --outdir runs/inference/sequence_homology_confidence_gate/BP

uv run --extra "$EXTRA" python scripts/inference/run_inference_seq_hom.py \
  --mode predict \
  --go_aspect MF \
  --checkpoint runs/sequence_homology_confidence_gate/final/MF/best_model.pt \
  --outdir runs/inference/sequence_homology_confidence_gate/MF

uv run --extra "$EXTRA" python scripts/inference/run_inference_seq_hom.py \
  --mode predict \
  --go_aspect CC \
  --checkpoint runs/sequence_homology_confidence_gate/final/CC/best_model.pt \
  --outdir runs/inference/sequence_homology_confidence_gate/CC
```

Prediction mode does not load an annotation TSV. It uses `go-basic.obo` only
to add the GO name, aspect, and definition to each prediction. The terminal
output is displayed as:

```text
Rank | GO ID | GO name | Aspect | Probability Score | Definition
```

`topk_predictions.csv` contains the same GO metadata together with the protein
ID, global index, branch probabilities, and gate weights. Each output directory
also contains `per_protein_gate_scores.csv`, `prediction_metadata.json`, and
`inference.log`.

---

## Training and Reproducibility

For researchers, you can use the checked-in dataset splits and configuration files to rebuild the representations, repeat hyperparameter search, train the reported models, and evaluate the resulting checkpoints.

### 1. Rebuild all input artifacts

To retraining EIHR-PROT, extract embeddings for all three splits. Repeat the Step 1 (Extract ESM Embeddings) embedding command with the following split, FASTA, and output combinations:

```text
train  data/cleaned_dataset/cleaned_pdb_train.fasta  esm_embeddings/train
val    data/cleaned_dataset/cleaned_pdb_val.fasta    esm_embeddings/val
test   data/cleaned_dataset/cleaned_pdb_test.fasta   esm_embeddings/test
```

Then run `prepare_diamond_hits.py` and `build_homology_shards.py` without `--splits test`. This builds train, validation, and test homology shards for subsequent prediction and evaluation:

On Windows PowerShell:

```powershell
$extra = "cu128" # To run on a cpu instead, change to "cpu"

uv run --extra $extra python scripts/prepare_diamond_hits.py

foreach ($aspect in @("BP", "MF", "CC")) {
  uv run --extra $extra python scripts/build_homology_shards.py `
    --go_aspect $aspect
}
```

On Linux:

```bash
EXTRA=cu128 # To run on a cpu instead, change to "cpu"

uv run --extra "$EXTRA" python scripts/prepare_diamond_hits.py

for aspect in BP MF CC; do
  uv run --extra "$EXTRA" python scripts/build_homology_shards.py \
    --go_aspect "$aspect"
done
```

This produces:

```text
esm_embeddings/{train,val,test}/
diamond_db/{BP,MF,CC}/
```

The GO vocabulary is built from training annotations only. Training homology priors exclude self-hits, while validation and test priors search only against the training database.

### 2. Choose a model configuration

| Model | Ablation argument | Configuration |
| --- | --- | --- |
| EIHR-PROT confidence gate | `sequence_homology_confidence_gate` | `configs/sequence_homology_confidence_gate.yaml` |
| Internal learned gate | `sequence_homology_internal_gate` | `configs/sequence_homology_internal_gate.yaml` |
| Sequence-only baseline | `sequence_only` | `configs/sequence_only.yaml` |
| Homology-only baseline | `homology_only` | `configs/homology_only.yaml` |

The YAML files (in the **configs** directory) contain the search spaces, selected hyperparameters, epoch
limits, patience values, output directories, and W&B settings used by the training entry point.

### 3. Repeat randomized hyperparameter search

The following example repeats the confidence-gate search for BP, MF, and CC.

On Windows PowerShell:

```powershell
$extra = "cu128" # To run on a cpu instead, change to "cpu"

foreach ($aspect in @("BP", "MF", "CC")) {
  uv run --extra $extra python scripts/run_model_training.py `
    --ablation sequence_homology_confidence_gate `
    --go_aspect $aspect `
    --hparams configs/sequence_homology_confidence_gate.yaml `
    --run_type randomized_search
}
```

On Linux:

```bash
EXTRA=cu128 # To run on a cpu instead, change to "cpu"

for aspect in BP MF CC; do
  uv run --extra "$EXTRA" python scripts/run_model_training.py \
    --ablation sequence_homology_confidence_gate \
    --go_aspect "$aspect" \
    --hparams configs/sequence_homology_confidence_gate.yaml \
    --run_type randomized_search
done
```

To search another trainable model, replace both `--ablation` and `--hparams` with the corresponding pair from the table above. Search results are written under the config's `base_dir_search` directory.

### 4. Train the selected configuration

`full_training` uses the aspect-specific `promising_hparams` stored in the selected YAML file.

On Windows PowerShell:

```powershell
$extra = "cu128" # To run on a cpu instead, change to "cpu"

foreach ($aspect in @("BP", "MF", "CC")) {
  uv run --extra $extra python scripts/run_model_training.py `
    --ablation sequence_homology_confidence_gate `
    --go_aspect $aspect `
    --hparams configs/sequence_homology_confidence_gate.yaml `
    --run_type full_training
}
```

On Linux:

```bash
EXTRA=cu128 # To run on a cpu instead, change to "cpu"

for aspect in BP MF CC; do
  uv run --extra "$EXTRA" python scripts/run_model_training.py \
    --ablation sequence_homology_confidence_gate \
    --go_aspect "$aspect" \
    --hparams configs/sequence_homology_confidence_gate.yaml \
    --run_type full_training
done
```

The confidence-gate checkpoints produced by retraining are saved as:

```text
runs/sequence_homology_confidence_gate/final/BP/run_000/best_model.pt
runs/sequence_homology_confidence_gate/final/MF/run_000/best_model.pt
runs/sequence_homology_confidence_gate/final/CC/run_000/best_model.pt
```

Use these `run_000/best_model.pt` paths with the inference commands from Step 3. The downloadable archive may use the shorter flattened checkpoint layout documented earlier.

### 5. Evaluate the homology-only baseline

The homology-only baseline has no trainable checkpoint and uses `evaluate_only`.

On Windows PowerShell:

```powershell
$extra = "cu128" # To run on a cpu instead, change to "cpu"

foreach ($aspect in @("BP", "MF", "CC")) {
  uv run --extra $extra python scripts/run_model_training.py `
    --ablation homology_only `
    --go_aspect $aspect `
    --hparams configs/homology_only.yaml `
    --run_type evaluate_only
}
```

On Linux:

```bash
EXTRA=cu128 # To run on a cpu instead, change to "cpu"

for aspect in BP MF CC; do
  uv run --extra "$EXTRA" python scripts/run_model_training.py \
    --ablation homology_only \
    --go_aspect "$aspect" \
    --hparams configs/homology_only.yaml \
    --run_type evaluate_only
done
```

### 6. Evaluate held-out test data

After rebuilding the test embeddings and homology shards, evaluate a trained
neural/gated checkpoint on the held-out test split with `--mode evaluate`. The
example below evaluates the confidence-gate model for every GO aspect. Replace
the checkpoint path with `run_000/best_model.pt` when using checkpoints
produced by the training commands above.

On Windows PowerShell:

```powershell
$extra = "cu128" # To run on a CPU instead, change to "cpu"

foreach ($aspect in @("BP", "MF", "CC")) {
  uv run --extra $extra python scripts/inference/run_inference_seq_hom.py `
    --mode evaluate `
    --gate confidence `
    --go_aspect $aspect `
    --checkpoint "runs/sequence_homology_confidence_gate/final/$aspect/best_model.pt" `
    --outdir "runs/test_evaluation/sequence_homology_confidence_gate/$aspect"
}
```

The homology-only baseline has no checkpoint:

```powershell
$extra = "cu128" # To run on a CPU instead, change to "cpu"

foreach ($aspect in @("BP", "MF", "CC")) {
  uv run --extra $extra python scripts/inference/run_inference_hom_only.py `
    --mode evaluate `
    --go_aspect $aspect `
    --outdir "runs/test_evaluation/homology_only/$aspect"
}
```

On Linux:

```bash
EXTRA=cu128 # To run on a CPU instead, change to "cpu"

for aspect in BP MF CC; do
  uv run --extra "$EXTRA" python scripts/inference/run_inference_seq_hom.py \
    --mode evaluate \
    --gate confidence \
    --go_aspect "$aspect" \
    --checkpoint "runs/sequence_homology_confidence_gate/final/$aspect/best_model.pt" \
    --outdir "runs/test_evaluation/sequence_homology_confidence_gate/$aspect"

  uv run --extra "$EXTRA" python scripts/inference/run_inference_hom_only.py \
    --mode evaluate \
    --go_aspect "$aspect" \
    --outdir "runs/test_evaluation/homology_only/$aspect"
done
```

Each evaluation output directory contains `metrics.json`. Evaluation mode
expects the held-out test FASTA, test embedding manifest, and test homology
shards to describe the same proteins. Supply the corresponding `--test_fasta`,
`--test_esm_shard_dir`, `--test_manifest_path`, and
`--test_homology_shard_dir` options if you use non-default locations.

### CAFA-style metrics

The repository evaluates each GO aspect separately and writes `Fmax`, `AUPR`,
`Smin`, their selected thresholds, precision/recall at Fmax, remaining
uncertainty (RU), and misinformation (MI) to `metrics.json`. The implemented
procedure is:

* Predictions and ground-truth annotations are propagated to GO ancestors;
  terms outside the selected aspect and the aspect root are removed. Proteins
  with no remaining ground-truth term are excluded.
* Thresholds from 0.01 to 1.00, inclusive, are evaluated. At each threshold,
  precision is averaged over proteins that have at least one prediction,
  whereas recall, RU, and MI are averaged over all evaluated proteins. `Fmax`
  is the largest resulting F-score.
* `AUPR` is the trapezoidal area under the threshold-derived precision-recall
  points after sorting them by recall.
* `Smin` is the smallest value of `sqrt(RU^2 + MI^2)`. Term information
  content is calculated from propagated training annotations as
  `log2(minimum direct-parent count / term count)`; terms not observed in the
  training annotations have zero contribution to RU and MI.

For strict comparisons, keep the dataset files, `uv.lock`, GO OBO file, annotation TSV, DIAMOND version, random seed, and YAML configuration unchanged. The embedding stage writes `run_metadata.json`, and training writes checkpoints and run metadata beneath `runs/`.

---
