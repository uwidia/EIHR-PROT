# Fusion baselines: step-by-step guide

Run all commands from the **EIHR-PROT repository root**. The runner processes BP, MF, and CC by default; add `--aspects BP` where supported to select one aspect.

Both baselines train new neural branches using existing ESM embeddings and homology priors. Both exclude `2VAU-A` and `5LSQ-A` from PDB validation, leaving 3,320 proteins before aspect filtering. Original models keep their full validation cohort.

## 1. Set up the environment and paths

Install the project environment:

```bash
uv sync --extra cu128
```

Use `--extra cpu` for CPU-only execution. Activate the environment before running the remaining commands:

- Bash: `source .venv/bin/activate`
- PowerShell: `.\.venv\Scripts\Activate.ps1`

Update these configuration files:

| File | What to configure |
| --- | --- |
| `configs/fusion_baseline_resources.yaml` | Dataset, embedding, manifest, original/enriched hit, prior, and output paths |
| `configs/sequence_homology_fixed_fusion.yaml` | Fixed-fusion search and training settings |
| `configs/sequence_homology_identity_fusion.yaml` | Identity-fusion search and training settings, including `identity_a` and `identity_k` |

Resource paths are repository-relative; `{aspect}` expands to BP, MF, or CC. Use PDB for training and validation. AF test annotations use `data/HEAL_dataset/merged_GO_annot.tsv`.

Defaults per model and aspect: seed 42, 20 search trials, 8 epochs per trial, top 5 configurations for final training, up to 150 final epochs, and patience 10. Choose new output directories when changing resources or training settings.

Check resource availability:

```bash
python scripts/fusion_baselines.py check
```

Read `runs/fusion_resources/validation_exclude_two_v1/resource_check.json`. The command reports missing files; it does not create them.

## 2. Validate evidence and build identity resources

Keep the two preparation workflows separate:

| Workflow | Commands | Hit files | Validation policy |
| --- | --- | --- | --- |
| Original homology | `prepare_diamond_hits.py`, then `build_homology_shards.py` | `diamond_db/*_hits.tsv`; optional `nident` is ignored | Keep all proteins |
| Fusion identity resources | `fusion_baselines.py enrich`, then `prepare` | Enriched paths below | Exclude both validation queries before strict identity checks |

Reuse these enriched files if available:

| Split | Configured enriched input |
| --- | --- |
| PDB train | `diamond_db/nident/train_hits.tsv` |
| PDB validation | `diamond_db/nident/val_hits_rebuilt.tsv` |
| PDB test | `diamond_db/nident/test_hits.tsv` |
| AF test | `diamond_db/nident/af_test_hits.tsv` |

**Only if enriched files are missing**, run the corresponding commands with new output paths:

```bash
python scripts/fusion_baselines.py enrich --splits pdb_train --output runs/fusion_resources/enrichment/pdb_train_hits.tsv --diamond ./diamond --threads 8
python scripts/fusion_baselines.py enrich --splits pdb_val --output runs/fusion_resources/enrichment/pdb_val_hits.tsv --diamond ./diamond --threads 8
python scripts/fusion_baselines.py enrich --splits pdb_test --output runs/fusion_resources/enrichment/pdb_test_hits.tsv --diamond ./diamond --threads 8
python scripts/fusion_baselines.py enrich --splits af_test --output runs/fusion_resources/enrichment/af_test_hits.tsv --diamond ./diamond --threads 8
```

Set `--diamond` to your executable, especially on Windows. Update each `enriched_hits` path in the resource YAML. If replacing previously prepared inputs, also choose a new `prepared_dir` and matching `identity` output paths.

Prepare the identity resources:

```bash
python scripts/fusion_baselines.py prepare
```

This writes filtered hit copies, query lists, identity sidecars, and evidence reports under `runs/fusion_resources/validation_exclude_two_v1/`. It removes every validation row for `2VAU-A` and `5LSQ-A`, then validates exact counts and checks retained alignments against the original hits. Source files remain intact.

Resolve any failed evidence report before continuing. Do not substitute `val_hits_noiterate.tsv`, clamp counts, or drop additional queries. Use `prepare` rather than the standalone `build_identity_sidecar.py` for validation resources.

## 3. Supply missing shards, then verify priors

Reuse existing compatible ESM and homology shards by setting their paths in the resource YAML. Skip the optional builds below when those files exist.

<a id="missing-training-shards"></a>

**If PDB train/validation homology shards are unavailable**, set these destinations in the resource YAML, keeping all other fields:

```yaml
pdb_train:
  homology_shards: runs/fusion_resources/pdb_train/{aspect}/homology_shards
pdb_val:
  homology_shards: runs/fusion_resources/pdb_val/{aspect}/homology_shards
```

Build separate fusion copies after completing step 2:

```bash
python scripts/fusion_baselines.py build-missing-homology --splits pdb_train pdb_val
```

Destinations must be empty and outside `diamond_db`. These copies preserve manifest indices and exclude the two validation queries' hit evidence. Both fusion loaders omit those queries.

**If AF embeddings are missing**, generate them using settings compatible with your training embeddings:

```bash
python scripts/get_embeddings.py --split test --fasta_file data/cleaned_dataset/cleaned_af_test.fasta --outdir esm_embeddings/af_test --manifest_filename af_test_manifest --model esm2_t33_650M_UR50D --repr_layer 33 --truncation_seq_length 1022 --seed 42 --deterministic
```

Add `--use_fp16` only if it matches your existing embedding precision.

**If AF homology shards are missing**, build them:

```bash
python scripts/fusion_baselines.py build-missing-homology --splits af_test
```

Verify all configured priors before training or inference:

```bash
python scripts/fusion_baselines.py verify-priors --splits pdb_train pdb_val pdb_test af_test
```

Check `<split>_<aspect>_prior_report.json` in the prepared directory. Verification checks manifest alignment, prior values, and retrieval features against original evidence, excluding the two validation queries.

## 4. Search and train the fixed-fusion baseline

Run search first, then final training:

```bash
python scripts/fusion_baselines.py search --model fixed
python scripts/fusion_baselines.py final --model fixed
```

Final training uses the top configurations from the matching search and selects the best checkpoint by validation Fmax.

Checkpoint: `runs/sequence_homology_fixed_fusion/validation_exclude_two_v1/final/{aspect}/best_model.pt`.

## 5. Search and train the identity-fusion baseline

```bash
python scripts/fusion_baselines.py search --model identity
python scripts/fusion_baselines.py final --model identity
```

Search selects `identity_a`, `identity_k`, and neural training settings together. Final training uses those complete configurations.

Checkpoint: `runs/sequence_homology_identity_fusion/validation_exclude_two_v1/final/{aspect}/best_model.pt`.

## 6. Evaluate both baselines on PDB and AF test sets

After selecting all six checkpoints, run:

```bash
python scripts/fusion_baselines.py infer --model fixed --dataset pdb_test
python scripts/fusion_baselines.py infer --model identity --dataset pdb_test
python scripts/fusion_baselines.py infer --model fixed --dataset af_test
python scripts/fusion_baselines.py infer --model identity --dataset af_test
```

AF evaluation reuses the PDB-selected checkpoints without tuning. Each command saves metrics and full prediction caches under:

`runs/test_evaluation/fusion_validation_exclude_two_v1/<model>/<dataset>/<aspect>/`

## 7. Provide confidence-gate reference predictions

Set `reference_caches` in the resource YAML to compatible full prediction caches for both datasets and all three aspects. They must match the evaluation cohorts, GO terms, annotations, and IC policy. Top-k CSV exports are insufficient.

**If reference caches are missing**, configure the three existing confidence-gate checkpoints in `reference_checkpoints` and six new `.npz` destinations in `reference_caches`, then run:

```bash
python scripts/fusion_baselines.py reference --dataset pdb_test
python scripts/fusion_baselines.py reference --dataset af_test
```

These commands run inference using the existing checkpoints. If caches have incorrect dataset labels or missing provenance, regenerate compatible caches rather than editing their metadata.

## 8. Generate paired comparisons

```bash
python scripts/fusion_baselines.py compare --n-resamples 10000 --seed 42
```

Output: `runs/fusion_comparisons/validation_exclude_two_v1/fusion_baseline_comparisons.xlsx`.

The workbook reports Fmax, AUPR, and Smin with paired 95% confidence intervals for full-minus-baseline differences. Negative Smin differences favor the full model. Missing caches produce NA/status rows; incompatible caches cause an error.

Rerun the same command to reuse saved caches and resampling plans. For changed checkpoints or resources, use new `evaluation_dir` and `comparison_dir` paths and regenerate matching predictions.

## 9. Locate outputs and troubleshoot

| Output | Location |
| --- | --- |
| Resource, evidence, and prior reports | `runs/fusion_resources/validation_exclude_two_v1/` |
| Identity sidecars and per-hit audits | Same directory: `<split>_identity.json`, `<split>_identity_per_hit.json` |
| Search results | Model run directory: `search/{aspect}/search_results.json` |
| Selected weights and metadata | Model run directory: `final/{aspect}/best_model.pt`, `best_model_metadata.json`, `protocol.json` |
| Test predictions and metrics | Test output directory: `predictions.npz`, `predictions.json`, `metrics.json`, `topk_predictions.csv` |
| Comparison workbook and bootstrap plans | `runs/fusion_comparisons/validation_exclude_two_v1/` |

Use `best_model.pt` for inference. For stale hashes, protocol mismatches, invalid counts, or changed priors, restore matching inputs or prepare a new output directory. Keep manifest rows intact.

For command options:

```bash
python scripts/fusion_baselines.py --help
```

For workflow checks:

```bash
python -m pytest -q tests/test_homology_workflow_separation.py tests/test_fusion_protocol.py tests/test_fusion_baselines.py tests/test_diamond_hit_validation.py
```

The identity rule is adapted from [InterLabelGO+](https://github.com/QuanEvans/InterLabelGO): `s = sum(bitscore * nident/min(qlen,slen)) / sum(bitscore)` over the top five retained hits; `w_neural = a + (1-a)*exp(-k*s)`. Queries with no retained hits use neural predictions only.
