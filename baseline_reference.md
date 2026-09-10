# Fusion baselines: step-by-step guide

Run all commands from the **EIHR-PROT repository root**. The runner processes BP, MF, and CC by default; add `--aspects BP` where supported to select one aspect.

Both baselines train new neural branches using existing ESM embeddings and homology priors. Both exclude `2VAU-A` and `5LSQ-A` from PDB validation, leaving 3,320 proteins before aspect filtering. Original models keep their full validation cohort.

## 1. Set up the environment and paths

Install the project environment (Python 3.11–3.12):

```bash
uv sync --extra cu128
```

Every command below uses `uv run --extra cu128` to select the project interpreter and dependencies. For CPU-only execution, replace `--extra cu128` with `--extra cpu` throughout this guide. The commands work in Bash and PowerShell; environment activation is not required.

Update these configuration files:

| File | What to configure |
| --- | --- |
| `configs/fusion_baseline_resources.yaml` | Dataset, embedding, manifest, original/enriched hit, prior, and output paths |
| `configs/sequence_homology_fixed_fusion.yaml` | Fixed-fusion search and training settings |
| `configs/sequence_homology_identity_fusion.yaml` | Identity-fusion search and training settings, including `identity_a` and `identity_k` |

Resource paths are repository-relative; `{aspect}` expands to BP, MF, or CC. Use PDB for training and validation. AF test annotations use `data/HEAL_dataset/merged_GO_annot.tsv`.

Review each model YAML for its search budget, final-training budget, and seed. Final training reads its candidates from completed search results. Choose new output directories when changing resources or training settings.

**For a fresh checkout**, complete [README: Rebuild all input artifacts](README.md#1-rebuild-all-input-artifacts) first. You need the cleaned FASTAs and annotations, PDB train/validation/test ESM shards and manifests, `diamond_db/train_db.dmnd`, original hits, and BP/MF/CC vocabularies and subject GO indices. This guide's enrichment commands reuse that training database.

Set `reference_source_dir` to the directory directly containing `predict.py` and `alignment_knn.py` (default: `../InterLabelGO-main`). Training hashes these files for provenance; it does not execute them. This source directory must be supplied separately.

All default fusion preparation, training, evaluation, and comparison outputs use `validation_exclude_two_v2`. This is an output-directory name: a fresh run does not need any v1 files or migration.

Check resource availability:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py check
```

Read `runs/fusion_resources/validation_exclude_two_v2/resource_check.json`. The command reports missing files; it does not create them. Missing enriched hits, identity sidecars, and AF shards are expected before steps 2–3.

## 2. Validate evidence and build identity resources

Keep the two preparation workflows separate:

| Workflow | Commands | Hit files | Validation policy |
| --- | --- | --- | --- |
| Original homology | `prepare_diamond_hits.py`, then `build_homology_shards.py` | `diamond_db/*_hits.tsv`; optional `nident` is ignored | Keep all proteins |
| Fusion identity resources | `fusion_baselines.py enrich`, then `prepare` | Enriched paths below | Exclude both validation queries before strict identity checks |

Reuse these enriched files if available:

| Split | Configured enriched input |
| --- | --- |
| PDB train | `runs/fusion_resources/enrichment/pdb_train_hits.tsv` |
| PDB validation | `runs/fusion_resources/enrichment/pdb_val_hits.tsv` |
| PDB test | `runs/fusion_resources/enrichment/pdb_test_hits.tsv` |
| AF test | `runs/fusion_resources/enrichment/af_test_hits.tsv` |

**Only if enriched files are missing**, run the corresponding commands with new output paths:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py enrich --splits pdb_train --output runs/fusion_resources/enrichment/pdb_train_hits.tsv --diamond ./diamond --threads 8
uv run --extra cu128 python scripts/fusion_baselines.py enrich --splits pdb_val --output runs/fusion_resources/enrichment/pdb_val_hits.tsv --diamond ./diamond --threads 8
uv run --extra cu128 python scripts/fusion_baselines.py enrich --splits pdb_test --output runs/fusion_resources/enrichment/pdb_test_hits.tsv --diamond ./diamond --threads 8
uv run --extra cu128 python scripts/fusion_baselines.py enrich --splits af_test --output runs/fusion_resources/enrichment/af_test_hits.tsv --diamond ./diamond --threads 8
```

Set `--diamond` to your executable, especially on Windows. These output paths already match the resource YAML; update `enriched_hits` only if you choose different destinations. If replacing previously prepared inputs, also choose a new `prepared_dir` and matching `identity` output paths.

Prepare the identity resources:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py prepare
```

This writes filtered hit copies, query lists, identity sidecars, and evidence reports under `runs/fusion_resources/validation_exclude_two_v2/`. It removes every validation row for `2VAU-A` and `5LSQ-A`, then validates exact counts and checks retained alignments against the original hits. Source files remain intact.

Resolve any failed evidence report before continuing. Failed reruns preserve existing sidecars and evidence reports, recording the error in `<split>_preparation_failure.json`. Do not substitute `val_hits_noiterate.tsv`, clamp counts, or drop additional queries. Use `prepare` rather than the standalone `build_identity_sidecar.py` for validation resources.

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
uv run --extra cu128 python scripts/fusion_baselines.py build-missing-homology --splits pdb_train pdb_val
```

Destinations must be empty and outside `diamond_db`. These copies preserve manifest indices and exclude the two validation queries' hit evidence. Both fusion loaders omit those queries.

**If AF embeddings are missing**, generate them using settings compatible with your training embeddings:

```bash
uv run --extra cu128 python scripts/get_embeddings.py --split test --fasta_file data/cleaned_dataset/cleaned_af_test.fasta --outdir esm_embeddings/af_test --manifest_filename af_test_manifest --model esm2_t33_650M_UR50D --repr_layer 33 --truncation_seq_length 1022 --seed 42 --deterministic
```

Add `--use_fp16` only if it matches your existing embedding precision.

**If AF homology shards are missing**, build them:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py build-missing-homology --splits af_test
```

Verify all configured priors before training or inference:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py verify-priors --splits pdb_train pdb_val pdb_test af_test
```

Rerun `uv run --extra cu128 python scripts/fusion_baselines.py check` and resolve the reported missing resources before training. Check `<split>_<aspect>_prior_report.json` in the prepared directory. Verification checks manifest alignment, prior values, and retrieval features against original evidence, excluding the two validation queries.

## 4. Search and train the fixed-fusion baseline

For a new experiment, run search first, then final training. If a compatible search is already complete, skip `search`:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py search --model fixed
uv run --extra cu128 python scripts/fusion_baselines.py final --model fixed
```

Final training requires `search_results.json` and `protocol.json` for every requested aspect under the configured `base_dir_search`. It uses the top configurations from that matching search and selects the best checkpoint by validation Fmax.

Checkpoint: `runs/sequence_homology_fixed_fusion/validation_exclude_two_v2/final/{aspect}/best_model.pt`.

## 5. Search and train the identity-fusion baseline

For a new experiment:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py search --model identity
uv run --extra cu128 python scripts/fusion_baselines.py final --model identity
```

Search selects `identity_a`, `identity_k`, and neural training settings together. Final training uses those complete configurations.

**If search is already complete**, skip the search command. For results in the configured search directory, run only `final`. If results are in another directory, pass its path with `--search-dir`. For example, reuse earlier v1 identity searches with the current prepared resources:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py final --model identity --search-dir runs/sequence_homology_identity_fusion/validation_exclude_two_v1/search --check-only
uv run --extra cu128 python scripts/fusion_baselines.py final --model identity --search-dir runs/sequence_homology_identity_fusion/validation_exclude_two_v1/search
```

The first command validates all three aspects without training. The second trains the selected candidates and saves them under the current final-output directory. Use `--aspects BP` to select one aspect. The same options work with `--model fixed` and its search directory.

Reuse requires unchanged cohorts, GO vocabulary, training settings, ESM/prior contents, reference code, and identity evidence. Only paths, regenerated sidecar provenance, and the homology builder summary may differ; every homology tensor file must remain byte-identical. Keep the historical sidecars available: their hashes are checked against the saved search before comparison with the independently validated current sidecars. Each final run records the source search and verification results. Do not rename search folders or edit saved protocol hashes.

Checkpoint: `runs/sequence_homology_identity_fusion/validation_exclude_two_v2/final/{aspect}/best_model.pt`.

## 6. Evaluate both baselines on PDB and AF test sets

After selecting all six checkpoints, run:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py infer --model fixed --dataset pdb_test
uv run --extra cu128 python scripts/fusion_baselines.py infer --model identity --dataset pdb_test
uv run --extra cu128 python scripts/fusion_baselines.py infer --model fixed --dataset af_test
uv run --extra cu128 python scripts/fusion_baselines.py infer --model identity --dataset af_test
```

AF evaluation reuses the PDB-selected checkpoints without tuning. Each command saves metrics and full prediction caches under:

`runs/test_evaluation/fusion_validation_exclude_two_v2/<model>/<dataset>/<aspect>/`

## 7. Provide confidence-gate reference predictions

Set `reference_caches` in the resource YAML to compatible full prediction caches for both datasets and all three aspects. They must match the evaluation cohorts, GO terms, annotations, and IC policy. Top-k CSV exports are insufficient.

**If reference caches are missing**, configure the three existing confidence-gate checkpoints in `reference_checkpoints` and six new `.npz` destinations in `reference_caches`, then run:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py reference --dataset pdb_test
uv run --extra cu128 python scripts/fusion_baselines.py reference --dataset af_test
```

These commands run inference using the existing checkpoints. If caches have incorrect dataset labels or missing provenance, regenerate compatible caches rather than editing their metadata.

## 8. Generate paired comparisons

```bash
uv run --extra cu128 python scripts/fusion_baselines.py compare --n-resamples 10000 --seed 42
```

Output: `runs/fusion_comparisons/validation_exclude_two_v2/fusion_baseline_comparisons.xlsx`.

The workbook reports Fmax, AUPR, and Smin with paired 95% confidence intervals for full-minus-baseline differences. Negative Smin differences favor the full model. Missing caches produce NA/status rows; incompatible caches cause an error.

Rerun the same command to reuse saved caches and resampling plans. For changed checkpoints or resources, use new `evaluation_dir` and `comparison_dir` paths and regenerate matching predictions.

## 9. Locate outputs and troubleshoot

| Output | Location |
| --- | --- |
| Resource, evidence, and prior reports | `runs/fusion_resources/validation_exclude_two_v2/` |
| Identity sidecars and per-hit audits | Same directory: `<split>_identity.json`, `<split>_identity_per_hit.json` |
| Search results | Model run directory: `search/{aspect}/search_results.json` |
| Selected weights and metadata | Model run directory: `final/{aspect}/best_model.pt`, `best_model_metadata.json`, `protocol.json` |
| Test predictions and metrics | Test output directory: `predictions.npz`, `predictions.json`, `metrics.json`, `topk_predictions.csv` |
| Comparison workbook and bootstrap plans | `runs/fusion_comparisons/validation_exclude_two_v2/` |

Use `best_model.pt` for inference. Rerunning `prepare` with unchanged inputs reuses the sidecars. When replacing prepared inputs, update `prepared_dir` and all four `identity` paths together. Changed model inputs require a new search; equivalent relocated resources can reuse an existing search through `--search-dir`. Use a new final-output directory when changing its protocol. Skip enrichment, shard building, and inference outputs that already exist. Keep manifest rows intact.

For command options:

```bash
uv run --extra cu128 python scripts/fusion_baselines.py --help
```

For workflow checks:

```bash
uv run --extra cu128 python -m pytest -q tests/test_fusion_search_reuse.py tests/test_fusion_guide.py tests/test_homology_workflow_separation.py tests/test_fusion_protocol.py tests/test_fusion_baselines.py tests/test_diamond_hit_validation.py
```

The identity rule is adapted from [InterLabelGO+](https://github.com/QuanEvans/InterLabelGO): `s = sum(bitscore * nident/min(qlen,slen)) / sum(bitscore)` over the top five retained hits; `w_neural = a + (1-a)*exp(-k*s)`. Queries with no retained hits use neural predictions only.
