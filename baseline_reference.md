# Fusion baselines: detailed reference

Work from the **EIHR-PROT repository root**. The commands below work in Bash and PowerShell: the Python runner loops over BP, MF, and CC, so no shell-specific loops are needed. This guide prepares two new baselines; it does not retrain existing models.

Both new baselines train fresh attention pooling and prediction heads. Existing ESM embeddings and the EIHR homology prior are fixed inputs. Neither baseline loads the confidence-gate neural branch. Each baseline is independently tuned, so differences cannot be attributed solely to the fusion formula.

**Validation change:** exclude the complete query proteins `2VAU-A` and `5LSQ-A` from both new baselines' PDB validation loaders and identity resources. This leaves **3,320 of 3,322 validation proteins**, before the existing aspect eligibility rules. All hit rows for these two queries are excluded from separate working copies. The original FASTA, manifests, embedding shards, DIAMOND files, prior shards, and identity-bin artifacts are preserved. Train on PDB **training** proteins; use reduced PDB validation only for selection/early stopping. PDB test (3,416 proteins) and AF test (567 proteins) remain complete, subject to the same aspect eligibility rules. Historical models retain their original settings and selection histories; document this difference when comparing them with the new baselines.

The identity variant is an **InterLabelGO+-inspired identity-conditioned fusion baseline**. Its implementation reference is the supplied `alignment_knn.py::AlignmentKNN.annotate_protein` and `predict.py::Main_pipeline.combine_score`:

- Within EIHR's final retained hits, use at most five hits ordered by descending bitscore, descending coverage, ascending E-value, then subject ID. Compute `s = sum(bitscore * nident/min(qlen,slen)) / sum(bitscore)` with unmodified bitscores. This is a fraction, with no additional division by 100. For bitscores 100/50 and identity fractions 0.5/0.8, `s=0.6`.
- `w_neural = a + (1-a)*exp(-k*s)`. The homology weight is `1-w_neural`. Each candidate fixes its own `a/k` while its attention/head learn. No retained hit is explicitly missing identity, numerically `s=0`, giving exactly neural-only predictions. A zero aspect-specific prior does not mean no retained hits.
- EIHR retains E-value ≤1e-5, query coverage ≥0.30, and at most 10 distinct targets; training self-hits are excluded. Its shared PDB-training database, bitscore×coverage prior weighting, architecture, evaluator, and full precision exports remain in use. These differ from the source's aspect-specific databases, alignment scorer, network, missing-lookup default 0.01, and output conventions. The source's `max_seqid` name represents a **bitscore-weighted top-five statistic**, not maximum local PID. Its reference a/k pairs (BP 0.45/1.5, MF 0.25/5, CC 0.45/4) are not selected parameters for this study.

Attribution: Liu, Zhang & Freddolino (2024), *InterLabelGO+: unraveling label correlations in protein function prediction*, Bioinformatics 40(11), btae655, [DOI](https://doi.org/10.1093/bioinformatics/btae655), [publication record](https://pubmed.ncbi.nlm.nih.gov/39499152/), [public source](https://github.com/QuanEvans/InterLabelGO). The supplied code's weighted calculation is the implementation reference rather than an interpretation of the paper's averaging description. Local source hashes are saved by preparation in `interlabelgo_source.json` and included in training provenance. The supplied reference repository is read-only.

## 1. Configure once

Activate your existing project environment. If installing a new environment, choose the appropriate existing project extra:

```bash
uv sync --extra cu128
```

For CPU-only use `uv sync --extra cpu`. In Bash activate with `source .venv/bin/activate`; in PowerShell use `.\.venv\Scripts\Activate.ps1`. All remaining commands use `python` from that environment. CPU is sufficient for correctness tests; real training should use your normal accelerator environment.

Edit **`configs/fusion_baseline_resources.yaml`** once for dataset, annotation, embedding, manifest, prior, and output paths. `{aspect}` is expanded by the runner. Paths are repository-relative. Keep train/validation sourced from PDB. AF annotations use `data/HEAL_dataset/merged_GO_annot.tsv`; the local check confirmed all 567 AF IDs are present. AF evaluation uses the PDB training annotation file separately for fixed IC. No global `config.py` changes are required.

The two model YAMLs are:

- `configs/sequence_homology_fixed_fusion.yaml`
- `configs/sequence_homology_identity_fusion.yaml`

They share the resource YAML and initially specify seed 42, 20 jointly sampled configurations per aspect, 8 epochs per screening trial, top 5 final candidates, up to 150 epochs per final run, and patience 10. Across both models and three aspects this is 120 screening runs and up to 30 longer final runs. These are configurable budgets, not a claim of scientific sufficiency. The identity search includes `identity_a` and `identity_k`; neither model searches unused gate-MLP settings.

Search training seed = `seed + zero_based_trial`; final seed = `seed + 10000 + zero_based_candidate_rank`. Python, NumPy, PyTorch/CUDA, and samplers are seeded. The independent search RNG uses seed 42. This does not relabel historical checkpoints as seeded or guarantee bitwise equivalence across different hardware/software.

Outputs use the new `validation_exclude_two_v1` directories. Changing exclusions, identity definitions, no-hit policy, resources, or training settings requires new output directories and a new selection run. `protocol.json` prevents silently mixing incompatible runs. Final training requires the matching search protocol; it does not use old manually populated confidence-gate hyperparameters.

```bash
python scripts/fusion_baselines.py check
```

See `runs/fusion_resources/validation_exclude_two_v1/resource_check.json`. This is a readiness report; a successful check command does not mean missing resources have been created.

**Missing in this checkout at implementation time:** PDB train and validation homology shards for BP/MF/CC, AF embeddings/manifest, and AF homology shards. Existing PDB train/validation/test embedding manifests cover all their FASTA IDs. Point the resource YAML to your existing compatible shard copies first. Optional missing-resource commands are in step 2.

## 2. Validate evidence and build identity resources

The exact-count TSVs already present in this checkout can be reused:

| Split | Enriched input |
|---|---|
| PDB train | `diamond_db/nident/train_hits.tsv` |
| PDB validation | `diamond_db/nident/val_hits_rebuilt.tsv` |
| PDB test | `diamond_db/nident/test_hits.tsv` |
| AF test | `diamond_db/nident/af_test_hits.tsv` |

Run:

```bash
python scripts/fusion_baselines.py prepare
```

This writes separate filtered original/enriched hit copies, FASTAs/ID lists, evidence reports, identity JSONs, and per-hit audits under `runs/fusion_resources/validation_exclude_two_v1/`. Validation removes all rows whose **query ID** is either excluded protein, including otherwise valid rows for those queries. It does not delete the database subject `1XX8-A`.

For each remaining query, preparation reapplies the original duplicate resolution, filters, train-only self-hit exclusion, and top-10 retention. Original/enriched retained order, targets, and all nine available alignment fields must match numerically. Equivalent numeric serialization such as `100`/`100.0` is accepted for floating fields; counts and sequence lengths remain exact integers. Conflicting exact counts for an identical alignment fingerprint, any other invalid row, or changed retained evidence blocks that split with a discrepancy report. No counts are inferred from rounded PID or clipped.

All four local comparisons passed: 29,902 training records, 3,320 validation records, 3,416 PDB test records, and 567 AF test records. The generated validation audit records the two excluded query IDs and removed source rows. **Legacy nine-field records have no alignment coordinates**: these checks establish agreement on available serialized evidence, not a unique residue-level alignment path. The AF comparison currently uses the same enriched file as its original evidence; it is not an independent rerun-equivalence check. Actual prior agreement is checked separately below.

Do not substitute `val_hits_noiterate.tsv`: the prior investigation found it changes retained evidence for many remaining queries. This workflow implements the chosen two-query exclusion instead of changing retrieval settings. Existing other-model resources and bins remain untouched.

If an enriched input is missing, a DIAMOND rerun is optional preparation you execute yourself. It reuses the configured existing training DB and original sensitive/iterative search settings, appends exact `nident`, and records the command, executable version, schema, DB/query/output hashes. Give a **new** output path:

```bash
python scripts/fusion_baselines.py enrich --splits pdb_train --output runs/fusion_resources/enrichment/pdb_train_hits.tsv --diamond ./diamond --threads 8
python scripts/fusion_baselines.py enrich --splits pdb_val --output runs/fusion_resources/enrichment/pdb_val_hits.tsv --diamond ./diamond --threads 8
python scripts/fusion_baselines.py enrich --splits pdb_test --output runs/fusion_resources/enrichment/pdb_test_hits.tsv --diamond ./diamond --threads 8
python scripts/fusion_baselines.py enrich --splits af_test --output runs/fusion_resources/enrichment/af_test_hits.tsv --diamond ./diamond --threads 8
```

On Windows set `--diamond` to your installed executable. Set each corresponding `enriched_hits` path in the resource YAML, select a new `prepared_dir` and matching identity output paths, then rerun `prepare`. Existing outputs are never overwritten implicitly. There is no need for another ESM representation when enriching DIAMOND counts.

Once the original shard copies are available, verify them without changing their contents:

```bash
python scripts/fusion_baselines.py verify-priors --splits pdb_train pdb_val pdb_test
```

This checks manifest alignment, complete coverage, prior values in their stored precision, and retrieval features against the original retained evidence, excluding only the two validation queries. Training and baseline inference also perform this check before use. Reports are `<split>_<aspect>_prior_report.json`. Resource fingerprints bind checkpoints to the actual files; hashing large embedding directories can take time at startup.

<a id="missing-training-shards"></a>

**Only if the original PDB train/validation shards cannot be restored:** reconstruct missing baseline-local copies from the same original hits, vocabulary, and subject GO index. First change only these two entries in the baseline resource YAML:

```yaml
pdb_train:
  # Keep all other fields from the provided YAML.
  homology_shards: runs/fusion_resources/pdb_train/{aspect}/homology_shards
pdb_val:
  # Keep all other fields from the provided YAML.
  homology_shards: runs/fusion_resources/pdb_val/{aspect}/homology_shards
```

Then:

```bash
python scripts/fusion_baselines.py build-missing-homology --splits pdb_train pdb_val
python scripts/fusion_baselines.py verify-priors --splits pdb_train pdb_val
```

The missing-resource builder refuses destinations under the original `diamond_db` and refuses nonempty destinations. It keeps the full manifest indexing; excluded validation queries are not used by either new model. Reconstructed copies can be checked against the original hit formula; unavailable historical shard bytes cannot be retrospectively verified.

**Only if AF embeddings are missing:** the existing training metadata specifies ESM2 `esm2_t33_650M_UR50D`, layer 33, truncation 1022, no pooling. Match your checkpoint-compatible embedding precision and extraction settings. This is a shared AF split preparation, not another identity-model embedding network:

```bash
python scripts/get_embeddings.py --split test --fasta_file data/cleaned_dataset/cleaned_af_test.fasta --outdir esm_embeddings/af_test --manifest_filename af_test_manifest --model esm2_t33_650M_UR50D --repr_layer 33 --truncation_seq_length 1022 --seed 42 --deterministic
python scripts/fusion_baselines.py build-missing-homology --splits af_test
python scripts/fusion_baselines.py verify-priors --splits af_test
```

Use `--use_fp16` only if it matches the existing embedding resources you are reusing. The manifest check must cover all 567 AF test IDs. AF homology uses the same PDB training subject index and GO vocabularies, in distinct AF directories. Do not run global preparation with `--force` or rebuild any bins.

## 3. Fixed baseline screening

```bash
python scripts/fusion_baselines.py search --model fixed
```

This runs independent BP/MF/CC randomized searches. Add `--aspects BP` to run a single aspect. The scalar starts at `z=0`, hence 50/50 fusion, and learns jointly with the branch. It has one AdamW parameter group, the configured gate learning-rate multiplier, and zero weight decay; branch parameters use the configured weight decay. No-hit proteins retain the same global convex weight with their zero prior.

Sorted results: `runs/sequence_homology_fixed_fusion/validation_exclude_two_v1/search/{aspect}/search_results.json`.

## 4. Fixed baseline longer final training

```bash
python scripts/fusion_baselines.py final --model fixed
```

The runner loads the top five complete configurations from the matching sorted search results, trains each from scratch, and selects the best epoch and final run only by validation Fmax. It saves actual inference-loadable weights at:

`runs/sequence_homology_fixed_fusion/validation_exclude_two_v1/final/{aspect}/best_model.pt`

Patience counts **consecutive epochs without a strict new best**. Ties count as non-improvement; recovery below the previous best does not reset it; any new best does. With patience 3, `[.50,.49,.51,.50,.505,.51]` stops at epoch 6 and selects epoch 3. At the screening cap of 8 epochs, patience 10 cannot normally trigger early stopping. Short trials are screening, not converged final runs.

## 5. Identity baseline screening and final training

```bash
python scripts/fusion_baselines.py search --model identity
python scripts/fusion_baselines.py final --model identity
```

Each trial jointly samples `a/k` and applicable neural/loss/training parameters. Its own newly initialized branch learns under that exact fusion rule. The `a/k` buffers do not receive optimizer updates. Final runs carry complete candidate tuples forward, including boundary cases `a=1` or `k=0` if validation selects them. There is no frozen-score fitting or post-hoc parameter substitution.

Canonical checkpoints:

`runs/sequence_homology_identity_fusion/validation_exclude_two_v1/final/{aspect}/best_model.pt`

Each contains its own neural weights, a/k buffers and hparams, ordered GO vocabulary/hash, aspect/model type, selected epoch/validation metrics, seeds, validation exclusions, adaptation policy, and resource provenance. Inference rejects parameter-summary/state disagreement. Neither baseline needs a full-model checkpoint to load.

The runner invokes the existing training entry point. Equivalent single-aspect commands are:

```bash
python scripts/run_model_training.py --ablation sequence_homology_fixed_fusion --go_aspect BP --hparams configs/sequence_homology_fixed_fusion.yaml --run_type randomized_search
python scripts/run_model_training.py --ablation sequence_homology_identity_fusion --go_aspect BP --hparams configs/sequence_homology_identity_fusion.yaml --run_type full_training
```

## 6. Complete PDB test inference

After all six checkpoints have been selected:

```bash
python scripts/fusion_baselines.py infer --model fixed --dataset pdb_test
python scripts/fusion_baselines.py infer --model identity --dataset pdb_test
```

Each command runs all three aspects once, with point estimates and full score caches. Scores are not rounded or truncated to top-k for metrics/caching. Existing aspect eligibility and CAFA propagation are retained. The threshold grid is 0.01 through 1.00 inclusive, using `>=`. No bootstrap runs during training, screening, or these inference passes.

## 7. Complete AF test inference

After the AF split resources pass validation:

```bash
python scripts/fusion_baselines.py infer --model fixed --dataset af_test
python scripts/fusion_baselines.py infer --model identity --dataset af_test
```

These commands reuse exactly the same six PDB-selected checkpoints, including unchanged identity a/k. AF is never used for gradient updates, early stopping, or configuration selection. The same PDB training-derived IC, GO vocabulary, and OBO are used. The inference runner checks complete FASTA/manifest coverage and complete aspect-eligible output coverage.

## 8. Paired comparisons and Excel workbook

The shared bootstrap framework is present. Provide the existing confidence-gate **full prediction caches** in `reference_caches` in the resource YAML, separately for PDB/AF and BP/MF/CC. They must use the same complete aspect-eligible cohorts, labels, fixed training annotations/IC, GO vocabulary, OBO, and annotation policy. Protein/GO order can be aligned by ID; differing sets are rejected. Existing top-k CSVs cannot reconstruct full caches.

If compatible reference caches do not exist, set the three already-selected historical paths in `reference_checkpoints`, and six destination `.npz` paths in `reference_caches`, then run only inference:

```bash
python scripts/fusion_baselines.py reference --dataset pdb_test
python scripts/fusion_baselines.py reference --dataset af_test
```

This loads the old confidence-gate checkpoints without training, changing their weights, changing DIAMOND settings, or selecting anything from test scores. If earlier AF caches were incorrectly labeled `PDB` by the previous exporter, the comparison rejects that mismatch; obtain correctly labeled caches with the commands above. If historical caches lack required provenance, resolve that evidence gap rather than editing labels/hashes to make them pass.

Then:

```bash
python scripts/fusion_baselines.py compare --n-resamples 10000 --seed 42
```

The command loads saved caches and fixed evaluation resources, not embeddings or models. It uses the same sampled protein indices for full-versus-fixed and full-versus-identity within each dataset/aspect. It recomputes aggregate Fmax/AUPR/Smin per replicate, keeps original-cohort point estimates, and reports full-minus-baseline differences with 95% percentile CIs. Negative Smin differences favor the full model. No result direction is assumed.

Output: `runs/fusion_comparisons/validation_exclude_two_v1/fusion_baseline_comparisons.xlsx`, with 36 numeric metric/status rows (2 datasets × 3 aspects × 2 comparisons × 3 metrics), `Performance`, `PDB`, `AF`, `Fixed_weights`, `Identity_parameters`, and `Provenance` sheets. Unconfigured/missing caches produce explicit NA/status rows; incompatible present caches fail. NA rows are not completed scientific results.

**Cache-only rerun:** rerun the exact `compare` command. It reuses the matching saved resampling plans. Inference refuses to overwrite existing caches. For changed checkpoints, resources, or policy, select a new `evaluation_dir` and `comparison_dir`, produce matching references, and run again.

## 9. Find results and resolve errors

| Artifact | Path pattern |
|---|---|
| Readiness / evidence / prior reports | `runs/fusion_resources/validation_exclude_two_v1/*_report.json`, `resource_check.json` |
| Identity and per-hit audits | Same directory: `<split>_identity.json`, `<split>_identity_per_hit.json` |
| Selected model weights | `runs/sequence_homology_<fixed or identity>_fusion/validation_exclude_two_v1/final/{aspect}/best_model.pt` |
| Selected parameters / source run / provenance | Beside weights: `best_model_metadata.json`, `protocol.json`, `final_results.json` |
| Search records and individual histories | Model `search/{aspect}/search_results.json`, `trial_*/history.pt`; final `run_*/history.pt` |
| Test outputs | `runs/test_evaluation/fusion_validation_exclude_two_v1/<model>/<dataset>/<aspect>/` |
| Complete eligible score cache | Test output directory: `predictions.npz` and `predictions.json` |
| Human-readable exports | Test output directory: `topk_predictions.csv`, `per_protein_gate_scores.csv`, `prediction_metadata.json`, `metrics.json` |
| Paired results / plans / replicates | `runs/fusion_comparisons/validation_exclude_two_v1/` |

Use `best_model.pt` for inference, never metadata-only `best_final_run.pt` or `final_meta.pt`. Missing identity records, invalid counts, stale evidence hashes, cohort mismatches, changed prior values, or wrong aspect/vocabulary are errors, not no-hit fallbacks. Restore matching resources or use a new fully prepared protocol directory. Do not clamp values, drop additional queries, substitute the non-iterative search, or delete manifest rows to bypass validation.

Lightweight verification:

```bash
python -m pytest -q tests/test_fusion_protocol.py tests/test_fusion_baselines.py tests/test_bootstrap_framework.py tests/test_prediction_mode.py tests/test_inference_layout.py tests/test_diamond_hit_validation.py
python scripts/fusion_baselines.py --help
python scripts/run_model_training.py --help
python scripts/inference/run_inference_seq_hom.py --help
```

No full training, search sweep, embedding extraction, scientific inference, or 10,000-resample evaluation was launched during implementation. The six selected checkpoints, twelve baseline test outputs, and populated study comparison workbook are future outputs of this guide. Synthetic test workbooks stay in temporary test directories.
