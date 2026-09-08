# Fixed and identity-conditioned fusion baselines

Work from `EIHR-PROT` with the project environment active. These are fresh attention-pooling/head runs: ESM residue shards and EIHR retained-hit priors are reused, but neither branch loads an EIHR-PROT confidence-gate checkpoint.

## Important adaptation choices

The identity model is an **InterLabelGO+-inspired identity-conditioned fusion baseline**, not InterLabelGO+. It uses EIHR’s training-only shared DIAMOND database, retained-hit filtering (E-value <= 1e-5, qcov >= .30, top 10; training self-hits removed), unchanged EIHR priors, and the supplied implementation’s concrete top-five, bitscore-weighted `nident / min(qlen,slen)` statistic. Ties follow EIHR retention order: bitscore descending, qcov descending, E-value ascending, subject ID. No retained hit is stored as missing and evaluated with `s=0`, exactly neural-only. This differs from InterLabelGO+ databases, alignment scorer, neural network, postprocessing and its missing lookup default.

For provenance, see Liu, Zhang & Freddolino (2024), Bioinformatics 40(11), btae655, DOI 10.1093/bioinformatics/btae655, and local `InterLabelGO-main/InterLabelGO-main/{predict.py,alignment_knn.py}`. The supplied code’s combiner is `w=a+(1-a)*exp(-k*s)` where `w` weights neural scores.

## 1. Configure and validate resources

Copy the two YAMLs only to set path/budget values. Keep `seed: 42`; search sampling is independent of training seed. Trial epochs are screening only: at the supplied 8-epoch cap, patience 10 cannot normally stop a trial. Final runs select epoch and candidate by PDB validation Fmax only. Never use AF labels for selection.

Legacy nine-column TSVs cannot build an identity sidecar. Generate a separate enriched output (do not overwrite raw evidence or bins):

```bash
python scripts/prepare_diamond_hits.py --output_dir diamond_db/nident
# Run DIAMOND with the current nident-enabled outfmt into dataset-qualified files.
# Example output names: diamond_db/nident/train_hits.tsv, val_hits.tsv, test_hits.tsv, af_test_hits.tsv
```

Before scientific execution, compare enriched retained membership and all prior-driving alignment fields to the original evidence. This checkout provides schema validation and sidecar construction but does not fabricate that equivalence check; resolve any mismatch rather than replacing priors or bins.

Build a sidecar after confirmed enrichment (training uses self-hit exclusion):

```bash
python scripts/build_identity_sidecar.py --queries data/cleaned_dataset/cleaned_pdb_train.fasta --hits diamond_db/nident/train_hits.tsv --output diamond_db/nident/pdb_train_identity.json --exclude-self
python scripts/build_identity_sidecar.py --queries data/cleaned_dataset/cleaned_pdb_val.fasta --hits diamond_db/nident/val_hits.tsv --output diamond_db/nident/pdb_val_identity.json
python scripts/build_identity_sidecar.py --queries data/cleaned_dataset/cleaned_pdb_test.fasta --hits diamond_db/nident/test_hits.tsv --output diamond_db/nident/pdb_test_identity.json
python scripts/build_identity_sidecar.py --queries data/cleaned_dataset/cleaned_af_test.fasta --hits diamond_db/nident/af_test_hits.tsv --output diamond_db/nident/af_test_identity.json
```

## 2. Search then train

```bash
for aspect in BP MF CC; do python scripts/run_model_training.py --ablation sequence_homology_fixed_fusion --go_aspect $aspect --hparams configs/sequence_homology_fixed_fusion.yaml --run_type randomized_search; done
for aspect in BP MF CC; do python scripts/run_model_training.py --ablation sequence_homology_fixed_fusion --go_aspect $aspect --hparams configs/sequence_homology_fixed_fusion.yaml --run_type full_training; done
```

The identity config must contain the PDB train/validation sidecar paths before its equivalent commands. Each candidate fixes `identity_a` and `identity_k` while its new branch learns under the fusion rule. `a` and `k` are selected jointly with training parameters by PDB validation Fmax.

```bash
for aspect in BP MF CC; do python scripts/run_model_training.py --ablation sequence_homology_identity_fusion --go_aspect $aspect --hparams configs/sequence_homology_identity_fusion.yaml --run_type randomized_search; done
for aspect in BP MF CC; do python scripts/run_model_training.py --ablation sequence_homology_identity_fusion --go_aspect $aspect --hparams configs/sequence_homology_identity_fusion.yaml --run_type full_training; done
```

On PowerShell, replace each loop with `BP,MF,CC | % { python ... --go_aspect $_ ... }`. Full training consumes `search_results.json`, takes the configured top five, trains each afresh, and writes an inference-loadable `best_model.pt`; never use `best_final_run.pt` as weights.

## 3. Inference

Use the canonical files `runs/<model>/final/<aspect>/best_model.pt`, once for PDB test and once for AF test. Keep dataset-qualified ESM/homology/annotation paths separate. For identity inference also provide that dataset’s sidecar:

```bash
python scripts/inference/run_inference_seq_hom.py --ablation sequence_homology_identity_fusion --go_aspect BP --checkpoint runs/sequence_homology_identity_fusion/final/BP/best_model.pt --test_fasta data/cleaned_dataset/cleaned_pdb_test.fasta --identity_sidecar_path diamond_db/nident/pdb_test_identity.json --mode evaluate --outdir runs/test_evaluation/sequence_homology_identity_fusion/pdb_test/BP --no-bootstrap
```

Repeat for all aspects/datasets and the fixed model (which has no sidecar flag). AF uses the same six PDB-selected checkpoints; no AF training or selection. Existing cache/bootstrap infrastructure is present, but this implementation does not run the future 10,000-resample scientific comparison or create an Excel result workbook automatically.

## Results and caveats

Search records: `runs/<model>/search/<aspect>/search_results.json`. Canonical weights and source-run metadata: `runs/<model>/final/<aspect>/{best_model.pt,best_model_metadata.json}`. Inference outputs retain top-k, per-protein gates and full cache options. Fixed fusion scalar is its own AdamW group (gate LR multiplier, zero weight decay); identity `a/k` are checkpoint buffers, not trainable parameters.

Patience means consecutive validation epochs without a strict new best Fmax. Ties are failures; any new best resets the counter. Different identity policy, a/k candidate space, or training configuration requires a new output directory and selection run. Independent branch training means observed differences are not attributable to the formula alone.
