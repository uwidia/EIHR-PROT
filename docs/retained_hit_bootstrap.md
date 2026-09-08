# Retained-hit identity bootstrap workflow

The final analysis is cache-first. A cache contains a lossless `N × C` probability matrix, ordered IDs and GO terms, labels, eligibility, gates (where applicable), checkpoint hashes and immutable training annotations for fixed IC. It is never reconstructed from top-k CSV files.

Generate each cache once (replace the six independent checkpoint paths from the architectural-ablation study):

```bash
python scripts/inference/run_inference_seq_hom.py --gate confidence --go_aspect BP --mode evaluate --checkpoint runs/sequence_homology_confidence_gate/final/BP/best_model.pt --prediction_cache_path runs/caches/PDB_BP_sequence_homology_confidence_gate.npz --no-bootstrap
```

Repeat with `BP MF CC` and `--gate internal`, changing the output path. `--no-bootstrap` avoids redundant whole-test marginal intervals while preparing the six caches; labeled inference otherwise computes default 10,000-resample marginal intervals. Prediction mode remains unlabeled and does not bootstrap.

Run the requested paired, within-stratum analysis with no checkpoint, ESM, or DIAMOND access:

```bash
python scripts/experiments/run_retained_hit_identity_bootstrap.py --config configs/retained_hit_identity_bootstrap.example.yaml --output-dir runs/retained_hit_identity_bootstrap
```

It creates the workbook, JSON summary, compressed replicate sidecar and reproducible `plans/` (seed 42, B=10000). The paired difference is always full minus ablation; positive favors full for Fmax/AUPR, negative favors full for Smin.

External predictions must first be exported by their own pipeline as `protein_id,go_term,probability`, exactly once for every reference-cache cell. No unseen terms, missing rows, zero-filling or vocabulary intersection is accepted:

```bash
python scripts/experiments/import_external_prediction_cache.py --csv deepgoplus_full.csv --reference-cache runs/caches/PDB_BP_sequence_homology_confidence_gate.npz --output-cache runs/caches/PDB_BP_deepgoplus.npz --model-id DeepGOPlus
```

The current source archive has retained-hit assignment provenance and required sequence/homology artifacts. A compatibility check should compare `provenance.json` hashes with locally supplied retained DIAMOND hits/training DB before rebuilding shards; do not substitute a fresh search. ESM extraction, if missing, must use `scripts/get_embeddings.py` with `esm2_t33_650M_UR50D`, residue-level layer settings matching the checkpoint and 1022-residue truncation. The legacy `run_all_similarity_bin_inference.py` works on obsolete five FASTA subsets and is not this retained-hit analysis.

Bootstrap intervals describe sampling of PDB test proteins conditional on fitted checkpoints. They do not measure training variability, prove causal feature effects, or receive multiplicity adjustment.
