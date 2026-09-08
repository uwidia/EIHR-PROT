# Run the fusion baselines

Start with the identity model below. Each training command runs **BP, MF, and CC**.
Both new baselines exclude `2VAU-A` and `5LSQ-A` from validation only. Other models and original DIAMOND resources stay unchanged.

## 1. Activate your environment

```bash
cd /workspace/EIHR-PROT
source .venv/bin/activate
```

## 2. Prepare the identity inputs

```bash
python scripts/fusion_baselines.py prepare --splits pdb_train pdb_val
```

## 3. Verify the training resources

Paths are in `configs/fusion_baseline_resources.yaml`. They must point to your PDB training and validation embedding/homology shards.

```bash
python scripts/fusion_baselines.py verify-priors --splits pdb_train pdb_val
```

**Continue only when verification succeeds.** If it reports missing homology shards, point the YAML to your existing copies or complete the [one-time missing-shard setup](baseline_reference.md#missing-training-shards). The `prepare` command above creates identity inputs; it does not create homology shards.

## 4. Start identity randomized training

```bash
python scripts/fusion_baselines.py search --model identity
```

Default budget: **20 trials × 8 epochs per aspect**. Settings are in `configs/sequence_homology_identity_fusion.yaml`. To run BP alone, append `--aspects BP`.

## 5. After the search finishes, run final training

```bash
python scripts/fusion_baselines.py final --model identity
```

This trains the top five configurations afresh for up to 150 epochs each, with patience 10. Selection uses validation Fmax.

Results are under `runs/sequence_homology_identity_fusion/validation_exclude_two_v1/`:

- Search: `search/{aspect}/search_results.json`
- Selected checkpoint: `final/{aspect}/best_model.pt`

## Fixed baseline, when you are ready

Use the same verified training resources:

```bash
python scripts/fusion_baselines.py search --model fixed
python scripts/fusion_baselines.py final --model fixed
```

For test inference, AF preparation, comparisons, and methodology, use the [detailed reference](baseline_reference.md). You do not need those steps to start training.
