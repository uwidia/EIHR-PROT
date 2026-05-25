# New Main Model Configs

These templates cover the current active model set only:

- `sequence_only`
- `homology_only`
- `sequence_homology_fixed`
- `sequence_homology_internal_gate`
- `sequence_homology_confidence_gate`

They intentionally omit structure, graph, GAT, and old fusion hyperparameters.

```bash
uv run python scripts/run_model_training.py --ablation sequence_only --go_aspect BP --run_type randomized_search --hparams configs/new_main_models/sequence_only_search.yaml

uv run python scripts/run_model_training.py --ablation sequence_homology_confidence_gate --go_aspect BP --run_type randomized_search --hparams configs/new_main_models/sequence_homology_confidence_gate_search.yaml

uv run python scripts/run_model_training.py --ablation homology_only --go_aspect BP --run_type evaluate_only --hparams configs/new_main_models/homology_only_eval.yaml
```
