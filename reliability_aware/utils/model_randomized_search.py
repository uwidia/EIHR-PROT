from __future__ import annotations
import random
from pathlib import Path

from models.sequence_only_ablation import run_one_batch_smoke_test_sequence_only

from reliability_aware.utils.losses import compute_pos_weight_from_label_indices
from reliability_aware.utils.model_training import build_record, save_and_track_best


def _sample_hparams(search_space) -> dict:
    return {k: random.choice(v) for k, v in search_space.items()}


# Sequence Only Ablation
def run_randomized_search(
    *,
    train_keep_ids_for_aspect,
    train_label_to_indices,
    go_terms,
    child_parent_pairs,
    go_aspect,
    obo_path,
    train_annotations,
    search_space: dict,
    device,
    num_trials: int = 20,
    trial_epochs: int = 6,
    train_loader,
    val_loader,
    fit_function,
    build_model_fn,
    smoke_test_fn,
    patience: int = 15,
    base_dir: str | Path = "runs/seq_only_search",
    smoke_test: bool = True,
    top_k_params: int = 5,
    use_wandb: bool = False,
    wandb_project: str = "reliability-aware-pfp",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    ablation: str | None = None,
    run_type: str = "randomized_search",
) -> list[dict]:
    """
    Randomly samples `num_trials` hyperparameter configurations, trains each
    for `trial_epochs` epochs, and returns the full list of trial records
    sorted by descending val_Fmax.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    records = []
    best_score = -1.0
    best_record = None

    for trial in range(num_trials):
        sample_hparams = _sample_hparams(search_space)
        print(f"\n{'='*60}")
        print(f"[Search] Trial {trial+1}/{num_trials}  hparams={sample_hparams}")
        print(f"{'='*60}")

        pos_weight = compute_pos_weight_from_label_indices(
            label_to_indices=train_label_to_indices,
            num_go_terms=len(go_terms),
            train_ids=train_keep_ids_for_aspect,
            cap=sample_hparams["pos_weight_cap"],
        )

        model, optimizer = build_model_fn(sample_hparams, go_terms, device)

        # Smoke test on the very first trial only
        if smoke_test and trial == 0:
            print("Running smoke test on trial 0...")
            smoke_test_fn(
                model=model,
                train_loader=train_loader,
                pos_weight=pos_weight,
                child_parent_pairs=child_parent_pairs,
                lambda_hier=sample_hparams["lambda_hier"],
                device=device,
            )
            print("Smoke test passed. Starting search trials.")

        trial_dir = base_dir / f"trial_{trial:03d}"

        history = fit_function(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            pos_weight=pos_weight.to(device),
            child_parent_pairs=child_parent_pairs.to(device),
            go_terms=go_terms,
            go_aspect=go_aspect,
            obo_path=obo_path,
            train_annotations=train_annotations,
            device=device,
            lambda_hier=sample_hparams["lambda_hier"],
            num_epochs=trial_epochs,
            patience=patience,
            out_dir=trial_dir,
            hparams=sample_hparams,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            wandb_run_name=f"{ablation}_trial_{trial:03d}",
            wandb_config={
                "ablation": ablation,
                "run_type": run_type,
                "trial": trial,
                "go_terms": len(go_terms),
                **sample_hparams,
            },
        )

        record = build_record(trial, history, sample_hparams)
        best_score, best_record = save_and_track_best(
            record=record,
            records=records,
            best_score=best_score,
            best_record=best_record,
            save_path=trial_dir / "best_meta.pt",
            best_save_path=base_dir / "best_meta.pt",
        )

    print(f"\n[Search complete] Best trial score: {best_score:.4f}")
    print(f"Best 5 hparams: {[record['hparams'] for record in records[:top_k_params]]}")

    return records
