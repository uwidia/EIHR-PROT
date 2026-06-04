from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

from models.sequence_homology_common import (
    ESMSequenceBranch,
    SequenceHomologyShardDataset,
    SequencePredictionHead,
    make_sequence_homology_collate_fn,
)
from reliability_aware.utils.losses import hierarchy_loss, weighted_bce_on_probs

logger = logging.getLogger(__name__)


def initialize_gate_to_balanced(final_layer: nn.Linear) -> None:
    nn.init.zeros_(final_layer.weight)
    nn.init.zeros_(final_layer.bias)
    logger.info("Initialized gate final layer for near 50/50 neural/homology fusion.")


def _compute_neural_probs(seq_branch, head, padded, mask):
    seq_repr, seq_attn = seq_branch(padded, mask)
    neural_logits = head(seq_repr)
    neural_probs = torch.sigmoid(neural_logits)
    return seq_repr, seq_attn, neural_logits, neural_probs


class SequenceHomologyInternalGateModel(nn.Module):
    def __init__(
        self,
        num_go_terms: int,
        attn_hidden_dim: int = 256,
        attn_dropout: float = 0.1,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_branch = ESMSequenceBranch(
            esm_dim=1280,
            attn_hidden_dim=attn_hidden_dim,
            attn_dropout=attn_dropout,
            out_dim=None,
        )
        self.head = SequencePredictionHead(
            input_dim=1280,
            hidden_dim=head_hidden_dim,
            num_go_terms=num_go_terms,
            dropout=head_dropout,
        )
        self.gate = nn.Sequential(
            nn.Linear(1280, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        # Start the gate at neutral 50/50 fusion before training.
        initialize_gate_to_balanced(self.gate[-2])

    def forward(
        self,
        padded: torch.Tensor,
        mask: torch.Tensor,
        homology_scores: torch.Tensor,
        graph_batch=None,
        gate_features=None,
    ):
        seq_repr, seq_attn, neural_logits, neural_probs = _compute_neural_probs(
            self.seq_branch,
            self.head,
            padded,
            mask,
        )
        homology_scores = homology_scores.to(
            device=neural_probs.device,
            dtype=neural_probs.dtype,
        )
        gate_weights = self.gate(seq_repr)
        alpha_n = gate_weights[:, 0].unsqueeze(-1)
        alpha_h = gate_weights[:, 1].unsqueeze(-1)
        fused_probs = alpha_n * neural_probs + alpha_h * homology_scores

        return {
            "probs": fused_probs,
            "fused_probs": fused_probs,
            "neural_probs": neural_probs,
            "neural_logits": neural_logits,
            "homology_scores": homology_scores,
            "gate_weights": gate_weights,
            "seq_repr": seq_repr,
            "seq_attn": seq_attn,
        }


class SequenceHomologyConfidenceGateModel(nn.Module):
    def __init__(
        self,
        num_go_terms: int,
        attn_hidden_dim: int = 256,
        attn_dropout: float = 0.1,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_branch = ESMSequenceBranch(
            esm_dim=1280,
            attn_hidden_dim=attn_hidden_dim,
            attn_dropout=attn_dropout,
            out_dim=None,
        )
        self.head = SequencePredictionHead(
            input_dim=1280,
            hidden_dim=head_hidden_dim,
            num_go_terms=num_go_terms,
            dropout=head_dropout,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        # Start the gate at neutral 50/50 fusion before training.
        initialize_gate_to_balanced(self.gate[-2])

    def forward(
        self,
        padded: torch.Tensor,
        mask: torch.Tensor,
        homology_scores: torch.Tensor,
        gate_features: torch.Tensor,
        graph_batch=None,
    ):
        seq_repr, seq_attn, neural_logits, neural_probs = _compute_neural_probs(
            self.seq_branch,
            self.head,
            padded,
            mask,
        )
        homology_scores = homology_scores.to(
            device=neural_probs.device,
            dtype=neural_probs.dtype,
        )
        gate_features = gate_features.to(
            device=neural_probs.device,
            dtype=neural_probs.dtype,
        )
        if gate_features.ndim != 2 or gate_features.shape[1] != 4:
            raise ValueError(
                "gate_features must have shape (B, 4) with columns "
                "[b_max, cov_max, log1p_n_hits, has_hit]; "
                f"got {tuple(gate_features.shape)}"
            )

        gate_weights = self.gate(gate_features)
        alpha_n = gate_weights[:, 0].unsqueeze(-1)
        alpha_h = gate_weights[:, 1].unsqueeze(-1)
        fused_probs = alpha_n * neural_probs + alpha_h * homology_scores

        return {
            "probs": fused_probs,
            "fused_probs": fused_probs,
            "neural_probs": neural_probs,
            "neural_logits": neural_logits,
            "homology_scores": homology_scores,
            "gate_weights": gate_weights,
            "seq_repr": seq_repr,
            "seq_attn": seq_attn,
        }


def _sequence_homology_model_kwargs(sample_hparams: dict) -> dict:
    dropout = float(sample_hparams.get("dropout", 0.2))
    return {
        "attn_hidden_dim": int(sample_hparams.get("attn_hidden_dim", 256)),
        "attn_dropout": float(sample_hparams.get("attn_dropout", dropout)),
        "head_hidden_dim": int(sample_hparams.get("head_hidden_dim", 512)),
        "head_dropout": float(sample_hparams.get("head_dropout", dropout)),
    }


def _build_optimizer(model, sample_hparams: dict, *, use_gate_lr: bool = False):
    base_lr = float(sample_hparams["learning_rate"])
    weight_decay = float(sample_hparams.get("weight_decay", 1e-4))

    if use_gate_lr:
        gate_lr_multiplier = float(sample_hparams.get("gate_lr_multiplier", 0.1))
        gate_lr = base_lr * gate_lr_multiplier
        non_gate_params = [
            param
            for name, param in model.named_parameters()
            if not name.startswith("gate.")
        ]
        gate_params = list(model.gate.parameters())

        logger.info("Using base_lr=%s and gate_lr=%s", base_lr, gate_lr)
        return torch.optim.AdamW(
            [
                {
                    "params": non_gate_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": gate_params,
                    "lr": gate_lr,
                    "weight_decay": weight_decay,
                },
            ]
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )


def build_sequence_homology_internal_gate_model(sample_hparams, go_terms, device):
    dropout = float(sample_hparams.get("dropout", 0.2))
    model = SequenceHomologyInternalGateModel(
        num_go_terms=len(go_terms),
        **_sequence_homology_model_kwargs(sample_hparams),
        gate_hidden_dim=int(sample_hparams.get("gate_hidden_dim", 128)),
        gate_dropout=float(sample_hparams.get("gate_dropout", dropout)),
    ).to(device)
    return model, _build_optimizer(model, sample_hparams, use_gate_lr=True)


def build_sequence_homology_confidence_gate_model(sample_hparams, go_terms, device):
    dropout = float(sample_hparams.get("dropout", 0.2))
    model = SequenceHomologyConfidenceGateModel(
        num_go_terms=len(go_terms),
        **_sequence_homology_model_kwargs(sample_hparams),
        gate_hidden_dim=int(sample_hparams.get("gate_hidden_dim", 128)),
        gate_dropout=float(sample_hparams.get("gate_dropout", dropout)),
    ).to(device)
    return model, _build_optimizer(model, sample_hparams, use_gate_lr=True)


def _assert_initial_gate_is_balanced(model, kwargs, batch_size, device):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(**kwargs)
        mean_gate = outputs["gate_weights"].mean(dim=0)
        expected = torch.full(
            (2,),
            0.5,
            device=mean_gate.device,
            dtype=mean_gate.dtype,
        )
        assert torch.allclose(mean_gate, expected, atol=0.05), (
            "Expected near-balanced initial gate weights, "
            f"got {mean_gate.detach().cpu().tolist()}"
        )
        assert outputs["gate_weights"].shape == (batch_size, 2), (
            f"Gate shape mismatch: gate_weights={outputs['gate_weights'].shape}, "
            f"expected={(batch_size, 2)}"
        )
    if was_training:
        model.train()


def _assert_sequence_homology_smoke_outputs(outputs, homology_scores, targets, device):
    probs = outputs["probs"]
    gate_weights = outputs["gate_weights"]

    assert (
        probs.shape == targets.shape
    ), f"Shape mismatch: probs={probs.shape}, targets={targets.shape}"
    assert outputs["neural_probs"].shape == targets.shape, (
        f"Shape mismatch: neural_probs={outputs['neural_probs'].shape}, "
        f"targets={targets.shape}"
    )
    assert homology_scores.shape == targets.shape, (
        f"Shape mismatch: homology_scores={homology_scores.shape}, "
        f"targets={targets.shape}"
    )
    assert gate_weights.shape == (targets.shape[0], 2), (
        f"Gate shape mismatch: gate_weights={gate_weights.shape}, "
        f"expected={(targets.shape[0], 2)}"
    )
    assert torch.allclose(
        gate_weights.sum(dim=1),
        torch.ones(targets.shape[0], device=device, dtype=gate_weights.dtype),
        atol=1e-5,
    ), "Gate weights should sum to 1 for each sample"
    assert torch.isfinite(gate_weights).all(), "Gate weights must be finite"
    assert (gate_weights >= 0).all() and (
        gate_weights <= 1
    ).all(), "Gate weights must be between 0 and 1"


def _run_sequence_homology_smoke_test(
    model,
    train_loader,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    device,
    *,
    use_gate_features: bool,
    check_balanced_initial_gate: bool = False,
):
    model_copy = copy.deepcopy(model)
    model_copy.train()
    optimizer_copy = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)

    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    batch = next(iter(train_loader))
    padded = batch["padded"].to(device)
    mask = batch["mask"].to(device)
    homology_scores = batch["homology_scores"].to(device)
    targets = batch["targets"].to(device)

    optimizer_copy.zero_grad(set_to_none=True)

    kwargs = {
        "padded": padded,
        "mask": mask,
        "homology_scores": homology_scores,
    }
    if use_gate_features:
        kwargs["gate_features"] = batch["gate_features"].to(device)

    if check_balanced_initial_gate:
        _assert_initial_gate_is_balanced(
            model=model_copy,
            kwargs=kwargs,
            batch_size=targets.shape[0],
            device=device,
        )

    outputs = model_copy(**kwargs)
    probs = outputs["probs"]

    _assert_sequence_homology_smoke_outputs(
        outputs=outputs,
        homology_scores=homology_scores,
        targets=targets,
        device=device,
    )

    bce = weighted_bce_on_probs(
        probs=probs,
        targets=targets,
        pos_weight=pos_weight,
    )
    hier = hierarchy_loss(
        fused_probs=probs,
        child_parent_pairs=child_parent_pairs,
    )
    loss = bce + lambda_hier * hier

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

    loss.backward()
    optimizer_copy.step()

    print("Smoke test passed")
    print(f"batch_size: {targets.shape[0]}")
    print(f"num_go_terms: {targets.shape[1]}")
    print(f"bce_loss: {bce.item():.6f}")
    print(f"hier_loss: {hier.item():.6f}")
    print(f"total_loss: {loss.item():.6f}")
    print(f"probs range: {probs.min().item():.6f} to {probs.max().item():.6f}")
    print(
        "mean gate weights: "
        f"neural={outputs['gate_weights'][:, 0].mean().item():.4f}, "
        f"homology={outputs['gate_weights'][:, 1].mean().item():.4f}"
    )


def run_one_batch_smoke_test_sequence_homology_internal_gate(
    model,
    train_loader,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    device,
):
    _run_sequence_homology_smoke_test(
        model=model,
        train_loader=train_loader,
        pos_weight=pos_weight,
        child_parent_pairs=child_parent_pairs,
        lambda_hier=lambda_hier,
        device=device,
        use_gate_features=False,
        check_balanced_initial_gate=True,
    )


def run_one_batch_smoke_test_sequence_homology_confidence_gate(
    model,
    train_loader,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    device,
):
    _run_sequence_homology_smoke_test(
        model=model,
        train_loader=train_loader,
        pos_weight=pos_weight,
        child_parent_pairs=child_parent_pairs,
        lambda_hier=lambda_hier,
        device=device,
        use_gate_features=True,
        check_balanced_initial_gate=True,
    )


__all__ = [
    "SequenceHomologyShardDataset",
    "make_sequence_homology_collate_fn",
    "SequenceHomologyInternalGateModel",
    "SequenceHomologyConfidenceGateModel",
    "build_sequence_homology_internal_gate_model",
    "build_sequence_homology_confidence_gate_model",
    "run_one_batch_smoke_test_sequence_homology_internal_gate",
    "run_one_batch_smoke_test_sequence_homology_confidence_gate",
]
