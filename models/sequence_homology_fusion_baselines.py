"""Fixed and InterLabelGO+-inspired identity fusion sequence/homology models."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.sequence_homology_ablation import _compute_neural_probs, _sequence_homology_model_kwargs
from models.sequence_homology_common import ESMSequenceBranch, SequencePredictionHead


def identity_neural_weight(identity_fraction: torch.Tensor, a: float, k: float) -> torch.Tensor:
    """InterLabelGO-inspired neural weight: a + (1-a) exp(-k*s)."""
    if not 0.0 <= float(a) <= 1.0 or float(k) < 0.0:
        raise ValueError("identity fusion requires 0 <= a <= 1 and k >= 0")
    if not torch.isfinite(identity_fraction).all() or not ((identity_fraction >= 0) & (identity_fraction <= 1)).all():
        raise ValueError("identity fractions must be finite values in [0, 1]")
    return float(a) + (1.0 - float(a)) * torch.exp(-float(k) * identity_fraction)


class _SequenceHomologyFusionBase(nn.Module):
    def __init__(self, num_go_terms: int, **kwargs):
        super().__init__()
        self.seq_branch = ESMSequenceBranch(esm_dim=1280, attn_hidden_dim=kwargs["attn_hidden_dim"], attn_dropout=kwargs["attn_dropout"], out_dim=None)
        self.head = SequencePredictionHead(input_dim=1280, hidden_dim=kwargs["head_hidden_dim"], num_go_terms=num_go_terms, dropout=kwargs["head_dropout"])

    def _branch(self, padded, mask):
        return _compute_neural_probs(self.seq_branch, self.head, padded, mask)


class SequenceHomologyFixedFusionModel(_SequenceHomologyFusionBase):
    """One trainable scalar per aspect; gate/reliability features are ignored."""
    def __init__(self, num_go_terms: int, **kwargs):
        super().__init__(num_go_terms, **kwargs)
        self.fusion_logit = nn.Parameter(torch.zeros(()))

    def forward(self, padded, mask, homology_scores, graph_batch=None, gate_features=None, **unused):
        seq_repr, seq_attn, neural_logits, neural_probs = self._branch(padded, mask)
        homology_scores = homology_scores.to(neural_probs)
        alpha_h = torch.sigmoid(self.fusion_logit)
        alpha_n = 1.0 - alpha_h
        weights = torch.stack((alpha_n, alpha_h)).expand(neural_probs.shape[0], 2)
        probs = alpha_n * neural_probs + alpha_h * homology_scores
        return {"probs": probs, "fused_probs": probs, "neural_probs": neural_probs, "neural_logits": neural_logits, "homology_scores": homology_scores, "gate_weights": weights, "seq_repr": seq_repr, "seq_attn": seq_attn}


class SequenceHomologyIdentityFusionModel(_SequenceHomologyFusionBase):
    """Fresh neural branch fused using fixed, candidate-specific a/k buffers."""
    def __init__(self, num_go_terms: int, a: float, k: float, **kwargs):
        super().__init__(num_go_terms, **kwargs)
        if not 0.0 <= float(a) <= 1.0 or float(k) < 0.0:
            raise ValueError("identity fusion requires 0 <= a <= 1 and k >= 0")
        self.register_buffer("identity_a", torch.tensor(float(a)))
        self.register_buffer("identity_k", torch.tensor(float(k)))

    def forward(self, padded, mask, homology_scores, identity_fraction=None, has_retained_hit=None, graph_batch=None, gate_features=None, **unused):
        if identity_fraction is None or has_retained_hit is None:
            raise ValueError("identity fusion requires identity_fraction and has_retained_hit from a validated sidecar")
        seq_repr, seq_attn, neural_logits, neural_probs = self._branch(padded, mask)
        s = identity_fraction.to(neural_probs).reshape(-1)
        has_hits = has_retained_hit.to(device=neural_probs.device, dtype=torch.bool).reshape(-1)
        if s.shape[0] != neural_probs.shape[0] or has_hits.shape[0] != neural_probs.shape[0]:
            raise ValueError("identity sidecar batch length does not match embeddings")
        if torch.any(~has_hits & (s != 0)):
            raise ValueError("no-retained-hit records must use identity_fraction=0")
        weights_n = identity_neural_weight(s, float(self.identity_a), float(self.identity_k))
        # s=0 gives exactly one neural weight by the documented no-hit rule.
        weights_n = torch.where(has_hits, weights_n, torch.ones_like(weights_n))
        weights_h = 1.0 - weights_n
        homology_scores = homology_scores.to(neural_probs)
        probs = weights_n[:, None] * neural_probs + weights_h[:, None] * homology_scores
        weights = torch.stack((weights_n, weights_h), dim=1)
        return {"probs": probs, "fused_probs": probs, "neural_probs": neural_probs, "neural_logits": neural_logits, "homology_scores": homology_scores, "gate_weights": weights, "identity_fraction": s, "seq_repr": seq_repr, "seq_attn": seq_attn}


def _fixed_optimizer(model, hparams):
    base_lr, wd = float(hparams["learning_rate"]), float(hparams.get("weight_decay", 1e-4))
    scalar = [model.fusion_logit]
    branch = [p for name, p in model.named_parameters() if name != "fusion_logit"]
    return torch.optim.AdamW([{ "params": branch, "lr": base_lr, "weight_decay": wd }, { "params": scalar, "lr": base_lr * float(hparams.get("gate_lr_multiplier", 0.1)), "weight_decay": 0.0 }])


def build_sequence_homology_fixed_fusion_model(hparams, go_terms, device):
    model = SequenceHomologyFixedFusionModel(num_go_terms=len(go_terms), **_sequence_homology_model_kwargs(hparams)).to(device)
    return model, _fixed_optimizer(model, hparams)


def build_sequence_homology_identity_fusion_model(hparams, go_terms, device):
    if "identity_a" not in hparams or "identity_k" not in hparams:
        raise KeyError("identity candidates must contain identity_a and identity_k")
    model = SequenceHomologyIdentityFusionModel(num_go_terms=len(go_terms), a=float(hparams["identity_a"]), k=float(hparams["identity_k"]), **_sequence_homology_model_kwargs(hparams)).to(device)
    return model, torch.optim.AdamW(model.parameters(), lr=float(hparams["learning_rate"]), weight_decay=float(hparams.get("weight_decay", 1e-4)))
