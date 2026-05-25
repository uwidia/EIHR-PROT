import torch


@torch.no_grad()
def fmax_score(y_true: torch.Tensor, y_prob: torch.Tensor, steps: int = 101):
    y_true = y_true.bool()
    y_prob = y_prob.float()
    thresholds = torch.linspace(0.0, 1.0, steps=steps, device=y_prob.device)

    best = {"Fmax": 0.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
    eps = 1e-12

    for t in thresholds:
        y_pred = y_prob >= t
        tp = (y_pred & y_true).sum().float()
        fp = (y_pred & ~y_true).sum().float()
        fn = (~y_pred & y_true).sum().float()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)

        if f1.item() > best["Fmax"]:
            best = {
                "Fmax": f1.item(),
                "threshold": t.item(),
                "precision": precision.item(),
                "recall": recall.item(),
            }

    return best


import torch


@torch.no_grad()
def smin_score(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    ic: torch.Tensor,
    steps: int = 101,
):
    """
    Returns both:
      - raw Smin: standard CAFA-style semantic distance
      - normalized Smin: RU/MI normalized by each protein's true IC sum
    """
    y_true = y_true.bool()
    y_prob = y_prob.float()
    ic = ic.to(device=y_prob.device, dtype=y_prob.dtype)

    thresholds = torch.linspace(0.0, 1.0, steps=steps, device=y_prob.device)

    best_raw = {"Smin": float("inf"), "threshold": 0.5, "ru": 0.0, "mi": 0.0}
    best_norm = {"Smin": float("inf"), "threshold": 0.5, "ru": 0.0, "mi": 0.0}

    n = max(y_true.shape[0], 1)
    eps = 1e-12

    true_ic_sum = (y_true.float() * ic).sum(dim=1)  # (B,)

    for t in thresholds:
        y_pred = y_prob >= t

        miss = y_true & ~y_pred
        wrong = y_pred & ~y_true

        # raw RU / MI
        ru = (miss.float() * ic).sum() / n
        mi = (wrong.float() * ic).sum() / n
        raw_s = torch.sqrt(ru * ru + mi * mi + eps)

        # normalized RU / MI per protein
        ru_per_protein = (miss.float() * ic).sum(dim=1)
        mi_per_protein = (wrong.float() * ic).sum(dim=1)

        denom = true_ic_sum.clamp_min(eps)
        ru_norm = (ru_per_protein / denom).mean()
        mi_norm = (mi_per_protein / denom).mean()
        norm_s = torch.sqrt(ru_norm * ru_norm + mi_norm * mi_norm + eps)

        if raw_s.item() < best_raw["Smin"]:
            best_raw = {
                "Smin": raw_s.item(),
                "threshold": t.item(),
                "ru": ru.item(),
                "mi": mi.item(),
            }

        if norm_s.item() < best_norm["Smin"]:
            best_norm = {
                "Smin": norm_s.item(),
                "threshold": t.item(),
                "ru": ru_norm.item(),
                "mi": mi_norm.item(),
            }

    return {
        "raw": best_raw,
        "normalized": best_norm,
    }
