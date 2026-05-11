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


@torch.no_grad()
def smin_score(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    ic: torch.Tensor,
    steps: int = 101,
):
    y_true = y_true.bool()
    y_prob = y_prob.float()
    ic = ic.to(device=y_prob.device, dtype=y_prob.dtype)
    thresholds = torch.linspace(0.0, 1.0, steps=steps, device=y_prob.device)

    best = {"Smin": float("inf"), "threshold": 0.5, "ru": 0.0, "mi": 0.0}
    n = max(y_true.shape[0], 1)
    eps = 1e-12

    for t in thresholds:
        y_pred = y_prob >= t
        miss = y_true & ~y_pred
        wrong = y_pred & ~y_true

        ru = (miss.float() * ic).sum() / n
        mi = (wrong.float() * ic).sum() / n
        s = torch.sqrt(ru * ru + mi * mi + eps)

        if s.item() < best["Smin"]:
            best = {
                "Smin": s.item(),
                "threshold": t.item(),
                "ru": ru.item(),
                "mi": mi.item(),
            }
