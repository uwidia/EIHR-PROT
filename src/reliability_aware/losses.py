import torch

def run_one_batch_smoke_test(
    model,
    train_loader,
    optimizer,
    pos_weight,
    child_parent_pairs,
    lambda_hier,
    device,
):
    model.train()

    batch = next(iter(train_loader))

    (
        padded,
        mask,
        graph_batch,
        homology_priors,
        gate_features,
        targets,
        global_indices,
        labels,
    ) = batch

    padded = padded.to(device)
    mask = mask.to(device)
    graph_batch = graph_batch.to(device)
    homology_priors = homology_priors.to(device)
    gate_features = gate_features.to(device)
    targets = targets.to(device)
    pos_weight = pos_weight.to(device)
    child_parent_pairs = child_parent_pairs.to(device)

    optimizer.zero_grad(set_to_none=True)

    outputs = model(
        padded=padded,
        mask=mask,
        graph_batch=graph_batch,
        homology_scores=homology_priors,
        gate_features=gate_features,
    )

    fused_probs = outputs["fused_probs"]

    assert fused_probs.shape == targets.shape, (
        f"Shape mismatch: fused_probs={fused_probs.shape}, targets={targets.shape}"
    )

    assert homology_priors.shape == targets.shape, (
        f"Shape mismatch: homology_priors={homology_priors.shape}, targets={targets.shape}"
    )

    assert gate_features.shape[1] == 8, (
        f"Expected gate_features shape (B, 8), got {gate_features.shape}"
    )

    bce = weighted_bce_on_probs(
        probs=fused_probs,
        targets=targets,
        pos_weight=pos_weight,
    )

    hier = hierarchy_loss(
        fused_probs=fused_probs,
        child_parent_pairs=child_parent_pairs,
    )

    loss = bce + lambda_hier * hier

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

    loss.backward()
    optimizer.step()

    print("Smoke test passed")
    print(f"batch_size: {targets.shape[0]}")
    print(f"num_go_terms: {targets.shape[1]}")
    print(f"bce_loss: {bce.item():.6f}")
    print(f"hier_loss: {hier.item():.6f}")
    print(f"total_loss: {loss.item():.6f}")
    print(f"fused_probs range: {fused_probs.min().item():.6f} to {fused_probs.max().item():.6f}")


def compute_pos_weight_from_label_indices(
    label_to_indices: dict[str, list[int]],
    num_go_terms: int,
    train_ids: set[str],
    cap: float = 20.0,
) -> torch.Tensor:
    """
    Computes term-specific positive weights from training labels only.

    Use train_ids = keep_ids_for_aspect if you filtered empty-label proteins.
    """
    pos_counts = torch.zeros(num_go_terms, dtype=torch.float32)

    valid_train_ids = [label for label in train_ids if label in label_to_indices]
    n_train = len(valid_train_ids)

    if n_train == 0:
        raise ValueError("No valid training IDs found for pos_weight calculation.")

    for label in valid_train_ids:
        idxs = label_to_indices[label]
        if idxs:
            pos_counts[idxs] += 1.0

    pos_weight = (n_train - pos_counts) / (pos_counts + 1.0)
    pos_weight = torch.clamp(pos_weight, max=cap)

    return pos_weight

def weighted_bce_on_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    probs:   (B, C), already sigmoid/probability-space
    targets: (B, C), multi-hot labels
    """
    probs = probs.clamp(eps, 1.0 - eps)
    pos_weight = pos_weight.to(device=probs.device, dtype=probs.dtype)

    loss_pos = -pos_weight * targets * torch.log(probs)
    loss_neg = -(1.0 - targets) * torch.log(1.0 - probs)

    return (loss_pos + loss_neg).mean()

def hierarchy_loss(fused_probs, child_parent_pairs):
    if child_parent_pairs.numel() == 0:
        return fused_probs.new_tensor(0.0)

    child_idx = child_parent_pairs[:, 0]
    parent_idx = child_parent_pairs[:, 1]

    child_probs = fused_probs[:, child_idx]
    parent_probs = fused_probs[:, parent_idx]

    return torch.relu(child_probs - parent_probs).mean()

    