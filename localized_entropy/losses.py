from typing import Optional

import numpy as np
import torch


# Custom BCE loss (loop-based, with logits)
def binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    conditions_np: np.ndarray,
    net_worth_np: np.ndarray,
    condition_weights: Optional[np.ndarray] = None,
    nw_threshold: Optional[float] = None,
    nw_multiplier: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    logits_flat = logits.view(-1)
    targets_flat = targets.view(-1)
    conds_flat = np.asarray(conditions_np).reshape(-1)
    nw_flat = np.asarray(net_worth_np).reshape(-1)
    total = torch.zeros((), device=logits.device, dtype=logits.dtype)
    for i in range(logits_flat.shape[0]):
        z = logits_flat[i]
        y = targets_flat[i]
        cond_id = int(conds_flat[i])
        nw_val = float(nw_flat[i])
        if condition_weights is not None:
            try:
                w = float(condition_weights[cond_id])
                if not np.isfinite(w) or w <= 0:
                    w = 1.0
            except Exception:
                w = 1.0
        else:
            w = 1.0
        if (nw_threshold is not None) and (nw_multiplier != 1.0):
            if np.isfinite(nw_val) and (nw_val >= nw_threshold):
                w = w * float(nw_multiplier)
        per_sample = (
            torch.clamp_min(z, 0.0)
            - z * y
            + torch.log1p(torch.exp(-torch.abs(z)))
        )
        total = total + (w * per_sample)
    if reduction == "mean":
        return total / logits_flat.shape[0]
    if reduction == "sum":
        return total
    return total


def localized_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    conditions: torch.Tensor,
    base_rates: Optional[torch.Tensor] = None,
    net_worth: Optional[torch.Tensor] = None,
    condition_weights: Optional[torch.Tensor] = None,
    nw_threshold: Optional[float] = None,
    nw_multiplier: float = 1.0,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Localized Entropy (LE) implementation using PyTorch.

    Mathematics
    -----------
    Let classes be indexed by j = 1..M, with N_j samples in class j,
    labels y_i ∈ {0,1}, predicted probs ŷ_i = σ(z_i) from logits z_i, and
    per-class base rate p_j = mean(y | class j).

    Define cross-entropy over class j:
        CE_j(y, q) = Σ_{i in class j} [ -y_i * log(q_i) - (1 - y_i) * log(1 - q_i) ]

    Localized Entropy:
        LE = ( Σ_{j=1..M}  CE_j(y, ŷ) / CE_j(y, p_j) ) / ( Σ_{j=1..M} N_j )

    - Numerator uses per-sample predictions (stable BCE-with-logits).
    - Denominator uses a constant predictor at the class base rate p_j.
    - We clamp probabilities to [eps, 1-eps] for numerical stability.
    - We divide by total samples Σ_j N_j (not by number of classes).

    Gradient safety
    ---------------
    - All math is in torch; gradients flow through logits z_i in the numerator.
    - Denominator depends on labels only (p_j from y), so it's treated as a
        constant w.r.t. logits, which matches the intended LE definition.

    Parameters
    ----------
    logits:        shape (N,) or (N,1). Raw model outputs z_i.
    targets:       shape (N,). Binary labels {0,1}.
    conditions:    shape (N,). Integer class IDs per sample.
    base_rates:    Optional 1D tensor of length >= max condition id + 1.
                    If provided, uses base_rates[j] as p_j for denominator.
                    Expected on same device/dtype as logits/targets (or will be cast).
    condition_weights:
                    Optional 1D tensor of per-class weights where index is class id.
                    If provided and indexable for a class id, scales that class term.
    reduction:     'mean' (default) returns LE as defined; 'sum' returns LE * N.
    eps:           Small constant for numerical stability.

    Returns
    -------
    Scalar torch.Tensor (loss).
    """
    z = logits.view(-1)
    y = targets.view(-1).to(z.dtype)
    c = conditions.view(-1).to(torch.long)

    bce_per = torch.clamp_min(z, 0) - z * y + torch.log1p(torch.exp(-torch.abs(z)))

    total = z.new_zeros(())
    unique_conds = torch.unique(c)
    N = y.numel()

    for cid in unique_conds:
        mask = (c == cid)
        num = bce_per[mask].sum()
        yj = y[mask]
        n = mask.sum()
        ones = yj.sum()
        zeros = n.to(y.dtype) - ones
        if base_rates is not None:
            idx = cid.item()
            if 0 <= idx < base_rates.numel():
                pj = base_rates[idx].to(y.dtype)
                if not torch.isfinite(pj):
                    pj = ones / n.clamp_min(1)
            else:
                pj = ones / n.clamp_min(1)
        else:
            pj = ones / n.clamp_min(1)
        pj = pj.clamp(eps, 1.0 - eps)

        den = ones * (-torch.log(pj)) + zeros * (-torch.log1p(-pj))
        class_term = num / den.clamp_min(eps)

        if condition_weights is not None:
            idx = cid.item()
            if 0 <= idx < condition_weights.numel():
                w = condition_weights[idx]
                if torch.isfinite(w) and (w > 0):
                    class_term = class_term * w

        total += class_term

    loss = total / max(N, 1)
    if reduction == "sum":
        return loss * N
    return loss
