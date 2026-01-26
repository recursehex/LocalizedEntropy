from typing import Optional

import numpy as np
import torch


# Custom BCE loss (loop-based, with logits)
def binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    conditions_np: np.ndarray,
    condition_weights: Optional[np.ndarray] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute a loop-based BCE loss with optional condition weighting."""
    logits_flat = logits.view(-1)
    targets_flat = targets.view(-1)
    conds_flat = np.asarray(conditions_np).reshape(-1)
    total = torch.zeros((), device=logits.device, dtype=logits.dtype)
    for i in range(logits_flat.shape[0]):
        z = logits_flat[i]
        y = targets_flat[i]
        cond_id = int(conds_flat[i])
        if condition_weights is not None:
            try:
                w = float(condition_weights[cond_id])
                if not np.isfinite(w) or w <= 0:
                    w = 1.0
            except Exception:
                w = 1.0
        else:
            w = 1.0
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
    condition_weights: Optional[torch.Tensor] = None,
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
    z = logits.view(-1)  # Flatten logits to 1D so each sample has one logit.
    y = targets.view(-1).to(z.dtype)  # Flatten labels and match dtype for math.
    c = conditions.view(-1).to(torch.long)  # Flatten class ids for grouping.

    # Per-sample stable BCE-with-logits for the numerator CE_j(y, yhat).
    bce_per = torch.clamp_min(z, 0) - z * y + torch.log1p(torch.exp(-torch.abs(z)))

    total = z.new_zeros(())  # Accumulator for sum_j CE_j(y, yhat) / CE_j(y, p_j).
    unique_conds = torch.unique(c)  # Iterate over observed classes only.
    N = y.numel()  # Total samples for final normalization by sum_j N_j.

    for cid in unique_conds:
        mask = (c == cid)  # Select samples in this class j.
        num = bce_per[mask].sum()  # Numerator CE_j(y, yhat) over class j.
        yj = y[mask]  # Labels for class j to compute base rate p_j.
        n = mask.sum()  # N_j: number of samples in this class.
        ones = yj.sum()  # Count of positives in class j.
        zeros = n.to(y.dtype) - ones  # Count of negatives in class j.
        if base_rates is not None:
            idx = cid.item()
            if 0 <= idx < base_rates.numel():
                pj = base_rates[idx].to(y.dtype)  # Use provided p_j when available.
                if not torch.isfinite(pj):
                    print(f"[localized_entropy] base_rates[{idx}] is invalid; using empirical p_j for class {idx}.")
                    pj = ones / n.clamp_min(1)  # Fallback to empirical rate if invalid.
            else:
                print(f"[localized_entropy] base_rates missing for class {cid.item()}; using empirical p_j.")
                pj = ones / n.clamp_min(1)  # Fallback to empirical rate if index missing.
        else:
            print(f"[localized_entropy] base_rates not provided; using empirical p_j for class {cid.item()}.")
            pj = ones / n.clamp_min(1)  # Empirical base rate p_j from labels.
        pj = pj.clamp(eps, 1.0 - eps)  # Clamp to avoid log(0) in denominator.

        # Denominator CE_j(y, p_j) for constant predictor at base rate.
        den = ones * (-torch.log(pj)) + zeros * (-torch.log1p(-pj))
        class_term = num / den.clamp_min(eps)  # Normalize by base-rate CE_j.

        if condition_weights is not None:
            idx = cid.item()
            if 0 <= idx < condition_weights.numel():
                w = condition_weights[idx]
                if torch.isfinite(w) and (w > 0):
                    class_term = class_term * w  # Optional per-class scaling.

        total += class_term  # Sum normalized class terms across all classes.

    loss = total / max(N, 1)  # Final LE: average by total samples sum_j N_j.
    if reduction == "sum":
        return loss * N  # Return un-averaged total when requested.
    return loss
