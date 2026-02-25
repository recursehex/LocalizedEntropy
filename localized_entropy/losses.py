import math
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F


def _summarize_tensor(name: str, tensor: torch.Tensor, max_items: int = 8) -> str:
    """Return a compact debug summary for a tensor."""
    t = tensor.detach()
    shape = tuple(t.shape)
    dtype = t.dtype
    device = t.device
    numel = t.numel()
    if numel == 0:
        stats = "empty"
        sample = []
    else:
        t_float = t if torch.is_floating_point(t) else t.to(torch.float32)
        stats = (
            f"min={t_float.min().item():.6g} "
            f"max={t_float.max().item():.6g} "
            f"mean={t_float.mean().item():.6g}"
        )
        sample = t.view(-1)[:max_items].detach().cpu().tolist()
    return (
        f"{name}: shape={shape} dtype={dtype} device={device} "
        f"numel={numel} {stats} sample={sample}"
    )


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


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: Optional[float] = 0.25,
    gamma: Optional[float] = 2.0,
    sample_weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute binary focal loss from logits with optional sample weighting."""
    z = logits.view(-1)
    y = targets.view(-1).to(dtype=z.dtype)
    if not torch.is_floating_point(z):
        raise TypeError("focal_loss_with_logits expects floating-point logits.")
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    # Keep p_t away from exactly 0/1 to avoid unstable gradients when gamma < 1.
    dtype_eps = float(torch.finfo(z.dtype).eps)
    pt_eps = max(float(eps), dtype_eps)
    p = torch.sigmoid(z)
    pt = (p * y) + ((1.0 - p) * (1.0 - y))
    pt = pt.clamp(pt_eps, 1.0 - pt_eps)
    gamma_value = 2.0 if gamma is None else float(gamma)
    modulator = (1.0 - pt).clamp_min(pt_eps).pow(gamma_value)
    loss = modulator * bce
    if alpha is not None:
        alpha_value = float(alpha)
        alpha_t = (alpha_value * y) + ((1.0 - alpha_value) * (1.0 - y))
        loss = loss * alpha_t
    denom = None
    if sample_weights is not None:
        w = sample_weights.view(-1).to(dtype=z.dtype, device=z.device)
        loss = loss * w
        denom = w.sum().clamp_min(1.0)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if denom is None:
            denom = torch.tensor(loss.numel(), dtype=z.dtype, device=z.device).clamp_min(1.0)
        return loss.sum() / denom
    return loss


class CrossBatchHistory:
    """Track recent examples per condition with a moving window."""

    def __init__(self, amplification_rate: float, eps: float = 1e-12):
        if amplification_rate <= 0:
            raise ValueError("amplification_rate must be > 0 for cross-batch history.")
        self.amplification_rate = float(amplification_rate)
        self.eps = float(eps)
        self._buffers: Dict[int, deque] = {}
        self._capacity: Dict[int, int] = {}
        self._count_pos: Dict[int, int] = {}
        self._count_neg: Dict[int, int] = {}
        self._sum_pos: Dict[int, float] = {}
        self._sum_neg: Dict[int, float] = {}
        self._sum_num: Dict[int, float] = {}

    def _capacity_for_condition(self, base_rate: float) -> int:
        rate = base_rate
        if not math.isfinite(rate) or rate <= 0:
            rate = self.eps
        capacity = int(math.ceil(self.amplification_rate / max(rate, self.eps)))
        return max(1, capacity)

    def _ensure_condition(self, condition_id: int) -> None:
        if condition_id not in self._buffers:
            self._buffers[condition_id] = deque()
            self._capacity[condition_id] = 1
            self._count_pos[condition_id] = 0
            self._count_neg[condition_id] = 0
            self._sum_pos[condition_id] = 0.0
            self._sum_neg[condition_id] = 0.0
            self._sum_num[condition_id] = 0.0

    def _set_capacity(self, condition_id: int, capacity: int) -> None:
        self._ensure_condition(condition_id)
        capacity = max(1, int(capacity))
        self._capacity[condition_id] = capacity
        buf = self._buffers[condition_id]
        while len(buf) > capacity:
            label, weight, num_value = buf.popleft()
            if label == 1:
                self._count_pos[condition_id] -= 1
                self._sum_pos[condition_id] -= float(weight)
            else:
                self._count_neg[condition_id] -= 1
                self._sum_neg[condition_id] -= float(weight)
            self._sum_num[condition_id] -= float(num_value)

    def _append(self, condition_id: int, label: int, weight: float, num_value: float) -> None:
        self._ensure_condition(condition_id)
        buf = self._buffers[condition_id]
        capacity = self._capacity[condition_id]
        while len(buf) >= capacity:
            old_label, old_weight, old_num = buf.popleft()
            if old_label == 1:
                self._count_pos[condition_id] -= 1
                self._sum_pos[condition_id] -= float(old_weight)
            else:
                self._count_neg[condition_id] -= 1
                self._sum_neg[condition_id] -= float(old_weight)
            self._sum_num[condition_id] -= float(old_num)
        buf.append((label, float(weight), float(num_value)))
        if label == 1:
            self._count_pos[condition_id] += 1
            self._sum_pos[condition_id] += float(weight)
        else:
            self._count_neg[condition_id] += 1
            self._sum_neg[condition_id] += float(weight)
        self._sum_num[condition_id] += float(num_value)

    def update(
        self,
        conditions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        base_rates: Optional[torch.Tensor] = None,
        num_values: Optional[torch.Tensor] = None,
    ) -> None:
        """Update history buffers using a batch of conditions/labels and CE numerators."""
        c = conditions.view(-1).detach().cpu().numpy()
        y = targets.view(-1).detach().cpu().numpy()
        w = None
        if sample_weights is not None:
            w = sample_weights.view(-1).detach().cpu().numpy()
        num_np = None
        if num_values is not None:
            num_np = num_values.view(-1).detach().cpu().numpy()

        base_rates_np = None
        if base_rates is not None:
            base_rates_np = base_rates.detach().cpu().numpy()

        batch_weight_sums: Dict[int, Dict[int, float]] = {}
        for idx in range(c.shape[0]):
            cid = int(c[idx])
            label = 1 if float(y[idx]) >= 0.5 else 0
            weight = float(w[idx]) if w is not None else 1.0
            if cid not in batch_weight_sums:
                batch_weight_sums[cid] = {0: 0.0, 1: 0.0}
            batch_weight_sums[cid][label] += weight

        for cid, counts in batch_weight_sums.items():
            base_rate = None
            if base_rates_np is not None and 0 <= cid < base_rates_np.shape[0]:
                candidate = float(base_rates_np[cid])
                if math.isfinite(candidate) and 0.0 < candidate < 1.0:
                    base_rate = candidate
            if base_rate is None:
                self._ensure_condition(cid)
                ones_hist = self._sum_pos[cid]
                zeros_hist = self._sum_neg[cid]
                total_hist = ones_hist + zeros_hist
                if total_hist > 0:
                    base_rate = ones_hist / total_hist
                else:
                    total_batch = counts[0] + counts[1]
                    base_rate = (counts[1] / total_batch) if total_batch > 0 else 0.5

            capacity = self._capacity_for_condition(base_rate)
            self._set_capacity(cid, capacity)

        for idx in range(c.shape[0]):
            cid = int(c[idx])
            label = 1 if float(y[idx]) >= 0.5 else 0
            weight = float(w[idx]) if w is not None else 1.0
            num_value = float(num_np[idx]) if num_np is not None else 0.0
            if not math.isfinite(num_value):
                num_value = 0.0
            self._append(cid, label, weight, num_value)

    def stats(self, condition_id: int) -> Optional[Tuple[int, int, int, int, float, float, float]]:
        """Return (capacity, kept_total, count_pos, count_neg, sum_pos, sum_neg, sum_num)."""
        if condition_id not in self._capacity:
            return None
        capacity = self._capacity[condition_id]
        count_pos = self._count_pos[condition_id]
        count_neg = self._count_neg[condition_id]
        kept_total = len(self._buffers[condition_id])
        sum_pos = float(self._sum_pos[condition_id])
        sum_neg = float(self._sum_neg[condition_id])
        sum_num = float(self._sum_num[condition_id])
        return capacity, kept_total, count_pos, count_neg, sum_pos, sum_neg, sum_num


def localized_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    conditions: torch.Tensor,
    base_rates: Optional[torch.Tensor] = None,
    condition_weights: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-12,
    debug: bool = False,
    cross_batch_history: Optional[CrossBatchHistory] = None,
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
        If sample weights w_i are provided, N_j and the per-class CE terms
        are computed with weighted counts, and the denominator becomes Σ_i w_i.

    - Numerator uses per-sample predictions (stable BCE-with-logits).
    - Denominator uses a constant predictor at the class base rate p_j.
    - We clamp probabilities to [eps, 1-eps] for numerical stability.
    - We divide by total samples Σ_j N_j (not by number of classes), or by total
        weight Σ_i w_i when sample weights are provided.

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
    sample_weights:
                    Optional 1D tensor of per-sample weights (same length as logits).
                    If provided, each sample's contribution is scaled and totals are
                    normalized by the summed weights.
    reduction:     'mean' (default) returns LE as defined; 'sum' returns LE * total weight.
    eps:           Small constant for numerical stability.

    debug:        When True, prints a compact summary of LE inputs.
    cross_batch_history:
                    Optional CrossBatchHistory instance to smooth label counts
                    and numerator CE sums across batches via a moving window
                    per condition.

    Returns
    -------
    Scalar torch.Tensor (loss).
    """
    if debug:
        print("[localized_entropy][debug] input summary")
        print(_summarize_tensor("logits", logits))
        print(_summarize_tensor("targets", targets))
        print(_summarize_tensor("conditions", conditions))
        if base_rates is None:
            print("base_rates: None")
        else:
            print(_summarize_tensor("base_rates", base_rates))
        if condition_weights is None:
            print("condition_weights: None")
        else:
            print(_summarize_tensor("condition_weights", condition_weights))
        if sample_weights is None:
            print("sample_weights: None")
        else:
            print(_summarize_tensor("sample_weights", sample_weights))

    z = logits.view(-1)  # Flatten logits to 1D so each sample has one logit.
    y = targets.view(-1).to(z.dtype)  # Flatten labels and match dtype for math.
    c = conditions.view(-1).to(torch.long)  # Flatten class ids for grouping.

    # Per-sample stable BCE-with-logits for the numerator CE_j(y, yhat).
    bce_per = torch.clamp_min(z, 0) - z * y + torch.log1p(torch.exp(-torch.abs(z)))
    sample_w = None
    if sample_weights is not None:
        sample_w = sample_weights.view(-1).to(device=z.device, dtype=z.dtype)
        bce_per = bce_per * sample_w

    total = z.new_zeros(())  # Accumulator for sum_j CE_j(y, yhat) / CE_j(y, p_j).
    unique_conds = torch.unique(c)  # Iterate over observed classes only.
    total_weight = sample_w.sum() if sample_w is not None else y.numel()

    for cid in unique_conds:
        mask = (c == cid)  # Select samples in this class j.
        batch_num = bce_per[mask].sum()  # Current-batch CE_j(y, yhat) over class j.
        num = batch_num
        yj = y[mask]  # Labels for class j to compute base rate p_j.
        if sample_w is not None:
            wj = sample_w[mask]
            batch_n = wj.sum()
            batch_ones = (wj * yj).sum()
            batch_zeros = batch_n - batch_ones
        else:
            batch_n = mask.sum()  # N_j: number of samples in this class.
            batch_ones = yj.sum()  # Count of positives in class j.
            batch_zeros = batch_n.to(y.dtype) - batch_ones  # Count of negatives in class j.

        ones = batch_ones
        zeros = batch_zeros
        if cross_batch_history is not None:
            hist_stats = cross_batch_history.stats(cid.item())
            if hist_stats is not None:
                capacity, kept_total, count_pos, count_neg, sum_pos, sum_neg, sum_num = hist_stats
                ones = ones + z.new_tensor(sum_pos)
                zeros = zeros + z.new_tensor(sum_neg)
                num = num + z.new_tensor(sum_num)
                if debug:
                    total_hist = sum_pos + sum_neg
                    rate_hist = (sum_pos / total_hist) if total_hist > 0 else float("nan")
                    print(
                        "[localized_entropy][debug][history] "
                        f"condition={cid.item()} "
                        f"n={capacity} "
                        f"kept_total={kept_total} "
                        f"kept_pos={count_pos} kept_neg={count_neg} "
                        f"sum_num={sum_num:.6g} "
                        f"label1_rate={rate_hist:.6g}"
                    )
        n = ones + zeros
        if base_rates is not None:
            idx = cid.item()
            if 0 <= idx < base_rates.numel():
                pj = base_rates[idx].to(y.dtype)  # Use provided p_j when available.
                if not torch.isfinite(pj):
                    if debug:
                        print(
                            f"[localized_entropy][debug] base_rates[{idx}] is invalid; "
                            f"using empirical p_j for class {idx}."
                        )
                    pj = ones / n.clamp_min(1)  # Fallback to empirical rate if invalid.
            else:
                if debug:
                    print(
                        f"[localized_entropy][debug] base_rates missing for class "
                        f"{cid.item()}; using empirical p_j."
                    )
                pj = ones / n.clamp_min(1)  # Fallback to empirical rate if index missing.
        else:
            if debug:
                print(
                    f"[localized_entropy][debug] base_rates not provided; "
                    f"using empirical p_j for class {cid.item()}."
                )
            pj = ones / n.clamp_min(1)  # Empirical base rate p_j from labels.
        pj = pj.clamp(eps, 1.0 - eps)  # Clamp to avoid log(0) in denominator.

        # Denominator CE_j(y, p_j) for constant predictor at base rate.
        den = ones * (-torch.log(pj)) + zeros * (-torch.log1p(-pj))
        class_term = num / den.clamp_min(eps)  # Normalize by base-rate CE_j.


        # experimental adjustments for LE  uncomment line bellow
        # class_term = class_term * 1.0 / (pj.item() * pj.item())
        # class_term = class_term * np.fabs(np.log10(pj.item())) * np.fabs(np.log10(pj.item()))

        if condition_weights is not None:
            idx = cid.item()
            if 0 <= idx < condition_weights.numel():
                w = condition_weights[idx]
                if torch.isfinite(w) and (w > 0):
                    class_term = class_term * w  # Optional per-class scaling.

        total += class_term  # Sum normalized class terms across all classes.
        if debug:
            print(
                "[localized_entropy][debug] "
                f"condition={cid.item()} "
                f"pj={pj.item():.6g} "
                f"ones={ones.item():.6g} "
                f"zeros={zeros.item():.6g} "
                f"num={num.item():.6g} "
                f"den={den.item():.6g} "
                f"class_term={class_term.item():.6g} "
                f"amplifier={np.fabs(np.log10(pj.item())):6g}"
                f"total={total.item():.6g}"
            )

    if cross_batch_history is not None:
        with torch.no_grad():
            cross_batch_history.update(
                c,
                y,
                sample_weights=sample_weights,
                base_rates=base_rates,
                num_values=bce_per.detach(),
            )

    

    denom = total_weight if isinstance(total_weight, torch.Tensor) else torch.tensor(total_weight, dtype=z.dtype, device=z.device)
    loss = total / denom.clamp_min(1.0)  # Final LE: average by total weight.
    
    if debug:
        print(
            "[localized_entropy][debug] "
            f"targets={targets.numel()} "
            f"unique_conds={unique_conds.numel()} "
            f"total_weight={float(total_weight):.6g} "
            f"denom={denom.item():.6g} "
            f"total={total.item():.6g} "
            f"loss_before_reduction={loss.item():.6g}"
        )
    
    if reduction == "sum":
        return loss * denom  # Return un-averaged total when requested.
    return loss
