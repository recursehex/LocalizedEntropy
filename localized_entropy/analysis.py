import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def print_pred_summary(
    name: str,
    preds: np.ndarray,
    labels: Optional[np.ndarray] = None,
    conds: Optional[np.ndarray] = None,
    top_k: int = 8,
) -> None:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    q = np.quantile(p, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(f"{name} prediction summary:")
    print(f"  n={p.size:,} mean={p.mean():.6f} std={p.std():.6f}")
    print(
        f"  min={q[0]:.6f} p01={q[1]:.6f} p05={q[2]:.6f} p50={q[3]:.6f} "
        f"p95={q[4]:.6f} p99={q[5]:.6f} max={q[6]:.6f}"
    )
    if labels is not None:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        p_clip = np.clip(p, 1e-12, 1 - 1e-12)
        logloss = -np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
        brier = np.mean((p - y) ** 2)
        acc = np.mean((p >= 0.5) == (y >= 0.5))
        base_rate = float(np.mean(y))
        print(
            f"  label base_rate={base_rate:.6f} | logloss={logloss:.6f} | "
            f"brier={brier:.6f} | acc@0.5={acc:.6f}"
        )
    if conds is not None:
        df = pd.DataFrame({"cond": np.asarray(conds).reshape(-1), "pred": p})
        if labels is not None:
            df["label"] = np.asarray(labels).reshape(-1)
            group = df.groupby("cond").agg(
                count=("pred", "size"),
                pred_mean=("pred", "mean"),
                label_mean=("label", "mean"),
            )
        else:
            group = df.groupby("cond").agg(
                count=("pred", "size"),
                pred_mean=("pred", "mean"),
            )
        top = group.sort_values("count", ascending=False).head(top_k)
        print(f"  Top {top_k} conditions by count:")
        print(top.to_string(float_format=lambda x: f"{x:.6f}"))


def bce_log_loss(preds: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p_clip = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)))


def roc_auc_score(preds: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]
    y = (y > 0.5).astype(np.int64)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]

    ranks = np.empty_like(p_sorted, dtype=np.float64)
    i = 0
    rank = 1
    n = p_sorted.size
    while i < n:
        j = i
        while j + 1 < n and p_sorted[j + 1] == p_sorted[i]:
            j += 1
        avg_rank = 0.5 * (rank + rank + (j - i))
        ranks[i:j + 1] = avg_rank
        rank += (j - i + 1)
        i = j + 1

    sum_ranks_pos = float(ranks[y_sorted == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc_score(preds: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]
    y = (y > 0.5).astype(np.int64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(p)[::-1]
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    ap = float(precision[y_sorted == 1].sum() / n_pos)
    return ap


def expected_calibration_error(
    preds: np.ndarray,
    labels: np.ndarray,
    bins: int = 20,
    min_count: int = 1,
) -> Tuple[float, pd.DataFrame]:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(p, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    rows = []
    total = p.size
    ece = 0.0
    for b in range(bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count < min_count:
            rows.append(
                {
                    "bin": b,
                    "count": count,
                    "avg_pred": np.nan,
                    "avg_label": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        avg_pred = float(p[mask].mean())
        avg_label = float(y[mask].mean())
        gap = abs(avg_pred - avg_label)
        weight = count / max(1, total)
        ece += weight * gap
        rows.append(
            {
                "bin": b,
                "count": count,
                "avg_pred": avg_pred,
                "avg_label": avg_label,
                "abs_gap": gap,
            }
        )
    table = pd.DataFrame(rows)
    return float(ece), table


def per_condition_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    conds: np.ndarray,
    bins: int = 20,
    min_count: int = 1,
    small_prob_max: float = 0.01,
) -> pd.DataFrame:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    c = np.asarray(conds).reshape(-1)
    unique = np.unique(c)

    rows = []
    for cid in unique:
        mask = c == cid
        count = int(mask.sum())
        if count == 0:
            continue
        p_c = p[mask]
        y_c = y[mask]
        bce = bce_log_loss(p_c, y_c)
        ece, _ = expected_calibration_error(p_c, y_c, bins=bins, min_count=min_count)
        small_mask = p_c <= small_prob_max
        if small_mask.any():
            ece_small, _ = expected_calibration_error(
                p_c[small_mask],
                y_c[small_mask],
                bins=bins,
                min_count=min_count,
            )
        else:
            ece_small = float("nan")
        rows.append(
            {
                "condition": int(cid),
                "count": count,
                "base_rate": float(y_c.mean()),
                "bce": bce,
                "ece": ece,
                "ece_small": ece_small,
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def collect_le_stats_per_condition(
    logits: torch.Tensor,
    targets: torch.Tensor,
    conditions: torch.Tensor,
    eps: float = 1e-12,
):
    z = logits.view(-1)
    y = targets.view(-1).to(z.dtype)
    c = conditions.view(-1).to(torch.long)

    bce_per = torch.clamp_min(z, 0) - z * y + torch.log1p(torch.exp(-torch.abs(z)))

    stats = {}
    for cid in torch.unique(c):
        mask = (c == cid)
        n = int(mask.sum().item())
        if n == 0:
            continue
        yj = y[mask]
        num = float(bce_per[mask].sum().item())
        ones = float(yj.sum().item())
        zeros = float(n) - ones
        pj = ones / max(1.0, float(n))
        pj = max(eps, min(1.0 - eps, pj))
        den = ones * (-math.log(pj)) + zeros * (-math.log1p(-pj))
        ratio = num / (den if den > eps else eps)
        stats[int(cid.item())] = {
            "Numerator": num,
            "Denominator": den,
            "Average prediction for denominator": pj,
            "Number of samples with label 1": int(round(ones)),
            "Number of samples with label 0": int(round(zeros)),
            "Numerator/denominator": ratio,
        }
    return stats


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    non_blocking: bool = False,
):
    model.eval()
    all_logits, all_targets, all_conditions = [], [], []
    for xb, cb, yb, nw in loader:
        xb = xb.to(device, non_blocking=non_blocking)
        cb = cb.to(device, non_blocking=non_blocking)
        yb = yb.to(device, non_blocking=non_blocking)
        zb = model(xb, cb)
        all_logits.append(zb.detach().cpu())
        all_targets.append(yb.detach().cpu())
        all_conditions.append(cb.detach().cpu())
    if not all_logits:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    z_all = torch.cat(all_logits).view(-1)
    y_all = torch.cat(all_targets).view(-1)
    c_all = torch.cat(all_conditions).view(-1)
    return z_all, y_all, c_all
