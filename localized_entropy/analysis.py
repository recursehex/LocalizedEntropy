import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def print_feature_stats(name: str, features: Optional[np.ndarray]) -> None:
    """Print basic statistics for a feature matrix."""
    if features is None:
        print(f"{name}: None")
        return
    x = np.asarray(features)
    if x.size == 0:
        print(f"{name}: empty")
        return
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    near_zero = float(np.mean(stds < 1e-8))
    print(f"{name}: shape={x.shape}")
    print(f"  mean: min={means.min():.6g} max={means.max():.6g} avg={means.mean():.6g}")
    print(f"  std:  min={stds.min():.6g} max={stds.max():.6g} avg={stds.mean():.6g}")
    print(f"  std < 1e-8: {near_zero:.2%}")


def print_condition_stats(
    name: str,
    conds: Optional[np.ndarray],
    num_conditions: int,
) -> None:
    """Print per-condition counts and ranges."""
    if conds is None:
        print(f"{name}: None")
        return
    c = np.asarray(conds, dtype=np.int64).reshape(-1)
    if c.size == 0:
        print(f"{name}: empty")
        return
    counts = np.bincount(c, minlength=int(num_conditions))
    uniq = np.unique(c)
    print(f"{name}: n={c.size:,} unique={uniq.size} min={c.min()} max={c.max()}")
    print(
        f"  counts: min={counts.min()} max={counts.max()} mean={counts.mean():.2f} zeros={(counts == 0).sum()}"
    )


def print_label_stats(
    name: str,
    labels: Optional[np.ndarray],
    conds: Optional[np.ndarray],
    num_conditions: int,
) -> None:
    """Print label distribution stats with optional per-condition rates."""
    if labels is None:
        print(f"{name}: None")
        return
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if y.size == 0:
        print(f"{name}: empty")
        return
    print(f"{name}: n={y.size:,} base_rate={y.mean():.6g} min={y.min():.6g} max={y.max():.6g}")
    if conds is None:
        return
    c = np.asarray(conds, dtype=np.int64).reshape(-1)
    counts = np.bincount(c, minlength=int(num_conditions))
    sums = np.bincount(c, weights=y, minlength=int(num_conditions))
    rates = sums / np.maximum(counts, 1)
    valid = counts > 0
    if valid.any():
        print(
            "  per-cond base_rate: "
            f"min={rates[valid].min():.6g} max={rates[valid].max():.6g} avg={rates[valid].mean():.6g}"
        )


def print_pred_summary(
    name: str,
    preds: np.ndarray,
    labels: Optional[np.ndarray] = None,
    conds: Optional[np.ndarray] = None,
    top_k: int = 8,
) -> None:
    """Print summary statistics for predictions and optional labels/conditions."""
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


def print_pred_stats_by_condition(
    preds: np.ndarray,
    conds: np.ndarray,
    num_conditions: int,
    *,
    name: str = "Eval",
) -> None:
    """Print min/max/mean predictions per condition."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    c = np.asarray(conds, dtype=np.int64).reshape(-1)
    counts = np.bincount(c, minlength=int(num_conditions))
    print(f"{name} prediction stats per condition:")
    for cond in range(int(num_conditions)):
        n = int(counts[cond])
        if n == 0:
            print(f"  cond {cond}: n=0")
            continue
        pc = p[c == cond]
        print(
            f"  cond {cond}: n={n} min={pc.min():.6g} max={pc.max():.6g} "
            f"mean={pc.mean():.6g}"
        )


def bce_log_loss(preds: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Compute BCE log loss from probabilities and labels."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p_clip = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)))


def roc_auc_score(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC from probabilities and binary labels."""
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
        # Assign average ranks to tied prediction scores.
        avg_rank = 0.5 * (rank + rank + (j - i))
        ranks[i:j + 1] = avg_rank
        rank += (j - i + 1)
        i = j + 1

    sum_ranks_pos = float(ranks[y_sorted == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc_score(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute PR-AUC (average precision) from probabilities and labels."""
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


def _binary_classification_counts(
    preds: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Tuple[int, int, int, int]:
    """Return TP/FP/TN/FN counts at a probability threshold."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size == 0 or y.size == 0:
        return 0, 0, 0, 0
    y_true = y >= 0.5
    y_pred = p >= threshold
    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    tn = int(np.sum(~y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    return tp, fp, tn, fn


def binary_classification_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """Compute accuracy/precision/recall/F1 at a threshold."""
    tp, fp, tn, fn = _binary_classification_counts(
        preds, labels, threshold=threshold
    )
    total = tp + fp + tn + fn
    if total == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def expected_calibration_error(
    preds: np.ndarray,
    labels: np.ndarray,
    bins: int = 20,
    min_count: int = 1,
) -> Tuple[float, pd.DataFrame]:
    """Compute ECE and return a per-bin calibration table."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(p, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    rows = []
    total = p.size
    # Compute weighted average of per-bin calibration gaps.
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
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute metrics per condition, including calibration and accuracy."""
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
        cls_metrics = binary_classification_metrics(p_c, y_c, threshold=threshold)
        ece, _ = expected_calibration_error(p_c, y_c, bins=bins, min_count=min_count)
        # Track calibration on low-probability predictions separately.
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
                "accuracy": cls_metrics["accuracy"],
                "f1": cls_metrics["f1"],
                "ece": ece,
                "ece_small": ece_small,
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def per_condition_calibration(
    preds: np.ndarray,
    labels: np.ndarray,
    conds: np.ndarray,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute per-condition base rate and calibration ratio."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    c = np.asarray(conds, dtype=np.int64).reshape(-1)
    if c.size == 0:
        return pd.DataFrame(
            columns=["condition", "count", "base_rate", "pred_mean", "calibration"]
        )
    max_id = int(c.max())
    counts = np.bincount(c, minlength=max_id + 1)
    pred_sum = np.bincount(c, weights=p, minlength=max_id + 1)
    label_sum = np.bincount(c, weights=y, minlength=max_id + 1)
    denom = np.maximum(counts, 1)
    base_rate = label_sum / denom
    pred_mean = pred_sum / denom
    calibration = np.divide(
        pred_mean,
        base_rate,
        out=np.full_like(pred_mean, np.nan, dtype=np.float64),
        where=base_rate > eps,
    )
    df = pd.DataFrame(
        {
            "condition": np.arange(max_id + 1, dtype=np.int64),
            "count": counts.astype(np.int64),
            "base_rate": base_rate,
            "pred_mean": pred_mean,
            "calibration": calibration,
        }
    )
    return df.sort_values("count", ascending=False)


def per_condition_mean(
    values: Optional[np.ndarray],
    conds: Optional[np.ndarray],
    num_conditions: int,
) -> Optional[np.ndarray]:
    """Compute per-condition mean for a value array."""
    if values is None or conds is None:
        return None
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    c = np.asarray(conds, dtype=np.int64).reshape(-1)
    if c.size == 0:
        return np.zeros(int(num_conditions), dtype=np.float64)
    counts = np.bincount(c, minlength=int(num_conditions))
    sums = np.bincount(c, weights=v, minlength=int(num_conditions))
    return sums / np.maximum(counts, 1)


def per_condition_calibration_from_base_rates(
    preds: np.ndarray,
    conds: np.ndarray,
    base_rates: np.ndarray,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute calibration ratios using externally supplied base rates."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    c = np.asarray(conds, dtype=np.int64).reshape(-1)
    if c.size == 0:
        return pd.DataFrame(
            columns=["condition", "count", "base_rate", "pred_mean", "calibration"]
        )
    max_id = int(c.max())
    counts = np.bincount(c, minlength=max_id + 1)
    pred_sum = np.bincount(c, weights=p, minlength=max_id + 1)
    denom = np.maximum(counts, 1)
    pred_mean = pred_sum / denom
    base_rates = np.asarray(base_rates, dtype=np.float64).reshape(-1)
    if base_rates.size <= max_id:
        padded = np.full(max_id + 1, np.nan, dtype=np.float64)
        padded[:base_rates.size] = base_rates
        base_rates = padded
    calibration = np.divide(
        pred_mean,
        base_rates,
        out=np.full_like(pred_mean, np.nan, dtype=np.float64),
        where=base_rates > eps,
    )
    df = pd.DataFrame(
        {
            "condition": np.arange(max_id + 1, dtype=np.int64),
            "count": counts.astype(np.int64),
            "base_rate": base_rates[: max_id + 1],
            "pred_mean": pred_mean,
            "calibration": calibration,
        }
    )
    return df.sort_values("count", ascending=False)


def le_stats_to_frame(stats: dict) -> pd.DataFrame:
    """Convert LE stats dict into a DataFrame."""
    rows = []
    for cond, payload in stats.items():
        rows.append(
            {
                "condition": int(cond),
                "le_numerator": float(payload.get("Numerator", float("nan"))),
                "le_denominator": float(payload.get("Denominator", float("nan"))),
                "le_ratio": float(payload.get("Numerator/denominator", float("nan"))),
                "label_ones": int(payload.get("Number of samples with label 1", 0)),
                "label_zeros": int(payload.get("Number of samples with label 0", 0)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "condition",
                "le_numerator",
                "le_denominator",
                "le_ratio",
                "label_ones",
                "label_zeros",
            ]
        )
    return pd.DataFrame(rows).sort_values("condition")


def summarize_per_ad_train_eval_rates(
    train_labels: Optional[np.ndarray],
    train_conds: Optional[np.ndarray],
    eval_preds: Optional[np.ndarray],
    eval_conds: Optional[np.ndarray],
    num_conditions: int,
    *,
    condition_label: str,
    eval_name: str,
    top_k: int = 10,
) -> Optional[pd.DataFrame]:
    """Summarize per-condition train rates vs eval prediction averages."""
    top_k = max(int(top_k), 1)
    train_has = train_labels is not None and train_conds is not None
    eval_has = eval_preds is not None and eval_conds is not None

    if not train_has:
        print("[WARN] Train labels/conditions unavailable; skipping per-ad train click rates.")
    else:
        c_train = np.asarray(train_conds, dtype=np.int64).reshape(-1)
        y_train = np.asarray(train_labels, dtype=np.float64).reshape(-1)
        train_counts = np.bincount(c_train, minlength=int(num_conditions))
        train_clicks = np.bincount(c_train, weights=y_train, minlength=int(num_conditions))
        train_rates = train_clicks / np.maximum(train_counts, 1)

    if not eval_has:
        print(f"[WARN] {eval_name} conditions unavailable; skipping per-ad prediction averages.")
    else:
        c_eval = np.asarray(eval_conds, dtype=np.int64).reshape(-1)
        p_eval = np.asarray(eval_preds, dtype=np.float64).reshape(-1)
        eval_counts = np.bincount(c_eval, minlength=int(num_conditions))
        eval_pred_sums = np.bincount(c_eval, weights=p_eval, minlength=int(num_conditions))
        eval_pred_avgs = eval_pred_sums / np.maximum(eval_counts, 1)

    if train_has and eval_has:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(
                eval_pred_avgs,
                train_rates,
                out=np.full_like(eval_pred_avgs, np.nan, dtype=np.float64),
                where=train_rates > 0,
            )

        per_ad_rates = pd.DataFrame(
            {
                condition_label: np.arange(num_conditions, dtype=np.int64),
                "train_clicks": train_clicks.astype(np.int64),
                "train_total": train_counts.astype(np.int64),
                "train_click_rate": train_rates,
                f"{eval_name.lower()}_pred_avg": eval_pred_avgs,
                f"{eval_name.lower()}_total": eval_counts.astype(np.int64),
                "pred_to_train_rate": ratio,
            }
        ).sort_values("train_total", ascending=False)

        if num_conditions <= top_k:
            print(per_ad_rates.to_string(index=False))
            return per_ad_rates
        print(per_ad_rates.head(top_k).to_string(index=False))
        print(
            f"[INFO] Showing top {top_k} by train count; {num_conditions} total {condition_label} values."
        )
        return per_ad_rates.head(top_k)

    if train_has:
        train_only = pd.DataFrame(
            {
                condition_label: np.arange(num_conditions, dtype=np.int64),
                "train_clicks": train_clicks.astype(np.int64),
                "train_total": train_counts.astype(np.int64),
                "train_click_rate": train_rates,
            }
        ).sort_values("train_total", ascending=False)
        if num_conditions <= top_k:
            print(train_only.to_string(index=False))
        else:
            print(train_only.head(top_k).to_string(index=False))
            print(
                f"[INFO] Showing top {top_k} by train count; {num_conditions} total {condition_label} values."
            )
        return None

    if eval_has:
        eval_only = pd.DataFrame(
            {
                condition_label: np.arange(num_conditions, dtype=np.int64),
                f"{eval_name.lower()}_pred_avg": eval_pred_avgs,
                f"{eval_name.lower()}_total": eval_counts.astype(np.int64),
            }
        ).sort_values(f"{eval_name.lower()}_total", ascending=False)
        if num_conditions <= top_k:
            print(eval_only.to_string(index=False))
        else:
            print(eval_only.head(top_k).to_string(index=False))
            print(
                f"[INFO] Showing top {top_k} by eval count; {num_conditions} total {condition_label} values."
            )
    return None


def collect_le_stats_per_condition(
    logits: torch.Tensor,
    targets: torch.Tensor,
    conditions: torch.Tensor,
    eps: float = 1e-12,
):
    """Compute LE numerator/denominator stats per condition."""
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
    """Collect logits, targets, and conditions over a loader."""
    model.eval()
    first_param = next(model.parameters(), None)
    model_dtype = first_param.dtype if first_param is not None else torch.float32
    all_logits, all_targets, all_conditions = [], [], []
    for batch in loader:
        xb, x_cat, cb, yb, _ = batch
        xb = xb.to(device, non_blocking=non_blocking, dtype=model_dtype)
        x_cat = x_cat.to(device, non_blocking=non_blocking)
        cb = cb.to(device, non_blocking=non_blocking)
        yb = yb.to(device, non_blocking=non_blocking, dtype=model_dtype)
        zb = model(xb, x_cat, cb)
        all_logits.append(zb.detach().cpu())
        all_targets.append(yb.detach().cpu())
        all_conditions.append(cb.detach().cpu())
    if not all_logits:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    z_all = torch.cat(all_logits).view(-1)
    y_all = torch.cat(all_targets).view(-1)
    c_all = torch.cat(all_conditions).view(-1)
    return z_all, y_all, c_all
