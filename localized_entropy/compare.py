from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from localized_entropy.analysis import bce_log_loss, expected_calibration_error

from localized_entropy.analysis import (
    collect_le_stats_per_condition,
    le_stats_to_frame,
    per_condition_calibration,
)
from localized_entropy.experiments import TrainRunResult


def build_comparison_frame(
    bce_preds: np.ndarray,
    le_preds: np.ndarray,
    labels: np.ndarray,
    conds: np.ndarray,
    bce_le_stats: dict,
    le_le_stats: dict,
) -> pd.DataFrame:
    bce_cal = per_condition_calibration(bce_preds, labels, conds)
    le_cal = per_condition_calibration(le_preds, labels, conds)

    bce_cal = bce_cal.rename(
        columns={
            "pred_mean": "bce_pred_mean",
            "calibration": "bce_calibration",
        }
    )
    le_cal = le_cal.rename(
        columns={
            "pred_mean": "le_pred_mean",
            "calibration": "le_calibration",
        }
    )

    bce_le = le_stats_to_frame(bce_le_stats).rename(columns={"le_ratio": "bce_le_ratio"})
    le_le = le_stats_to_frame(le_le_stats).rename(columns={"le_ratio": "le_le_ratio"})

    # Merge per-condition calibration stats and LE ratios for side-by-side comparison.
    merged = bce_cal.merge(
        le_cal[["condition", "le_pred_mean", "le_calibration"]],
        on="condition",
        how="outer",
    )
    merged = merged.merge(
        bce_le[["condition", "bce_le_ratio"]],
        on="condition",
        how="left",
    )
    merged = merged.merge(
        le_le[["condition", "le_le_ratio"]],
        on="condition",
        how="left",
    )
    merged["delta_calibration"] = merged["le_calibration"] - merged["bce_calibration"]
    merged["delta_le_ratio"] = merged["le_le_ratio"] - merged["bce_le_ratio"]
    return merged


def sort_comparison_frame(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "abs_delta_calibration":
        return df.sort_values("delta_calibration", key=lambda s: s.abs(), ascending=False)
    if sort_by == "abs_delta_le_ratio":
        return df.sort_values("delta_le_ratio", key=lambda s: s.abs(), ascending=False)
    return df.sort_values(sort_by, ascending=False)


def format_comparison_table(
    df: pd.DataFrame,
    columns: list,
    *,
    top_k: int = 20,
    float_format: str = "{:.6g}",
) -> str:
    if df is None or len(df) == 0:
        return ""
    top_k = max(int(top_k), 1)
    columns = list(columns)
    table_df = df.loc[:, columns].head(top_k)
    formatted = []
    numeric_cols = {}
    for col in columns:
        numeric_cols[col] = pd.api.types.is_numeric_dtype(table_df[col])

    for _, row in table_df.iterrows():
        row_values = {}
        for col in columns:
            val = row[col]
            if isinstance(val, (np.integer, int)):
                row_values[col] = str(int(val))
            elif isinstance(val, (np.floating, float)):
                if np.isnan(val):
                    row_values[col] = "nan"
                else:
                    row_values[col] = float_format.format(float(val))
            else:
                row_values[col] = str(val)
        formatted.append(row_values)

    widths = {}
    for col in columns:
        width = len(str(col))
        for row in formatted:
            width = max(width, len(row[col]))
        widths[col] = width

    header_parts = []
    for col in columns:
        header_parts.append(str(col).ljust(widths[col]))
    lines = ["  ".join(header_parts)]

    for row in formatted:
        parts = []
        for col in columns:
            text = row[col]
            if numeric_cols[col]:
                parts.append(text.rjust(widths[col]))
            else:
                parts.append(text.ljust(widths[col]))
        lines.append("  ".join(parts))
    return "\n".join(lines)


def summarize_model_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    *,
    ece_bins: int = 20,
    ece_min_count: int = 1,
    threshold: float = 0.5,
    small_prob_max: float = 0.01,
    small_prob_quantile: float = 0.1,
) -> dict:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    logloss = bce_log_loss(p, y)
    brier = float(np.mean((p - y) ** 2))
    acc = float(np.mean((p >= threshold) == (y >= 0.5)))
    ece, _ = expected_calibration_error(p, y, bins=ece_bins, min_count=ece_min_count)
    small_threshold = float(small_prob_max)
    small_mask = p <= small_threshold
    if not small_mask.any() and small_prob_quantile is not None:
        quantile_threshold = float(np.quantile(p, small_prob_quantile))
        small_threshold = quantile_threshold
        small_mask = p <= small_threshold
    if small_mask.any():
        ece_small, _ = expected_calibration_error(
            p[small_mask],
            y[small_mask],
            bins=ece_bins,
            min_count=ece_min_count,
        )
    else:
        ece_small = float("nan")
    return {
        "logloss": logloss,
        "brier": brier,
        "accuracy": acc,
        "ece": float(ece),
        "ece_small": float(ece_small),
    }


def _winner(
    bce_val: float,
    le_val: float,
    *,
    lower_is_better: bool,
    tol: float = 1e-8,
) -> str:
    bce_ok = np.isfinite(bce_val)
    le_ok = np.isfinite(le_val)
    if not bce_ok and not le_ok:
        return "n/a"
    if bce_ok and not le_ok:
        return "BCE"
    if le_ok and not bce_ok:
        return "LE"
    diff = (bce_val - le_val) if lower_is_better else (le_val - bce_val)
    if abs(diff) <= tol:
        return "tie"
    return "LE" if diff > 0 else "BCE"


def format_bce_le_summary(
    bce_result: TrainRunResult,
    le_result: TrainRunResult,
    eval_labels: np.ndarray,
    *,
    ece_bins: int = 20,
    ece_min_count: int = 1,
    threshold: float = 0.5,
) -> str:
    labels = np.asarray(eval_labels, dtype=np.float64).reshape(-1)
    bce_metrics = summarize_model_metrics(
        bce_result.eval_preds,
        labels,
        ece_bins=ece_bins,
        ece_min_count=ece_min_count,
        threshold=threshold,
    )
    le_metrics = summarize_model_metrics(
        le_result.eval_preds,
        labels,
        ece_bins=ece_bins,
        ece_min_count=ece_min_count,
        threshold=threshold,
    )

    acc_winner = _winner(
        bce_metrics["accuracy"],
        le_metrics["accuracy"],
        lower_is_better=False,
    )
    logloss_winner = _winner(
        bce_metrics["logloss"],
        le_metrics["logloss"],
        lower_is_better=True,
    )
    brier_winner = _winner(
        bce_metrics["brier"],
        le_metrics["brier"],
        lower_is_better=True,
    )
    ece_winner = _winner(
        bce_metrics["ece"],
        le_metrics["ece"],
        lower_is_better=True,
    )

    def fmt(val: float) -> str:
        return "nan" if not np.isfinite(val) else f"{val:.6g}"

    lines = [
        "BCE vs LE summary (lower is better for logloss/brier/ece):",
        f"Accuracy@0.5: BCE={fmt(bce_metrics['accuracy'])} | LE={fmt(le_metrics['accuracy'])} -> {acc_winner}",
        f"Logloss:       BCE={fmt(bce_metrics['logloss'])} | LE={fmt(le_metrics['logloss'])} -> {logloss_winner}",
        f"Brier:         BCE={fmt(bce_metrics['brier'])} | LE={fmt(le_metrics['brier'])} -> {brier_winner}",
        f"ECE:           BCE={fmt(bce_metrics['ece'])} | LE={fmt(le_metrics['ece'])} -> {ece_winner}",
        f"Closer to actual data (logloss): {logloss_winner}",
    ]
    return "\n".join(lines)


def compare_bce_le_runs(
    bce_result: TrainRunResult,
    le_result: TrainRunResult,
    eval_labels: np.ndarray,
    eval_conds: np.ndarray,
    *,
    condition_label: str,
    sort_by: str = "count",
) -> pd.DataFrame:
    if bce_result.eval_logits is None or bce_result.eval_targets is None or bce_result.eval_conds is None:
        raise ValueError("BCE results missing eval logits/targets/conditions.")
    if le_result.eval_logits is None or le_result.eval_targets is None or le_result.eval_conds is None:
        raise ValueError("LE results missing eval logits/targets/conditions.")

    labels = np.asarray(eval_labels, dtype=np.float64).reshape(-1)
    conds = np.asarray(eval_conds, dtype=np.int64).reshape(-1)
    if labels.size == 0 or conds.size == 0:
        raise ValueError("Eval labels/conditions are empty; cannot compare calibration.")
    if labels.shape[0] != bce_result.eval_preds.reshape(-1).shape[0]:
        raise ValueError("Eval labels do not align with BCE prediction length.")
    if labels.shape[0] != le_result.eval_preds.reshape(-1).shape[0]:
        raise ValueError("Eval labels do not align with LE prediction length.")

    bce_stats = collect_le_stats_per_condition(
        bce_result.eval_logits,
        bce_result.eval_targets,
        bce_result.eval_conds,
        eps=1e-12,
    )
    le_stats = collect_le_stats_per_condition(
        le_result.eval_logits,
        le_result.eval_targets,
        le_result.eval_conds,
        eps=1e-12,
    )
    comparison = build_comparison_frame(
        bce_result.eval_preds,
        le_result.eval_preds,
        labels,
        conds,
        bce_stats,
        le_stats,
    )
    comparison = sort_comparison_frame(comparison, sort_by)
    comparison = comparison.rename(columns={"condition": condition_label})

    return comparison


_DEFAULT_REPEAT_METRICS: Dict[str, bool] = {
    "logloss": True,
    "brier": True,
    "ece": True,
    "ece_small": True,
    "accuracy": False,
}


def _resolve_repeat_metrics(metrics: Optional[Dict[str, bool]] = None) -> Dict[str, bool]:
    if metrics is None:
        return dict(_DEFAULT_REPEAT_METRICS)
    return dict(metrics)


def _safe_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
    *,
    zero_method: str = "wilcox",
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    try:
        # SciPy's wilcoxon signature varies across versions; handle both.
        stat, p_value = wilcoxon(x, y, zero_method=zero_method, alternative=alternative)
    except TypeError:
        stat, p_value = wilcoxon(x, y, zero_method=zero_method)
    except ValueError:
        return float("nan"), float("nan")
    return float(stat), float(p_value)


def build_repeat_metrics_frame(
    runs: List[TrainRunResult],
    eval_labels: np.ndarray,
    *,
    ece_bins: int = 20,
    ece_min_count: int = 1,
    threshold: float = 0.5,
    small_prob_max: float = 0.01,
    small_prob_quantile: float = 0.1,
    run_label: str = "run",
    run_values: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    labels = np.asarray(eval_labels, dtype=np.float64).reshape(-1)
    if labels.size == 0:
        raise ValueError("Eval labels are empty; cannot summarize repeated runs.")
    if run_values is not None:
        run_values = list(run_values)
        if len(run_values) != len(runs):
            raise ValueError("run_values length does not match run count.")

    rows = []
    for idx, result in enumerate(runs):
        metrics = summarize_model_metrics(
            result.eval_preds,
            labels,
            ece_bins=ece_bins,
            ece_min_count=ece_min_count,
            threshold=threshold,
            small_prob_max=small_prob_max,
            small_prob_quantile=small_prob_quantile,
        )
        rows.append(
            {
                run_label: run_values[idx] if run_values is not None else idx,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_repeat_metrics(
    metrics_df: pd.DataFrame,
    *,
    metrics: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    metric_map = _resolve_repeat_metrics(metrics)
    rows = []
    for metric in metric_map:
        if metric not in metrics_df.columns:
            continue
        values = metrics_df[metric].to_numpy()
        if values.size == 0:
            continue
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        rows.append(
            {
                "metric": metric,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "std": std,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def build_wilcoxon_summary(
    bce_metrics: pd.DataFrame,
    le_metrics: pd.DataFrame,
    *,
    metrics: Optional[Dict[str, bool]] = None,
    zero_method: str = "wilcox",
    alternative: str = "two-sided",
) -> pd.DataFrame:
    metric_map = _resolve_repeat_metrics(metrics)
    rows = []
    for metric, lower_is_better in metric_map.items():
        if metric not in bce_metrics.columns or metric not in le_metrics.columns:
            continue
        bce_vals = bce_metrics[metric].to_numpy()
        le_vals = le_metrics[metric].to_numpy()
        if bce_vals.size != le_vals.size:
            raise ValueError(f"Repeat runs for '{metric}' do not align between BCE and LE.")
        delta = (bce_vals - le_vals) if lower_is_better else (le_vals - bce_vals)
        stat, p_value = _safe_wilcoxon(
            bce_vals,
            le_vals,
            zero_method=zero_method,
            alternative=alternative,
        )
        rows.append(
            {
                "metric": metric,
                "n": int(bce_vals.size),
                "n_nonzero": int(np.count_nonzero(delta)),
                "median_delta": float(np.median(delta)) if delta.size else float("nan"),
                "mean_delta": float(np.mean(delta)) if delta.size else float("nan"),
                "statistic": stat,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def format_wilcoxon_summary(
    summary_df: pd.DataFrame,
    *,
    float_format: str = "{:.6g}",
) -> str:
    if summary_df is None or len(summary_df) == 0:
        return ""
    columns = [
        "metric",
        "n",
        "n_nonzero",
        "median_delta",
        "mean_delta",
        "statistic",
        "p_value",
    ]
    return format_comparison_table(
        summary_df,
        columns,
        top_k=len(summary_df),
        float_format=float_format,
    )


def _per_condition_pred_mean(
    preds: np.ndarray,
    conds: np.ndarray,
    num_conditions: int,
) -> np.ndarray:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    c = np.asarray(conds, dtype=np.int64).reshape(-1)
    sums = np.bincount(c, weights=p, minlength=num_conditions)
    counts = np.bincount(c, minlength=num_conditions)
    denom = np.maximum(counts, 1)
    return sums / denom


def build_per_condition_calibration_wilcoxon(
    bce_runs: List[TrainRunResult],
    le_runs: List[TrainRunResult],
    eval_labels: np.ndarray,
    eval_conds: np.ndarray,
    *,
    zero_method: str = "wilcox",
    alternative: str = "two-sided",
    min_count: int = 1,
) -> pd.DataFrame:
    if len(bce_runs) != len(le_runs):
        raise ValueError("Repeat run counts do not match between BCE and LE.")
    labels = np.asarray(eval_labels, dtype=np.float64).reshape(-1)
    conds = np.asarray(eval_conds, dtype=np.int64).reshape(-1)
    if labels.size == 0 or conds.size == 0:
        raise ValueError("Eval labels/conditions are empty; cannot compare calibration.")
    num_conditions = int(conds.max()) + 1
    counts = np.bincount(conds, minlength=num_conditions)
    label_sum = np.bincount(conds, weights=labels, minlength=num_conditions)
    # Base rates serve as the per-condition target for calibration gaps.
    base_rate = label_sum / np.maximum(counts, 1)

    bce_gaps = []
    for result in bce_runs:
        pred_mean = _per_condition_pred_mean(result.eval_preds, conds, num_conditions)
        bce_gaps.append(np.abs(pred_mean - base_rate))
    le_gaps = []
    for result in le_runs:
        pred_mean = _per_condition_pred_mean(result.eval_preds, conds, num_conditions)
        le_gaps.append(np.abs(pred_mean - base_rate))

    bce_gaps = np.stack(bce_gaps, axis=0)
    le_gaps = np.stack(le_gaps, axis=0)

    rows = []
    for cond_id in range(num_conditions):
        if counts[cond_id] < min_count:
            continue
        bce_vals = bce_gaps[:, cond_id]
        le_vals = le_gaps[:, cond_id]
        delta = bce_vals - le_vals
        stat, p_value = _safe_wilcoxon(
            bce_vals,
            le_vals,
            zero_method=zero_method,
            alternative=alternative,
        )
        rows.append(
            {
                "condition": int(cond_id),
                "count": int(counts[cond_id]),
                "base_rate": float(base_rate[cond_id]),
                "bce_gap": float(np.mean(bce_vals)),
                "le_gap": float(np.mean(le_vals)),
                "delta_mean": float(np.mean(delta)),
                "n": int(delta.size),
                "n_nonzero": int(np.count_nonzero(delta)),
                "statistic": stat,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def sort_per_condition_wilcoxon_frame(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "p_value":
        return df.sort_values("p_value", ascending=True)
    if sort_by == "abs_delta_mean":
        return df.sort_values("delta_mean", key=lambda s: s.abs(), ascending=False)
    if sort_by in {"delta_mean", "count", "base_rate", "bce_gap", "le_gap"}:
        return df.sort_values(sort_by, ascending=False)
    return df.sort_values(sort_by, ascending=False)
