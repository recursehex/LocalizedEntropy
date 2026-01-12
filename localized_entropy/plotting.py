from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8")


def _density_lines(
    values: np.ndarray,
    groups: np.ndarray,
    num_conditions: int,
    *,
    bins: int = 100,
    transform: Optional[str] = None,
    value_range: Optional[Tuple[float, float]] = None,
    title: str = "",
    x_label: str = "",
    density: bool = True,
) -> None:
    vals = values.astype(np.float64).copy()
    if transform == "log10":
        eps = 1e-12
        vals = np.log10(np.clip(vals, eps, None))
    if value_range is None:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    else:
        vmin, vmax = value_range
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    elif vmin == vmax:
        pad = max(1.0, abs(vmin) * 0.01)
        vmin, vmax = vmin - pad, vmax + pad
    edges = np.linspace(vmin, vmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(10, 6))
    for cond in range(int(num_conditions)):
        m = groups == cond
        if not np.any(m):
            continue
        vv = vals[m]
        hist, _ = np.histogram(vv, bins=edges, density=density)
        plt.plot(centers, hist, label=f"Condition {cond}")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Density" if density else "Count")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_training_distributions(
    net_worth: np.ndarray,
    ages: np.ndarray,
    probs: np.ndarray,
    conds: np.ndarray,
    num_conditions: int,
) -> None:
    _density_lines(
        values=net_worth,
        groups=conds,
        num_conditions=num_conditions,
        bins=120,
        transform="log10",
        title="Training Data: Distribution by Condition (log10(NetWorth))",
        x_label="log10(NetWorth)",
    )
    _density_lines(
        values=ages,
        groups=conds,
        num_conditions=num_conditions,
        bins=120,
        transform=None,
        title="Training Data: Distribution by Condition (Age)",
        x_label="Age",
    )
    _density_lines(
        values=probs,
        groups=conds,
        num_conditions=num_conditions,
        bins=120,
        transform="log10",
        value_range=(-12, 0),
        title="Training Data: Distribution by Condition (log10(true probability))",
        x_label="log10(true p)",
    )


def plot_eval_log10p_hist(preds: np.ndarray, epoch: int, bins: int = 100, *, name: str = "Eval") -> None:
    eps = 1e-12
    log10p = np.log10(np.clip(preds, eps, 1.0))
    plt.figure(figsize=(8, 5))
    plt.hist(log10p, bins=bins, range=(-12, 0), density=True, color="#4477aa", alpha=0.85)
    plt.title(f"{name} Predicted Probability: log10(p) (Epoch {epoch})")
    plt.xlabel("log10(pred p)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_loss_curves(train_losses, eval_losses, loss_label: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label=f"Train {loss_label}")
    plt.plot(eval_losses, label=f"Eval {loss_label}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss_label} Loss")
    plt.title(f"Training vs Evaluation {loss_label} over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_eval_predictions_by_condition(
    preds: np.ndarray,
    conds: np.ndarray,
    num_conditions: int,
    *,
    name: str = "Eval",
    title: Optional[str] = None,
    print_counts: bool = True,
    value_range: Tuple[float, float] = (-12, 0),
) -> None:
    if title is None:
        title = f"{name} Predictions: Distribution by Condition (log10(pred probability))"
    if preds.size == 0:
        print("No predictions available; skipping eval prediction plot.")
        return
    if print_counts:
        c = conds.astype(np.int64).reshape(-1)
        counts = np.bincount(c, minlength=int(num_conditions))
        print(f"{name} predictions per condition: {counts.tolist()}")
        missing = [idx for idx, cnt in enumerate(counts) if cnt == 0]
        if missing:
            print(f"Conditions with zero {name.lower()} samples: {missing}")
    _density_lines(
        values=preds,
        groups=conds,
        num_conditions=num_conditions,
        bins=120,
        transform="log10",
        value_range=value_range,
        title=title,
        x_label="log10(pred p)",
    )


def plot_feature_distributions_by_condition(
    xnum: np.ndarray,
    conds: np.ndarray,
    feature_names: list,
    num_conditions: int,
    *,
    max_features: int = 3,
    bins: int = 120,
    log10_features: Optional[set] = None,
    density: bool = True,
) -> None:
    if log10_features is None:
        log10_features = set()
    num_features = min(len(feature_names), xnum.shape[1], max_features)
    for idx in range(num_features):
        name = feature_names[idx]
        transform = "log10" if name in log10_features else None
        _density_lines(
            values=xnum[:, idx],
            groups=conds,
            num_conditions=num_conditions,
            bins=bins,
            transform=transform,
            title=f"Training Data: Distribution by Condition ({name})",
            x_label=name,
            density=density,
        )


def plot_label_rates_by_condition(
    labels: np.ndarray,
    conds: np.ndarray,
    num_conditions: int,
) -> None:
    c = conds.astype(np.int64).reshape(-1)
    y = labels.astype(np.float64).reshape(-1)
    counts = np.bincount(c, minlength=num_conditions)
    sums = np.bincount(c, weights=y, minlength=num_conditions)
    rates = sums / np.maximum(counts, 1)
    idx = np.arange(num_conditions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(idx, counts, color="#4477aa")
    axes[0].set_title("Samples per condition")
    axes[0].set_xlabel("Condition")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(idx, rates, color="#66c2a5")
    axes[1].set_title("Label base rate per condition")
    axes[1].set_xlabel("Condition")
    axes[1].set_ylabel("Mean label")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_le_stats_per_condition(stats: dict, title: str = "Localized Entropy terms per condition"):
    conds = sorted(stats.keys())
    nums = [stats[c]["Numerator"] for c in conds]
    dens = [stats[c]["Denominator"] for c in conds]
    ratios = [stats[c]["Numerator/denominator"] for c in conds]
    pj = [stats[c]["Average prediction for denominator"] for c in conds]
    n1 = [stats[c]["Number of samples with label 1"] for c in conds]
    n0 = [stats[c]["Number of samples with label 0"] for c in conds]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].bar(conds, ratios, color="#4477aa")
    axs[0, 0].set_title("Numerator / Denominator")
    axs[0, 0].set_xlabel("Condition")
    axs[0, 0].set_ylabel("Ratio")
    axs[0, 0].grid(True, alpha=0.3)

    x = np.arange(len(conds))
    width = 0.4
    axs[0, 1].bar(x - width / 2, nums, width=width, label="Numerator", color="#66c2a5")
    axs[0, 1].bar(x + width / 2, dens, width=width, label="Denominator", color="#fc8d62")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(conds)
    axs[0, 1].set_title("Numerator vs Denominator")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].bar(conds, n0, label="Label 0 count", color="#999999")
    axs[1, 0].bar(conds, n1, bottom=n0, label="Label 1 count", color="#1b9e77")
    axs[1, 0].set_title("Label counts per condition")
    axs[1, 0].set_xlabel("Condition")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].bar(conds, pj, color="#8da0cb")
    axs[1, 1].set_title("Average prediction for denominator (p_j)")
    axs[1, 1].set_xlabel("Condition")
    axs[1, 1].set_ylabel("p_j")
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_ctr_filter_stats(stats_df, labels, filter_col: str) -> None:
    if stats_df is None:
        return
    print("Top-filter stats (click):")
    print(stats_df.to_string())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(labels, stats_df["frequency"], color="#4477aa")
    axes[0].set_title("Frequency")
    axes[0].set_xlabel(filter_col)
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    mean_col = "mean" if "mean" in stats_df.columns else "median"
    axes[1].bar(labels, stats_df[mean_col], color="#66c2a5")
    axes[1].set_title("Mean Click Rate" if mean_col == "mean" else "Median Click")
    axes[1].set_xlabel(filter_col)
    axes[1].set_ylabel("Rate")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(labels, stats_df["std"], color="#fc8d62")
    axes[2].set_title("Std Dev Click")
    axes[2].set_xlabel(filter_col)
    axes[2].set_ylabel("Std Dev")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pred_to_train_rate(
    plot_df,
    *,
    condition_label: str,
    eval_name: str,
) -> None:
    if plot_df is None or len(plot_df) == 0:
        print("[WARN] Plot data unavailable; skipping pred/train ratio chart.")
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.bar(
        plot_df[condition_label].astype(np.int64),
        plot_df["pred_to_train_rate"],
        color="#4477aa",
    )
    ax.axhline(1.0, color="#d95f02", linestyle="--", linewidth=1)
    ax.set_title(
        f"{eval_name} avg prediction / train click rate by {condition_label} (top {len(plot_df)})"
    )
    ax.set_xlabel(condition_label)
    ax.set_ylabel("Prediction / Train Click Rate")
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_per_ad_f1_logp(
    per_ad_df,
    *,
    condition_label: str,
    name: str = "Eval",
    bins: int = 60,
) -> None:
    if per_ad_df is None or len(per_ad_df) == 0:
        print("[WARN] No per-condition metrics available; skipping F1 log plot.")
        return
    if "f1" not in per_ad_df.columns:
        print("[WARN] Missing per-condition F1 column; skipping F1 log plot.")
        return
    f1_scores = np.asarray(per_ad_df["f1"], dtype=np.float64)
    if not np.isfinite(f1_scores).any():
        print("[WARN] F1 scores are not finite; skipping F1 log plot.")
        return
    eps = 1e-12
    log10_f1 = np.log10(np.clip(f1_scores, eps, 1.0))
    finite = np.isfinite(log10_f1)
    if not finite.any():
        print("[WARN] Log10(F1) is empty after filtering; skipping plot.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(log10_f1[finite], bins=bins, color="#4477aa", alpha=0.85)
    plt.title(f"{name} per-{condition_label} F1 score (log10)")
    plt.xlabel("log10(F1 score)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def build_eval_epoch_plotter(
    train_eval_name: str,
    train_eval_conds: Optional[np.ndarray],
    num_conditions: int,
    eval_value_range: Tuple[float, float],
    condition_label: str,
    loss_label: str,
):
    def _plot(preds: np.ndarray, epoch: int) -> None:
        if preds.size == 0:
            print(f"Epoch {epoch}: {train_eval_name} set empty; skipping eval plots.")
            return
        if train_eval_conds is None:
            print(f"Epoch {epoch}: {train_eval_name} conditions unavailable; skipping eval plots.")
            return
        plot_eval_predictions_by_condition(
            preds,
            train_eval_conds,
            num_conditions,
            value_range=eval_value_range,
            title=(
                f"Epoch {epoch} {train_eval_name} Predictions by {condition_label} ("
                f"{loss_label})"
            ),
            print_counts=False,
        )
    return _plot


def build_eval_batch_plotter(
    train_eval_name: str,
    train_eval_conds: Optional[np.ndarray],
    num_conditions: int,
    eval_value_range: Tuple[float, float],
    condition_label: str,
    loss_label: str,
):
    def _plot(preds: np.ndarray, epoch: int, batch_idx: int) -> None:
        if preds.size == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx}: {train_eval_name} set empty; skipping eval plots."
            )
            return
        if train_eval_conds is None:
            print(
                f"Epoch {epoch} Batch {batch_idx}: {train_eval_name} conditions unavailable; skipping eval plots."
            )
            return
        plot_eval_predictions_by_condition(
            preds,
            train_eval_conds,
            num_conditions,
            value_range=eval_value_range,
            title=(
                f"Epoch {epoch} Batch {batch_idx} {train_eval_name} Predictions by {condition_label} ("
                f"{loss_label})"
            ),
            print_counts=False,
        )
    return _plot


def plot_metric_comparison_table(
    df,
    *,
    columns,
    title: str,
    max_rows: int = 20,
    output_path: Optional[str] = None,
    float_format: str = "{:.4g}",
    show: bool = True,
) -> None:
    if df is None or len(df) == 0:
        print("[WARN] No data for comparison table; skipping plot.")
        return
    max_rows = max(int(max_rows), 1)
    columns = list(columns)
    table_df = df.loc[:, columns].head(max_rows)
    cell_text = []
    for _, row in table_df.iterrows():
        row_values = []
        for col in columns:
            val = row[col]
            if isinstance(val, (np.integer, int)):
                row_values.append(str(int(val)))
            elif isinstance(val, (np.floating, float)):
                if np.isnan(val):
                    row_values.append("nan")
                else:
                    row_values.append(float_format.format(float(val)))
            else:
                row_values.append(str(val))
        cell_text.append(row_values)

    width = max(8.0, 1.2 * len(columns))
    height = max(2.0, 0.4 * (len(table_df) + 1))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    ax.set_title(title)
    fig.tight_layout()
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
