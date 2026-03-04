from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None


def configure_plot_theme(theme_cfg: Optional[dict] = None) -> None:
    """Configure plotting theme, preferring seaborn when available."""
    cfg = theme_cfg or {}
    use_seaborn = bool(cfg.get("enabled", True))
    style = str(cfg.get("style", "whitegrid"))
    context = str(cfg.get("context", "notebook"))
    palette = str(cfg.get("palette", "deep"))
    font_scale = float(cfg.get("font_scale", 1.0))
    despine = bool(cfg.get("despine", False))

    if use_seaborn and sns is not None:
        sns.set_theme(
            style=style,
            context=context,
            palette=palette,
            font_scale=font_scale,
        )
        if despine:
            sns.despine()
        return
    plt.style.use("seaborn-v0_8")


configure_plot_theme()


def _resolve_max_plot_conditions(max_plot_conditions: Optional[int]) -> Optional[int]:
    """Resolve per-condition plot limit; return None to disable the limit."""
    if max_plot_conditions is None:
        return 20
    try:
        limit = int(max_plot_conditions)
    except (TypeError, ValueError):
        return 20
    if limit <= 0:
        return None
    return limit


def _skip_per_condition_plot(
    *,
    plot_name: str,
    num_conditions: int,
    max_plot_conditions: Optional[int],
) -> bool:
    """Return True when a per-condition plot should be skipped."""
    limit = _resolve_max_plot_conditions(max_plot_conditions)
    if limit is None:
        return False
    if int(num_conditions) > limit:
        print(
            f"[INFO] Skipping {plot_name}: num_conditions={int(num_conditions)} "
            f"exceeds max_plot_conditions={limit}."
        )
        return True
    return False


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
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot per-condition density or count lines for a feature array."""
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
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    plt.show()


def plot_training_distributions(
    net_worth: np.ndarray,
    ages: np.ndarray,
    probs: np.ndarray,
    conds: np.ndarray,
    num_conditions: int,
) -> None:
    """Plot synthetic training distributions by condition."""
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
    """Plot a histogram of log10 predicted probabilities."""
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


def plot_loss_curves(
    train_losses,
    eval_losses,
    loss_label: str,
    *,
    eval_batch_losses=None,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot train/eval loss curves, optionally including batch evals."""
    plt.figure(figsize=(8, 5))
    epochs = np.arange(len(train_losses))
    plt.plot(epochs, train_losses, label=f"Train {loss_label}")
    plt.plot(epochs, eval_losses, label=f"Eval {loss_label}")
    if eval_batch_losses:
        batch_x = []
        batch_eval = []
        batch_train = []
        if train_losses and eval_losses:
            batch_x.append(0.0)
            batch_train.append(float(train_losses[0]))
            batch_eval.append(float(eval_losses[0]))
        for idx, item in enumerate(eval_batch_losses, start=1):
            x_val = item.get("epoch_progress") or item.get("epoch") or idx
            batch_x.append(x_val)
            batch_eval.append(float(item.get("loss", float("nan"))))
            batch_train.append(float(item.get("train_loss", float("nan"))))
        plt.plot(
            batch_x,
            batch_train,
            label=f"Train {loss_label} (batch)",
            linestyle="--",
            marker="o",
            markersize=3,
            linewidth=1.0,
            alpha=0.7,
        )
        plt.plot(
            batch_x,
            batch_eval,
            label=f"Eval {loss_label} (batch)",
            linestyle="--",
            marker="o",
            markersize=3,
            linewidth=1.0,
            alpha=0.7,
        )
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss_label} Loss")
    plt.title(f"Training vs Evaluation {loss_label} over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
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
    max_plot_conditions: Optional[int] = 20,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot per-condition prediction distributions."""
    if _skip_per_condition_plot(
        plot_name="eval predictions by condition",
        num_conditions=int(num_conditions),
        max_plot_conditions=max_plot_conditions,
    ):
        return
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
        output_path=output_path,
    )


def plot_calibration_ratio_by_condition(
    base_rates: np.ndarray,
    calibration: np.ndarray,
    *,
    name: str = "Eval",
    condition_label: str = "Condition",
    title: Optional[str] = None,
    max_plot_conditions: Optional[int] = 20,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot calibration ratio by condition base rate."""
    rates = np.asarray(base_rates, dtype=np.float64).reshape(-1)
    ratios = np.asarray(calibration, dtype=np.float64).reshape(-1)
    mask = np.isfinite(rates) & np.isfinite(ratios) & (rates > 0)
    if not mask.any():
        print("[WARN] No finite calibration ratios available; skipping calibration plot.")
        return
    x = rates[mask]
    y = ratios[mask]
    if _skip_per_condition_plot(
        plot_name="calibration ratio by condition",
        num_conditions=int(x.size),
        max_plot_conditions=max_plot_conditions,
    ):
        return
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if title is None:
        title = f"{name} Calibration Ratio by {condition_label}"
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linestyle="-", color="#4477aa", alpha=0.9)
    plt.axhline(1.0, color="#cc6677", linestyle="--", linewidth=1.0)
    plt.xscale("log")
    plt.title(title)
    plt.xlabel(f"{condition_label} base rate")
    plt.ylabel("Calibration ratio (pred_mean / base_rate)")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    plt.show()


def plot_reliability_diagram(
    ece_table: pd.DataFrame,
    *,
    name: str = "Eval",
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot a reliability diagram from an ECE bin table."""
    if ece_table is None or len(ece_table) == 0:
        print("[WARN] Calibration table is empty; skipping reliability diagram.")
        return
    needed = {"avg_pred", "avg_label", "count"}
    if not needed.issubset(set(ece_table.columns)):
        print("[WARN] Calibration table missing required columns; skipping reliability diagram.")
        return
    df = ece_table.copy()
    df = df[np.isfinite(df["avg_pred"]) & np.isfinite(df["avg_label"]) & (df["count"] > 0)]
    if df.empty:
        print("[WARN] No finite calibration bins available; skipping reliability diagram.")
        return
    df = df.sort_values("avg_pred")
    if title is None:
        title = f"{name} Reliability Diagram"

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    if sns is not None:
        sns.lineplot(data=df, x="avg_pred", y="avg_label", marker="o", linewidth=2.0, ax=ax)
        sns.scatterplot(
            data=df,
            x="avg_pred",
            y="avg_label",
            size="count",
            sizes=(40, 240),
            legend=False,
            ax=ax,
        )
    else:
        ax.plot(df["avg_pred"], df["avg_label"], marker="o", linewidth=2.0, color="#4477aa")
        sizes = np.sqrt(df["count"].to_numpy(dtype=np.float64)) * 3.0
        ax.scatter(df["avg_pred"], df["avg_label"], s=sizes, color="#4477aa", alpha=0.85)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#cc6677", linewidth=1.5, label="Perfect")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    plt.show()


def plot_reliability_comparison(
    reliability_tables: dict,
    *,
    title: str = "Reliability Diagram Comparison",
) -> None:
    """Plot a multi-model reliability comparison."""
    if not reliability_tables:
        print("[WARN] No reliability tables available; skipping comparison plot.")
        return
    frames = []
    for model_name, table in reliability_tables.items():
        if table is None or len(table) == 0:
            continue
        needed = {"avg_pred", "avg_label", "count"}
        if not needed.issubset(set(table.columns)):
            continue
        df = table.copy()
        df = df[np.isfinite(df["avg_pred"]) & np.isfinite(df["avg_label"]) & (df["count"] > 0)]
        if df.empty:
            continue
        df = df.sort_values("avg_pred")
        df["model"] = str(model_name)
        frames.append(df)
    if not frames:
        print("[WARN] Reliability tables have no finite bins; skipping comparison plot.")
        return
    plot_df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    if sns is not None:
        sns.lineplot(
            data=plot_df,
            x="avg_pred",
            y="avg_label",
            hue="model",
            marker="o",
            linewidth=2.0,
            ax=ax,
        )
    else:
        for model_name, group in plot_df.groupby("model"):
            ax.plot(group["avg_pred"], group["avg_label"], marker="o", linewidth=2.0, label=model_name)
        ax.legend()
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#cc6677", linewidth=1.5, label="Perfect")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_model_metric_summary(
    metrics_df: pd.DataFrame,
    *,
    model_col: str = "model",
    title: str = "Model Metric Summary",
    max_metrics: int = 8,
) -> None:
    """Plot key metrics as small-multiple bar charts for each model."""
    if metrics_df is None or len(metrics_df) == 0:
        print("[WARN] Metrics table is empty; skipping metric summary plot.")
        return
    if model_col not in metrics_df.columns:
        print("[WARN] Metrics table missing model column; skipping metric summary plot.")
        return
    metric_cols = [c for c in metrics_df.columns if c != model_col]
    if not metric_cols:
        print("[WARN] Metrics table has no metric columns; skipping metric summary plot.")
        return
    metric_cols = metric_cols[: max(1, int(max_metrics))]
    plot_df = metrics_df.loc[:, [model_col] + metric_cols].copy()
    long_df = plot_df.melt(id_vars=[model_col], value_vars=metric_cols, var_name="metric", value_name="value")
    long_df = long_df[np.isfinite(long_df["value"])]
    if long_df.empty:
        print("[WARN] Metrics are not finite; skipping metric summary plot.")
        return
    n_metrics = len(metric_cols)
    ncols = min(4, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metric_cols):
        ax = axes_flat[idx]
        metric_slice = long_df[long_df["metric"] == metric]
        if metric_slice.empty:
            ax.axis("off")
            continue
        if sns is not None:
            sns.barplot(data=metric_slice, x=model_col, y="value", hue=model_col, ax=ax)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        else:
            x = np.arange(len(metric_slice))
            ax.bar(x, metric_slice["value"].to_numpy(dtype=np.float64), color="#4477aa")
            ax.set_xticks(x)
            ax.set_xticklabels(metric_slice[model_col].astype(str).tolist(), rotation=20)
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_bce_le_calibration_comparison(
    comparison_df: pd.DataFrame,
    *,
    condition_label: str,
    top_k: int = 20,
    title: Optional[str] = None,
) -> None:
    """Plot BCE-vs-LE calibration gap diagnostics across conditions."""
    if comparison_df is None or len(comparison_df) == 0:
        print("[WARN] No comparison data available; skipping calibration comparison plot.")
        return
    required = {condition_label, "base_rate", "bce_calibration", "le_calibration"}
    if not required.issubset(set(comparison_df.columns)):
        print("[WARN] Comparison data missing calibration columns; skipping plot.")
        return
    df = comparison_df.loc[:, [condition_label, "base_rate", "bce_calibration", "le_calibration"]].copy()
    df = df[np.isfinite(df["base_rate"]) & (df["base_rate"] > 0)]
    df = df[np.isfinite(df["bce_calibration"]) & np.isfinite(df["le_calibration"])]
    if df.empty:
        print("[WARN] No finite calibration rows available; skipping comparison plot.")
        return
    df["bce_abs_gap"] = np.abs(1.0 - df["bce_calibration"])
    df["le_abs_gap"] = np.abs(1.0 - df["le_calibration"])
    top_k = int(top_k)
    if top_k > 0 and len(df) > top_k:
        df = df.nlargest(top_k, "base_rate")
    df = df.sort_values("base_rate")

    if title is None:
        title = f"BCE vs LE Calibration Comparison by {condition_label}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scatter_ax, ratio_ax = axes

    if sns is not None:
        sns.scatterplot(
            data=df,
            x="bce_abs_gap",
            y="le_abs_gap",
            hue="base_rate",
            size="base_rate",
            sizes=(40, 240),
            palette="viridis",
            ax=scatter_ax,
        )
    else:
        scatter_ax.scatter(df["bce_abs_gap"], df["le_abs_gap"], color="#4477aa", alpha=0.8)
    max_gap = max(
        float(np.nanmax(df["bce_abs_gap"])) if len(df) else 0.0,
        float(np.nanmax(df["le_abs_gap"])) if len(df) else 0.0,
    )
    max_gap = max(max_gap, 1e-6)
    scatter_ax.plot([0.0, max_gap], [0.0, max_gap], linestyle="--", color="#cc6677", linewidth=1.2)
    scatter_ax.set_xlabel("BCE |1 - calibration|")
    scatter_ax.set_ylabel("LE |1 - calibration|")
    scatter_ax.set_title("Calibration Abs Gap (lower is better)")
    scatter_ax.grid(True, alpha=0.3)

    ratio_df = df.melt(
        id_vars=[condition_label, "base_rate"],
        value_vars=["bce_calibration", "le_calibration"],
        var_name="loss",
        value_name="calibration_ratio",
    )
    ratio_df["loss"] = ratio_df["loss"].map(
        {"bce_calibration": "BCE", "le_calibration": "LE"}
    ).fillna(ratio_df["loss"])
    if sns is not None:
        sns.lineplot(
            data=ratio_df,
            x="base_rate",
            y="calibration_ratio",
            hue="loss",
            marker="o",
            ax=ratio_ax,
        )
    else:
        for loss_name, group in ratio_df.groupby("loss"):
            ratio_ax.plot(group["base_rate"], group["calibration_ratio"], marker="o", label=loss_name)
        ratio_ax.legend()
    ratio_ax.axhline(1.0, color="#cc6677", linestyle="--", linewidth=1.2)
    ratio_ax.set_xscale("log")
    ratio_ax.set_xlabel(f"{condition_label} base rate")
    ratio_ax.set_ylabel("Calibration ratio (pred_mean / base_rate)")
    ratio_ax.set_title("Calibration Ratio vs Base Rate")
    ratio_ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


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
    max_plot_conditions: Optional[int] = 20,
) -> None:
    """Plot selected feature distributions by condition."""
    if _skip_per_condition_plot(
        plot_name="feature distributions by condition",
        num_conditions=int(num_conditions),
        max_plot_conditions=max_plot_conditions,
    ):
        return
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
    *,
    max_plot_conditions: Optional[int] = 20,
) -> None:
    """Plot per-condition sample counts and label base rates."""
    if _skip_per_condition_plot(
        plot_name="label rates by condition",
        num_conditions=int(num_conditions),
        max_plot_conditions=max_plot_conditions,
    ):
        return
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


def plot_le_stats_per_condition(
    stats: dict,
    title: str = "Localized Entropy terms per condition",
    *,
    max_plot_conditions: Optional[int] = 20,
):
    """Plot LE numerator/denominator stats per condition."""
    conds = sorted(stats.keys())
    if _skip_per_condition_plot(
        plot_name="LE stats per condition",
        num_conditions=len(conds),
        max_plot_conditions=max_plot_conditions,
    ):
        return
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


def plot_grad_sq_sums_by_condition(
    bce_sums: Optional[np.ndarray],
    le_sums: Optional[np.ndarray],
    *,
    condition_label: str,
    title: str = "Gradient mean square per condition",
    top_k: int = 0,
    log10: bool = True,
    max_plot_conditions: Optional[int] = 20,
) -> None:
    """Plot per-condition gradient mean-square values for BCE vs LE."""
    if bce_sums is None and le_sums is None:
        print("[WARN] No gradient stats available; skipping grad plot.")
        return
    bce_vals = np.asarray(bce_sums, dtype=np.float64).reshape(-1) if bce_sums is not None else None
    le_vals = np.asarray(le_sums, dtype=np.float64).reshape(-1) if le_sums is not None else None
    if (bce_vals is not None) and (le_vals is not None) and (bce_vals.size != le_vals.size):
        print("[WARN] Grad arrays have different sizes; skipping grad plot.")
        return
    num_conditions = int(bce_vals.size if bce_vals is not None else le_vals.size)
    if _skip_per_condition_plot(
        plot_name="gradient mean square by condition",
        num_conditions=num_conditions,
        max_plot_conditions=max_plot_conditions,
    ):
        return
    cond_ids = np.arange(num_conditions)
    top_k = int(top_k)
    if top_k > 0 and top_k < num_conditions:
        if bce_vals is not None and le_vals is not None:
            scores = np.maximum(bce_vals, le_vals)
        else:
            scores = bce_vals if bce_vals is not None else le_vals
        order = np.argsort(scores)[::-1][:top_k]
        cond_ids = cond_ids[order]
        if bce_vals is not None:
            bce_vals = bce_vals[order]
        if le_vals is not None:
            le_vals = le_vals[order]

    def _transform(vals: np.ndarray) -> np.ndarray:
        """Optionally log10-transform values for plotting."""
        if not log10:
            return vals
        eps = 1e-12
        return np.log10(np.clip(vals, eps, None))

    bce_plot = _transform(bce_vals) if bce_vals is not None else None
    le_plot = _transform(le_vals) if le_vals is not None else None
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    x = np.arange(len(cond_ids))
    if bce_plot is not None and le_plot is not None:
        width = 0.4
        ax.bar(x - width / 2, bce_plot, width=width, label="BCE", color="#4477aa")
        ax.bar(x + width / 2, le_plot, width=width, label="LE", color="#66c2a5")
        ax.legend()
    else:
        label = "BCE" if bce_plot is not None else "LE"
        vals = bce_plot if bce_plot is not None else le_plot
        color = "#4477aa" if bce_plot is not None else "#66c2a5"
        ax.bar(x, vals, color=color, label=label)
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(condition_label)
    y_label = "log10(mean grad^2)" if log10 else "mean grad^2"
    ax.set_ylabel(y_label)
    if len(cond_ids) > 20:
        ax.set_xticks(x, cond_ids, rotation=90)
    else:
        ax.set_xticks(x, cond_ids)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_ctr_filter_stats(stats_df, labels, filter_col: str) -> None:
    """Plot CTR filter summary stats."""
    if stats_df is None:
        return
    print("Filter stats (click):")
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
    max_plot_conditions: Optional[int] = 20,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot eval prediction averages vs train base rates."""
    if plot_df is None or len(plot_df) == 0:
        print("[WARN] Plot data unavailable; skipping pred/train ratio chart.")
        return
    if _skip_per_condition_plot(
        plot_name="prediction to train-rate by condition",
        num_conditions=len(plot_df),
        max_plot_conditions=max_plot_conditions,
    ):
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
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    plt.show()


def plot_per_ad_f1_logp(
    per_ad_df,
    *,
    condition_label: str,
    name: str = "Eval",
    bins: int = 60,
) -> None:
    """Plot log10(F1) distribution across conditions."""
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
    """Build a per-epoch prediction plot callback."""
    def _plot(preds: np.ndarray, epoch: int) -> None:
        """Plot eval predictions for a single epoch."""
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
    """Build a per-batch prediction plot callback."""
    def _plot(preds: np.ndarray, epoch: int, batch_idx: int) -> None:
        """Plot eval predictions for a single batch checkpoint."""
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
    """Render a DataFrame as a matplotlib table."""
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
