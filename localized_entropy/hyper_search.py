from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from localized_entropy.analysis import (
    expected_calibration_error,
    per_condition_calibration,
    per_condition_log_ratio_calibration_error,
    per_condition_metrics,
)
from localized_entropy.config import get_data_source, load_and_resolve, resolve_ctr_config
from localized_entropy.data.pipeline import prepare_data
from localized_entropy.experiments import build_loss_loaders, build_model, train_single_loss
from localized_entropy.training import compute_base_rates_from_loader, evaluate, predict_probs
from localized_entropy.utils import init_device, set_seed


@dataclass
class SearchContext:
    cfg: dict
    device: torch.device
    use_cuda: bool
    use_mps: bool
    non_blocking: bool
    model_dtype: torch.dtype
    splits: Any
    loss_loaders: Any
    le_train_cfg: dict
    data_source: str
    eval_loader: Any
    eval_labels: np.ndarray
    eval_conds: np.ndarray
    eval_name: str
    base_rates_train: np.ndarray


def resolve_lr_category(train_cfg: dict) -> Optional[float]:
    if not isinstance(train_cfg, dict):
        return None
    if "lr_category" in train_cfg:
        return train_cfg.get("lr_category")
    return train_cfg.get("LRCategory")


def _global_calibration_ratio(preds: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    pred_mean = float(np.mean(preds))
    label_mean = float(np.mean(labels))
    if label_mean <= eps:
        return float("nan")
    return pred_mean / label_mean


def collect_all_data_metrics(
    cfg: dict,
    model: torch.nn.Module,
    *,
    eval_loader: Any,
    eval_labels: np.ndarray,
    eval_conds: np.ndarray,
    device: torch.device,
    non_blocking: bool,
) -> tuple[dict, pd.DataFrame]:
    eval_preds = predict_probs(model, eval_loader, device, non_blocking=non_blocking)
    eval_preds = np.asarray(eval_preds, dtype=np.float64).reshape(-1)
    if eval_preds.shape[0] != eval_labels.shape[0] or eval_preds.shape[0] != eval_conds.shape[0]:
        raise ValueError(
            f"Mismatched eval lengths: preds={eval_preds.shape[0]} "
            f"labels={eval_labels.shape[0]} conds={eval_conds.shape[0]}"
        )
    global_calibration = _global_calibration_ratio(eval_preds, eval_labels)
    global_calibration_error = float(abs(global_calibration - 1.0)) if np.isfinite(global_calibration) else float("nan")

    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    ece_bins = int(eval_cfg.get("ece_bins", 20))
    ece_min_count = int(eval_cfg.get("ece_min_count", 1))
    ece_method = str(eval_cfg.get("ece_method", "custom"))
    ece_smooth_bandwidth = float(eval_cfg.get("ece_smooth_bandwidth", 0.05))
    ece_smooth_grid_bins = eval_cfg.get("ece_smooth_grid_bins")
    if ece_smooth_grid_bins is not None:
        ece_smooth_grid_bins = int(ece_smooth_grid_bins)
    small_prob_max = float(eval_cfg.get("small_prob_max", 0.01))

    ece, _ = expected_calibration_error(
        eval_preds,
        eval_labels,
        bins=ece_bins,
        min_count=ece_min_count,
        method=ece_method,
        smooth_bandwidth=ece_smooth_bandwidth,
        smooth_grid_bins=ece_smooth_grid_bins,
    )
    small_mask = eval_preds <= small_prob_max
    if np.any(small_mask):
        ece_small, _ = expected_calibration_error(
            eval_preds[small_mask],
            eval_labels[small_mask],
            bins=ece_bins,
            min_count=ece_min_count,
            method=ece_method,
            smooth_bandwidth=ece_smooth_bandwidth,
            smooth_grid_bins=ece_smooth_grid_bins,
        )
    else:
        ece_small = float("nan")

    per_cond_ratio = per_condition_calibration(eval_preds, eval_labels, eval_conds)
    per_cond_extra = per_condition_metrics(
        eval_preds,
        eval_labels,
        eval_conds,
        bins=ece_bins,
        min_count=ece_min_count,
        small_prob_max=small_prob_max,
        ece_method=ece_method,
        ece_smooth_bandwidth=ece_smooth_bandwidth,
        ece_smooth_grid_bins=ece_smooth_grid_bins,
    )
    per_cond = per_cond_ratio.merge(
        per_cond_extra[["condition", "ece", "ece_small"]],
        on="condition",
        how="left",
    )
    per_cond["calibration_abs_error"] = np.abs(per_cond["calibration"] - 1.0)

    per_cond_calibration_error, per_cond_calibration_error_macro = per_condition_log_ratio_calibration_error(
        eval_preds,
        eval_labels,
        eval_conds,
        min_count=1,
    )
    metrics = {
        "global_calibration": float(global_calibration),
        "global_calibration_error": float(global_calibration_error),
        "ece": float(ece),
        "ece_small": float(ece_small),
        "per_condition_calibration_error": float(per_cond_calibration_error),
        "per_condition_calibration_error_macro": float(per_cond_calibration_error_macro),
        "small_prob_max": float(small_prob_max),
    }
    return metrics, per_cond


def build_search_context(config_path: str) -> SearchContext:
    cfg = load_and_resolve(config_path)
    device_cfg = cfg.get("device", {})
    device, use_cuda, use_mps, non_blocking = init_device(use_mps=bool(device_cfg.get("use_mps", True)))
    cpu_float64 = device.type == "cpu" and not bool(device_cfg.get("use_mps", True))
    model_dtype = torch.float64 if cpu_float64 else torch.float32

    set_seed(int(cfg["project"]["seed"]), use_cuda)
    prepared = prepare_data(cfg, device, use_cuda, use_mps)
    splits = prepared.splits

    loss_loaders, le_train_cfg = build_loss_loaders(cfg, "localized_entropy", splits, device, use_cuda, use_mps)
    data_source = get_data_source(cfg)
    test_has_labels = not (data_source == "ctr" and not bool(resolve_ctr_config(cfg).get("test_has_labels", False)))
    if loss_loaders.test_loader is not None and splits.y_test is not None and test_has_labels:
        eval_loader = loss_loaders.test_loader
        eval_labels = np.asarray(splits.y_test).reshape(-1)
        eval_conds = np.asarray(splits.c_test).reshape(-1)
        eval_name = "test"
    else:
        eval_loader = loss_loaders.eval_loader
        eval_labels = np.asarray(splits.y_eval).reshape(-1)
        eval_conds = np.asarray(splits.c_eval).reshape(-1)
        eval_name = "eval"

    first_param_dtype = torch.float64 if cpu_float64 else torch.float32
    base_rates_train = compute_base_rates_from_loader(
        loss_loaders.train_loader,
        num_conditions=int(splits.num_conditions),
        device=device,
        dtype=first_param_dtype,
        non_blocking=non_blocking,
    )

    return SearchContext(
        cfg=cfg,
        device=device,
        use_cuda=use_cuda,
        use_mps=use_mps,
        non_blocking=non_blocking,
        model_dtype=model_dtype,
        splits=splits,
        loss_loaders=loss_loaders,
        le_train_cfg=le_train_cfg,
        data_source=data_source,
        eval_loader=eval_loader,
        eval_labels=eval_labels,
        eval_conds=eval_conds,
        eval_name=eval_name,
        base_rates_train=base_rates_train,
    )


def default_le_train_params(ctx: SearchContext) -> Dict[str, Any]:
    le_train_cfg = ctx.le_train_cfg
    return {
        "epochs": int(le_train_cfg.get("epochs", ctx.cfg["training"]["epochs"])),
        "lr": float(le_train_cfg.get("lr", ctx.cfg["training"]["lr"])),
        "lr_category": resolve_lr_category(le_train_cfg),
        "lr_decay": float(le_train_cfg.get("lr_decay", ctx.cfg.get("training", {}).get("lr_decay", 1.0))),
        "lr_category_decay": float(
            le_train_cfg.get(
                "lr_category_decay",
                ctx.cfg.get("training", {}).get("lr_category_decay", 1.0),
            )
        ),
        "lr_zero_after_epochs": le_train_cfg.get("lr_zero_after_epochs"),
        "le_cross_batch_cfg": copy.deepcopy(le_train_cfg.get("cross_batch"))
        if isinstance(le_train_cfg.get("cross_batch"), dict)
        else None,
    }


def run_le_hyper_search(
    ctx: SearchContext,
    *,
    param_grid: Sequence[Mapping[str, Any]],
    apply_params: Callable[[dict, Dict[str, Any], Mapping[str, Any]], None],
    record_params: Optional[Callable[[Mapping[str, Any]], Dict[str, Any]]] = None,
    sort_by: Sequence[str],
    ascending: Sequence[bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch_records: list[dict] = []
    condition_records: list[dict] = []

    if record_params is None:
        def record_params(p: Mapping[str, Any]) -> Dict[str, Any]:  # type: ignore[redefinition]
            return dict(p)

    for params in param_grid:
        cfg_run = copy.deepcopy(ctx.cfg)
        train_params = default_le_train_params(ctx)
        apply_params(cfg_run, train_params, params)

        set_seed(int(cfg_run["project"]["seed"]), ctx.use_cuda)
        model = build_model(cfg_run, ctx.splits, ctx.device, dtype=ctx.model_dtype)
        record_payload = record_params(params)

        def on_epoch_eval(_eval_preds: np.ndarray, epoch: int, payload: Dict[str, Any] = record_payload) -> None:
            le_loss, _ = evaluate(
                model,
                ctx.eval_loader,
                ctx.device,
                loss_mode="localized_entropy",
                base_rates=ctx.base_rates_train,
                non_blocking=ctx.non_blocking,
            )
            metrics, per_cond_df = collect_all_data_metrics(
                cfg_run,
                model,
                eval_loader=ctx.eval_loader,
                eval_labels=ctx.eval_labels,
                eval_conds=ctx.eval_conds,
                device=ctx.device,
                non_blocking=ctx.non_blocking,
            )
            row = dict(payload)
            row.update(
                {
                    "epoch": int(epoch),
                    "test_le": float(le_loss),
                    "global_calibration": float(metrics["global_calibration"]),
                    "global_calibration_error": float(metrics["global_calibration_error"]),
                    "ece": float(metrics["ece"]),
                    "ece_small": float(metrics["ece_small"]),
                    "per_condition_calibration_error": float(metrics["per_condition_calibration_error"]),
                    "per_condition_calibration_error_macro": float(metrics["per_condition_calibration_error_macro"]),
                    "small_prob_max": float(metrics["small_prob_max"]),
                }
            )
            epoch_records.append(row)

            for _, per_cond_row in per_cond_df.iterrows():
                cond_row = dict(payload)
                cond_row.update(
                    {
                        "epoch": int(epoch),
                        "condition": int(per_cond_row["condition"]),
                        "count": int(per_cond_row["count"]),
                        "base_rate": float(per_cond_row["base_rate"]),
                        "pred_mean": float(per_cond_row["pred_mean"]),
                        "calibration": float(per_cond_row["calibration"]),
                        "calibration_abs_error": float(per_cond_row["calibration_abs_error"]),
                        "ece": float(per_cond_row["ece"]),
                        "ece_small": float(per_cond_row["ece_small"]),
                    }
                )
                condition_records.append(cond_row)

        train_single_loss(
            model=model,
            loss_mode="localized_entropy",
            train_loader=ctx.loss_loaders.train_loader,
            train_eval_loader=ctx.eval_loader,
            eval_loader=ctx.eval_loader,
            device=ctx.device,
            epochs=train_params["epochs"],
            lr=train_params["lr"],
            lr_category=None if train_params["lr_category"] is None else float(train_params["lr_category"]),
            lr_decay=float(train_params["lr_decay"]),
            lr_category_decay=float(train_params["lr_category_decay"]),
            lr_zero_after_epochs=train_params["lr_zero_after_epochs"],
            eval_has_labels=True,
            le_base_rates_train=ctx.base_rates_train,
            le_base_rates_train_eval=ctx.base_rates_train,
            le_base_rates_eval=ctx.base_rates_train,
            non_blocking=ctx.non_blocking,
            eval_callback=on_epoch_eval,
            plot_eval_hist_epochs=False,
            print_embedding_table=False,
            le_cross_batch_cfg=train_params["le_cross_batch_cfg"],
        )

    results_df = pd.DataFrame(epoch_records)
    condition_df = pd.DataFrame(condition_records)
    if results_df.empty:
        raise RuntimeError("Search produced no results.")
    results_df = results_df.sort_values(list(sort_by), ascending=list(ascending)).reset_index(drop=True)
    return results_df, condition_df
