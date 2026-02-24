from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from localized_entropy.analysis import collect_logits
from localized_entropy.config import get_data_source, loss_label, resolve_training_cfg
from localized_entropy.data.pipeline import build_dataloaders
from localized_entropy.models import ConditionedLogitMLP
from localized_entropy.training import (
    GradSqStats,
    compute_base_rates_from_loader,
    evaluate,
    predict_probs,
    train_with_epoch_plots,
)
from localized_entropy.utils import set_seed


@dataclass
class TrainRunResult:
    loss_mode: str
    loss_label: str
    model: torch.nn.Module
    train_losses: List[float]
    eval_losses: List[float]
    eval_batch_losses: List[dict]
    eval_loss: float
    eval_preds: np.ndarray
    eval_logits: Optional[torch.Tensor]
    eval_targets: Optional[torch.Tensor]
    eval_conds: Optional[torch.Tensor]
    grad_sq_stats: Optional[GradSqStats]


def build_loss_loaders(
    cfg: dict,
    loss_mode: str,
    splits,
    device: torch.device,
    use_cuda: bool,
    use_mps: bool = False,
) -> Tuple[object, dict]:
    """Build dataloaders using per-loss training overrides when configured."""
    train_cfg = resolve_training_cfg(cfg, loss_mode)
    batch_size = train_cfg.get("batch_size", cfg.get("training", {}).get("batch_size"))
    loaders = build_dataloaders(splits, cfg, device, use_cuda, use_mps, batch_size=batch_size)
    return loaders, train_cfg


def select_eval_loader(eval_split: str, loaders: object) -> DataLoader:
    """Select eval loader for a split, falling back when test is unavailable."""
    split = str(eval_split).lower().strip()
    if split == "train":
        return loaders.train_loader
    if split == "test":
        return loaders.test_loader if loaders.test_loader is not None else loaders.eval_loader
    return loaders.eval_loader


def build_model(
    cfg: dict,
    splits,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> ConditionedLogitMLP:
    """Construct a ConditionedLogitMLP from config and data splits."""
    model_cfg = cfg["model"]
    hidden_sizes = model_cfg.get("hidden_sizes")
    if hidden_sizes is not None:
        if isinstance(hidden_sizes, (int, float)):
            hidden_sizes = (int(hidden_sizes),)
        else:
            hidden_sizes = tuple(int(v) for v in hidden_sizes)
    cat_embed_dim = model_cfg.get("cat_embed_dim", model_cfg.get("embed_dim", 16))
    activation = model_cfg.get("activation", "relu")
    norm = model_cfg.get("norm")
    dropout = model_cfg.get("dropout", 0.3)
    model = ConditionedLogitMLP(
        num_conditions=int(splits.num_conditions),
        num_numeric=int(splits.x_train.shape[1]),
        embed_dim=int(model_cfg.get("embed_dim", 16)),
        cat_dims=splits.cat_sizes,
        cat_embed_dim=cat_embed_dim,
        hidden_sizes=hidden_sizes,
        p_drop=dropout,
        activation=activation,
        norm=norm,
    )
    if dtype is None:
        return model.to(device)
    return model.to(device=device, dtype=dtype)


def resolve_eval_bundle(cfg, splits, loaders):
    """Resolve which split/loader/labels to use for evaluation."""
    eval_cfg = cfg.get("evaluation", {})
    eval_split = str(eval_cfg.get("split", "eval")).lower().strip()
    valid_splits = {"train", "eval", "test"}
    if eval_split not in valid_splits:
        print(f"[WARN] Unknown evaluation split '{eval_split}'; defaulting to 'eval'.")
        eval_split = "eval"

    data_source = get_data_source(cfg)
    test_labels_available = False
    if data_source == "ctr":
        test_labels_available = bool(cfg.get("ctr", {}).get("test_has_labels", False))

    if eval_split == "train":
        return "train", loaders.train_loader, splits.y_train, splits.c_train, "Train"
    if eval_split == "test":
        if loaders.test_loader is None:
            print("[WARN] Test loader unavailable; falling back to eval split.")
            return "eval", loaders.eval_loader, splits.y_eval, splits.c_eval, "Eval"
        use_test_labels = bool(eval_cfg.get("use_test_labels", False))
        if use_test_labels and not test_labels_available:
            print("[WARN] evaluation.use_test_labels enabled but test labels are unavailable; disabling.")
            use_test_labels = False
        labels = splits.y_test if use_test_labels else None
        if not use_test_labels:
            print("[INFO] Test labels are disabled; skipping label-based metrics.")
        return "test", loaders.test_loader, labels, splits.c_test, "Test"
    return "eval", loaders.eval_loader, splits.y_eval, splits.c_eval, "Eval"


def resolve_train_eval_bundle(
    eval_split: str,
    eval_loader: DataLoader,
    eval_labels: Optional[np.ndarray],
    eval_conds: Optional[np.ndarray],
    eval_name: str,
    loaders,
    splits,
) -> Tuple[DataLoader, Optional[np.ndarray], str]:
    """Resolve the eval loader used during training for plots/diagnostics."""
    train_eval_loader = loaders.eval_loader
    train_eval_conds = splits.c_eval
    train_eval_name = "Eval"
    if eval_split == "train":
        train_eval_loader = loaders.train_loader
        train_eval_conds = splits.c_train
        train_eval_name = "Train"
    elif eval_split == "test":
        if eval_labels is not None:
            train_eval_loader = eval_loader
            train_eval_conds = eval_conds
            train_eval_name = eval_name
        else:
            print("[INFO] Training eval uses Eval split; test labels unavailable.")
    return train_eval_loader, train_eval_conds, train_eval_name


def evaluate_or_predict(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    loss_mode: str,
    eval_has_labels: bool,
    base_rates: Optional[np.ndarray] = None,
    focal_alpha: Optional[float] = None,
    focal_gamma: Optional[float] = None,
    non_blocking: bool = False,
) -> Tuple[float, np.ndarray]:
    """Run eval loss if labels exist, otherwise return predictions only."""
    if eval_has_labels:
        return evaluate(
            model,
            eval_loader,
            device,
            loss_mode=loss_mode,
            base_rates=base_rates,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            non_blocking=non_blocking,
        )
    preds = predict_probs(
        model,
        eval_loader,
        device,
        non_blocking=non_blocking,
    )
    return float("nan"), preds


def train_single_loss(
    *,
    model: torch.nn.Module,
    loss_mode: str,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lr_decay: float = 1.0,
    lr_category_decay: float = 1.0,
    lr_category: Optional[float] = None,
    lr_zero_after_epochs: Optional[int] = None,
    eval_has_labels: bool,
    le_base_rates_train: Optional[np.ndarray] = None,
    le_base_rates_train_eval: Optional[np.ndarray] = None,
    le_base_rates_eval: Optional[np.ndarray] = None,
    focal_alpha: Optional[float] = None,
    focal_gamma: Optional[float] = None,
    non_blocking: bool = False,
    plot_eval_hist_epochs: bool = False,
    eval_callback: Optional[Callable[[np.ndarray, int], None]] = None,
    eval_every_n_batches: Optional[int] = None,
    eval_batch_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
    collect_eval_logits: bool = False,
    collect_grad_sq_sums: bool = False,
    collect_eval_batch_losses: bool = False,
    debug_gradients: bool = False,
    debug_le_inputs: bool = False,
    le_cross_batch_cfg: Optional[dict] = None,
    print_embedding_table: bool = False,
) -> TrainRunResult:
    """Train one model/loss mode and collect evaluation outputs."""
    base_rates_train = le_base_rates_train
    base_rates_train_eval = le_base_rates_train_eval
    base_rates_eval = le_base_rates_eval
    if loss_mode == "localized_entropy":
        if base_rates_train is None:
            num_conds_model = getattr(model, "embedding").num_embeddings if hasattr(model, "embedding") else 1
            first_param = next(model.parameters(), None)
            p_dtype = first_param.dtype if first_param is not None else torch.float32
            base_rates_train = compute_base_rates_from_loader(
                train_loader,
                num_conditions=num_conds_model,
                device=device,
                dtype=p_dtype,
                non_blocking=non_blocking,
            )
        if base_rates_train_eval is None:
            base_rates_train_eval = base_rates_train
        if base_rates_eval is None:
            base_rates_eval = base_rates_train
    train_losses, eval_losses, grad_sq_stats, eval_batch_losses = train_with_epoch_plots(
        model=model,
        train_loader=train_loader,
        val_loader=train_eval_loader,
        device=device,
        epochs=int(epochs),
        lr=float(lr),
        lr_decay=float(lr_decay),
        lr_category_decay=float(lr_category_decay),
        category_lr=None if lr_category is None else float(lr_category),
        lr_zero_after_epochs=lr_zero_after_epochs,
        loss_mode=loss_mode,
        base_rates_train=base_rates_train,
        base_rates_eval=base_rates_train_eval,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        non_blocking=non_blocking,
        plot_eval_hist_epochs=plot_eval_hist_epochs,
        eval_callback=eval_callback,
        eval_every_n_batches=eval_every_n_batches,
        eval_batch_callback=eval_batch_callback,
        track_eval_batch_losses=collect_eval_batch_losses,
        track_grad_sq_sums=collect_grad_sq_sums,
        debug_gradients=debug_gradients,
        debug_le_inputs=debug_le_inputs,
        le_cross_batch_cfg=le_cross_batch_cfg,
        print_embedding_table=print_embedding_table,
    )
    eval_loss, eval_preds = evaluate_or_predict(
        model,
        eval_loader,
        device,
        loss_mode=loss_mode,
        eval_has_labels=eval_has_labels,
        base_rates=base_rates_eval,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        non_blocking=non_blocking,
    )
    eval_logits = eval_targets = eval_conds = None
    if collect_eval_logits:
        eval_logits, eval_targets, eval_conds = collect_logits(
            model, eval_loader, device, non_blocking=non_blocking
        )
    return TrainRunResult(
        loss_mode=loss_mode,
        loss_label=loss_label(loss_mode),
        model=model,
        train_losses=train_losses,
        eval_losses=eval_losses,
        eval_batch_losses=eval_batch_losses,
        eval_loss=float(eval_loss),
        eval_preds=eval_preds,
        eval_logits=eval_logits,
        eval_targets=eval_targets,
        eval_conds=eval_conds,
        grad_sq_stats=grad_sq_stats,
    )


def build_seed_sequence(base_seed: int, num_runs: int, seed_stride: int = 1) -> List[int]:
    """Build a deterministic list of seeds for repeated runs."""
    if num_runs < 1:
        return []
    stride = int(seed_stride)
    return [int(base_seed) + stride * idx for idx in range(int(num_runs))]


def run_repeated_loss_experiments(
    *,
    cfg: dict,
    loss_modes: List[str],
    splits,
    eval_split: str,
    eval_labels: Optional[np.ndarray],
    eval_conds: Optional[np.ndarray],
    eval_name: str,
    device: torch.device,
    use_cuda: bool,
    use_mps: bool = False,
    model_dtype: Optional[torch.dtype] = None,
    seeds: Iterable[int],
    le_base_rates_train: Optional[np.ndarray] = None,
    le_base_rates_train_eval: Optional[np.ndarray] = None,
    le_base_rates_eval: Optional[np.ndarray] = None,
    non_blocking: bool = False,
    collect_eval_logits: bool = False,
) -> Dict[str, List[TrainRunResult]]:
    """Run repeated training for each loss mode and seed."""
    def _resolve_lr_category(train_cfg: dict) -> Optional[float]:
        if not isinstance(train_cfg, dict):
            return None
        if "lr_category" in train_cfg:
            return train_cfg.get("lr_category")
        return train_cfg.get("LRCategory")

    eval_has_labels = eval_labels is not None
    data_source = get_data_source(cfg)
    per_loss = {}
    for loss_mode in loss_modes:
        loss_loaders, loss_train_cfg = build_loss_loaders(cfg, loss_mode, splits, device, use_cuda, use_mps)
        focal_alpha = None
        focal_gamma = None
        focal_cfg = loss_train_cfg.get("focal") if isinstance(loss_train_cfg, dict) else None
        if isinstance(focal_cfg, dict):
            focal_alpha = focal_cfg.get("alpha")
            focal_gamma = focal_cfg.get("gamma")
        loss_eval_loader = select_eval_loader(eval_split, loss_loaders)
        loss_train_eval_loader, _, _ = resolve_train_eval_bundle(
            eval_split,
            loss_eval_loader,
            eval_labels,
            eval_conds,
            eval_name,
            loss_loaders,
            splits,
        )
        per_loss[loss_mode] = {
            "train_cfg": loss_train_cfg,
            "loaders": loss_loaders,
            "eval_loader": loss_eval_loader,
            "train_eval_loader": loss_train_eval_loader,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
        }
    results: Dict[str, List[TrainRunResult]] = {loss_mode: [] for loss_mode in loss_modes}
    for seed in seeds:
        for loss_mode in loss_modes:
            set_seed(int(seed), use_cuda)
            model = build_model(cfg, splits, device, dtype=model_dtype)
            loss_bundle = per_loss[loss_mode]
            train_cfg = loss_bundle["train_cfg"]
            debug_gradients = bool(
                train_cfg.get(
                    "debug_gradients",
                    cfg.get("training", {}).get("debug_gradients", False),
                )
            )
            debug_le_inputs = bool(
                train_cfg.get(
                    "debug_le_inputs",
                    cfg.get("training", {}).get("debug_le_inputs", False),
                )
            )
            lr_category = None
            lr_zero_after_epochs = None
            if data_source == "synthetic" and loss_mode == "localized_entropy":
                lr_category = _resolve_lr_category(train_cfg)
                lr_zero_after_epochs = train_cfg.get("lr_zero_after_epochs")
            le_cross_batch_cfg = None
            if loss_mode == "localized_entropy" and isinstance(train_cfg, dict):
                if isinstance(train_cfg.get("cross_batch"), dict):
                    le_cross_batch_cfg = train_cfg.get("cross_batch")
                else:
                    le_cfg = train_cfg.get("localized_entropy")
                    if isinstance(le_cfg, dict):
                        le_cross_batch_cfg = le_cfg.get("cross_batch")
            result = train_single_loss(
                model=model,
                loss_mode=loss_mode,
                train_loader=loss_bundle["loaders"].train_loader,
                train_eval_loader=loss_bundle["train_eval_loader"],
                eval_loader=loss_bundle["eval_loader"],
                device=device,
                epochs=train_cfg.get("epochs", cfg["training"]["epochs"]),
                lr=train_cfg.get("lr", cfg["training"]["lr"]),
                lr_decay=train_cfg.get(
                    "lr_decay",
                    cfg.get("training", {}).get("lr_decay", 1.0),
                ),
                lr_category_decay=train_cfg.get(
                    "lr_category_decay",
                    cfg.get("training", {}).get("lr_category_decay", 1.0),
                ),
                lr_category=lr_category,
                lr_zero_after_epochs=lr_zero_after_epochs,
                eval_has_labels=eval_has_labels,
                le_base_rates_train=le_base_rates_train,
                le_base_rates_train_eval=le_base_rates_train_eval,
                le_base_rates_eval=le_base_rates_eval,
                focal_alpha=loss_bundle.get("focal_alpha"),
                focal_gamma=loss_bundle.get("focal_gamma"),
                non_blocking=non_blocking,
                plot_eval_hist_epochs=False,
                eval_callback=None,
                eval_every_n_batches=None,
                eval_batch_callback=None,
                print_embedding_table=train_cfg.get("print_embedding_table", False),
                collect_eval_logits=collect_eval_logits,
                debug_gradients=debug_gradients,
                debug_le_inputs=debug_le_inputs,
                le_cross_batch_cfg=le_cross_batch_cfg,
            )
            results[loss_mode].append(result)
    return results
