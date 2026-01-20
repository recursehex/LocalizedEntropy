from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from localized_entropy.analysis import collect_logits
from localized_entropy.config import get_data_source, loss_label
from localized_entropy.models import ConditionProbNet
from localized_entropy.training import evaluate, predict_probs, train_with_epoch_plots
from localized_entropy.utils import set_seed


@dataclass
class TrainRunResult:
    loss_mode: str
    loss_label: str
    model: torch.nn.Module
    train_losses: List[float]
    eval_losses: List[float]
    eval_loss: float
    eval_preds: np.ndarray
    eval_logits: Optional[torch.Tensor]
    eval_targets: Optional[torch.Tensor]
    eval_conds: Optional[torch.Tensor]
    grad_sq_sum_per_condition: Optional[np.ndarray]


def build_model(cfg: dict, splits, device: torch.device) -> ConditionProbNet:
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
    model = ConditionProbNet(
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
    return model.to(device)


def resolve_eval_bundle(cfg, splits, loaders):
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
    non_blocking: bool = False,
) -> Tuple[float, np.ndarray]:
    if eval_has_labels:
        return evaluate(
            model,
            eval_loader,
            device,
            loss_mode=loss_mode,
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
    eval_has_labels: bool,
    non_blocking: bool = False,
    plot_eval_hist_epochs: bool = False,
    eval_callback: Optional[Callable[[np.ndarray, int], None]] = None,
    eval_every_n_batches: Optional[int] = None,
    eval_batch_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
    collect_eval_logits: bool = False,
    collect_grad_sq_sums: bool = False,
) -> TrainRunResult:
    train_losses, eval_losses, grad_sq_sums = train_with_epoch_plots(
        model=model,
        train_loader=train_loader,
        val_loader=train_eval_loader,
        device=device,
        epochs=int(epochs),
        lr=float(lr),
        loss_mode=loss_mode,
        non_blocking=non_blocking,
        plot_eval_hist_epochs=plot_eval_hist_epochs,
        eval_callback=eval_callback,
        eval_every_n_batches=eval_every_n_batches,
        eval_batch_callback=eval_batch_callback,
        track_grad_sq_sums=collect_grad_sq_sums,
    )
    eval_loss, eval_preds = evaluate_or_predict(
        model,
        eval_loader,
        device,
        loss_mode=loss_mode,
        eval_has_labels=eval_has_labels,
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
        eval_loss=float(eval_loss),
        eval_preds=eval_preds,
        eval_logits=eval_logits,
        eval_targets=eval_targets,
        eval_conds=eval_conds,
        grad_sq_sum_per_condition=grad_sq_sums,
    )


def build_seed_sequence(base_seed: int, num_runs: int, seed_stride: int = 1) -> List[int]:
    if num_runs < 1:
        return []
    stride = int(seed_stride)
    return [int(base_seed) + stride * idx for idx in range(int(num_runs))]


def run_repeated_loss_experiments(
    *,
    cfg: dict,
    loss_modes: List[str],
    splits,
    loaders,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    use_cuda: bool,
    eval_has_labels: bool,
    seeds: Iterable[int],
    non_blocking: bool = False,
    collect_eval_logits: bool = False,
) -> Dict[str, List[TrainRunResult]]:
    train_cfg = cfg["training"]
    results: Dict[str, List[TrainRunResult]] = {loss_mode: [] for loss_mode in loss_modes}
    for seed in seeds:
        for loss_mode in loss_modes:
            set_seed(int(seed), use_cuda)
            model = build_model(cfg, splits, device)
            result = train_single_loss(
                model=model,
                loss_mode=loss_mode,
                train_loader=loaders.train_loader,
                train_eval_loader=train_eval_loader,
                eval_loader=eval_loader,
                device=device,
                epochs=train_cfg["epochs"],
                lr=train_cfg["lr"],
                eval_has_labels=eval_has_labels,
                non_blocking=non_blocking,
                plot_eval_hist_epochs=False,
                eval_callback=None,
                eval_every_n_batches=None,
                eval_batch_callback=None,
                collect_eval_logits=collect_eval_logits,
            )
            results[loss_mode].append(result)
    return results
