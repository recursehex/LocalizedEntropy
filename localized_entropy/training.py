from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from localized_entropy.losses import CrossBatchHistory, focal_loss_with_logits, localized_entropy
from localized_entropy.plotting import plot_eval_log10p_hist


def _resolve_model_dtype(model: nn.Module, default: torch.dtype = torch.float32) -> torch.dtype:
    """Resolve the dtype used by the model parameters."""
    first_param = next(model.parameters(), None)
    return first_param.dtype if first_param is not None else default


@dataclass
class GradSqStats:
    """Per-condition and per-class gradient mean-square stats."""
    sum_by_condition: np.ndarray
    mean_by_condition: np.ndarray
    count_by_condition: np.ndarray
    class_mse: np.ndarray
    class_counts: np.ndarray
    class_ratio: float


def _format_embedding_table(embedding: nn.Embedding) -> str:
    """Format embedding weights without truncation."""
    weights = embedding.weight.detach().cpu().numpy()
    return np.array2string(weights, separator=", ", threshold=weights.size)


def _print_embedding_table(model: nn.Module, epoch: int) -> None:
    """Print the full embedding table after an epoch when available."""
    emb_layer = getattr(model, "embedding", None)
    if not isinstance(emb_layer, nn.Embedding):
        return
    table = _format_embedding_table(emb_layer)
    print(f"Epoch {epoch:3d} embedding table:\n{table}")


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


def _print_le_batch_inputs(
    epoch: int,
    batch_idx: int,
    x: torch.Tensor,
    x_cat: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    w: Optional[torch.Tensor] = None,
) -> None:
    """Print training batch inputs that feed the LE loss."""
    print(f"[LE][debug] epoch={epoch} batch={batch_idx} input features")
    print(_summarize_tensor("x", x))
    print(_summarize_tensor("x_cat", x_cat))
    print(_summarize_tensor("conditions", c))
    print(_summarize_tensor("targets", y))
    if w is not None:
        print(_summarize_tensor("weights", w))


@torch.no_grad()
def compute_base_rates_from_loader(
    loader: DataLoader,
    num_conditions: int,
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Compute per-condition base rates from a full pass over a loader."""
    counts = torch.zeros(int(num_conditions), dtype=dtype, device=device)
    sum_ones = torch.zeros(int(num_conditions), dtype=dtype, device=device)
    for batch in loader:
        _, _, c, y, w = batch
        c = c.to(device, non_blocking=non_blocking).view(-1).to(torch.long)
        y = y.to(device, non_blocking=non_blocking).view(-1).to(dtype)
        w = w.to(device, non_blocking=non_blocking).view(-1).to(dtype)
        counts += torch.bincount(c, weights=w, minlength=int(num_conditions))
        sum_ones += torch.bincount(c, weights=w * y, minlength=int(num_conditions))
    denom = counts.clamp_min(1)
    rates = sum_ones / denom
    rates = rates.masked_fill(counts == 0, torch.nan)
    return rates


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    condition_weights: Optional[np.ndarray] = None,
    loss_mode: str = "localized_entropy",
    base_rates: Optional[np.ndarray] = None,
    focal_alpha: Optional[float] = 0.25,
    focal_gamma: Optional[float] = 2.0,
    non_blocking: bool = False,
) -> Tuple[float, np.ndarray]:
    """Evaluate a model and return mean loss plus predictions."""
    model.eval()
    model_dtype = _resolve_model_dtype(model)
    total_loss = 0.0
    total_count = 0
    preds_all = []
    verified_cuda_batch = False
    loss_mode = loss_mode.lower().strip()
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    for batch in loader:
        x, x_cat, c, y, _ = batch
        x = x.to(device, non_blocking=non_blocking, dtype=model_dtype)
        x_cat = x_cat.to(device, non_blocking=non_blocking)
        c = c.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking, dtype=model_dtype)
        logits = model(x, x_cat, c)
        # Guard against inadvertently running evaluation on CPU when CUDA is expected.
        if (device.type == "cuda") and (not verified_cuda_batch):
            tensors = (x, x_cat, c, y, logits)
            if any(t.device.type != "cuda" for t in tensors):
                raise RuntimeError("Expected CUDA tensors during evaluation but found CPU tensors.")
            verified_cuda_batch = True
        if loss_mode == "localized_entropy":
            loss = localized_entropy(
                logits=logits,
                targets=y,
                conditions=c,
                base_rates=(
                    torch.as_tensor(base_rates, device=logits.device, dtype=logits.dtype)
                    if base_rates is not None else None
                ),
                condition_weights=(
                    torch.as_tensor(condition_weights, device=logits.device, dtype=logits.dtype)
                    if condition_weights is not None else None),
            )
        elif loss_mode == "bce":
            loss = bce_loss(logits, y).mean()
        elif loss_mode == "focal":
            loss = focal_loss_with_logits(
                logits=logits,
                targets=y,
                alpha=focal_alpha,
                gamma=focal_gamma,
            )
        else:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")
        total_loss += float(loss.item()) * x.size(0)
        total_count += x.size(0)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        preds_all.append(p)
    mean_loss = total_loss / max(1, total_count)
    preds = np.concatenate(preds_all, axis=0)
    return mean_loss, preds


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    non_blocking: bool = False,
) -> np.ndarray:
    """Run inference and return sigmoid probabilities."""
    model.eval()
    model_dtype = _resolve_model_dtype(model)
    preds_all = []
    verified_cuda_batch = False
    for batch in loader:
        x, x_cat, c, _, _ = batch
        x = x.to(device, non_blocking=non_blocking, dtype=model_dtype)
        x_cat = x_cat.to(device, non_blocking=non_blocking)
        c = c.to(device, non_blocking=non_blocking)
        if (device.type == "cuda") and (not verified_cuda_batch):
            tensors = (x, x_cat, c)
            if any(t.device.type != "cuda" for t in tensors):
                raise RuntimeError("Expected CUDA tensors during prediction but found CPU tensors.")
            verified_cuda_batch = True
        logits = model(x, x_cat, c)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        preds_all.append(p)
    if not preds_all:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(preds_all, axis=0)


class StreamingBaseRate:
    def __init__(self, num_conditions: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """Track per-condition base rates with streaming updates."""
        self.num_conditions = int(num_conditions)
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self):
        """Reset counts and sums for a new epoch."""
        self.counts = torch.zeros(self.num_conditions, dtype=torch.long, device=self.device)
        self.sum_ones = torch.zeros(self.num_conditions, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def update(self, y: torch.Tensor, c: torch.Tensor):
        """Update counts and sums from a batch of labels/conditions."""
        c = c.view(-1).to(torch.long)
        y = y.view(-1).to(self.dtype)
        cnt = torch.bincount(c, minlength=self.num_conditions)
        s1 = torch.bincount(c, weights=y, minlength=self.num_conditions)
        self.counts += cnt
        self.sum_ones += s1

    @torch.no_grad()
    def rates(self, eps: float = 1e-12) -> torch.Tensor:
        """Return base rates per condition with clamping."""
        denom = self.counts.clamp_min(1).to(self.sum_ones.dtype)
        return (self.sum_ones / denom).clamp(eps, 1.0 - eps)


def train_with_epoch_plots(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    category_lr: Optional[float] = None,
    lr_zero_after_epochs: Optional[int] = None,
    condition_weights: Optional[np.ndarray] = None,
    loss_mode: str = "localized_entropy",
    base_rates_train: Optional[np.ndarray] = None,
    base_rates_eval: Optional[np.ndarray] = None,
    focal_alpha: Optional[float] = 0.25,
    focal_gamma: Optional[float] = 2.0,
    non_blocking: bool = False,
    plot_eval_hist_epochs: bool = False,
    eval_callback: Optional[Callable[[np.ndarray, int], None]] = None,
    eval_every_n_batches: Optional[int] = None,
    eval_batch_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
    track_eval_batch_losses: bool = False,
    track_grad_sq_sums: bool = False,
    debug_gradients: bool = False,
    debug_le_inputs: bool = False,
    le_cross_batch_cfg: Optional[dict] = None,
    print_embedding_table: bool = False,
) -> Tuple[List[float], List[float], Optional[GradSqStats], List[dict]]:
    """Train a model while optionally collecting plots and diagnostics."""
    base_lr_group_indices = [0]
    if category_lr is not None:
        emb_layer = getattr(model, "embedding", None)
        category_params = list(emb_layer.parameters()) if emb_layer is not None else []
        if category_params:
            category_param_ids = {id(p) for p in category_params}
            base_params = [p for p in model.parameters() if id(p) not in category_param_ids]
            param_groups = []
            base_lr_group_indices = []
            if base_params:
                param_groups.append({"params": base_params, "lr": lr})
                base_lr_group_indices.append(0)
            param_groups.append({"params": category_params, "lr": float(category_lr)})
            opt = torch.optim.Adam(param_groups)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: List[float] = []
    val_losses: List[float] = []
    model_dtype = _resolve_model_dtype(model)
    loss_mode = loss_mode.lower().strip()
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    base_lr_zero_after = None
    if lr_zero_after_epochs is not None:
        base_lr_zero_after = int(lr_zero_after_epochs)
        if base_lr_zero_after < 0:
            raise ValueError("lr_zero_after_epochs must be >= 0.")
    if loss_mode == "localized_entropy":
        loss_label = "LE"
    elif loss_mode == "bce":
        loss_label = "BCE"
    elif loss_mode == "focal":
        loss_label = "Focal"
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    use_le = loss_mode == "localized_entropy"
    cross_batch_history = None
    if use_le and isinstance(le_cross_batch_cfg, dict):
        enabled = bool(le_cross_batch_cfg.get("enabled", False))
        amp_factor = le_cross_batch_cfg.get(
            "amplification_rate",
            le_cross_batch_cfg.get("amplification_factor"),
        )
        if enabled and amp_factor is not None:
            amp_value = float(amp_factor)
            if amp_value > 0:
                cross_batch_history = CrossBatchHistory(amp_value)
    base_rates_train_t = None
    base_rates_eval_t = None
    if use_le:
        first_param = next(model.parameters(), None)
        p_dtype = first_param.dtype if first_param is not None else torch.float32
        if base_rates_train is None:
            num_conds_model = getattr(model, "embedding").num_embeddings if hasattr(model, "embedding") else 1
            base_rates_train_t = compute_base_rates_from_loader(
                train_loader,
                num_conditions=num_conds_model,
                device=device,
                dtype=p_dtype,
                non_blocking=non_blocking,
            )
        else:
            base_rates_train_t = torch.as_tensor(base_rates_train, device=device, dtype=p_dtype)
        if base_rates_eval is None:
            base_rates_eval_t = base_rates_train_t
        else:
            base_rates_eval_t = torch.as_tensor(base_rates_eval, device=device, dtype=p_dtype)
    grad_sq_sums = None
    grad_sq_counts = None
    grad_sq_class_sums = None
    grad_sq_class_counts = None
    stats_dtype = None
    if track_grad_sq_sums:
        emb_layer = getattr(model, "embedding", None)
        if isinstance(emb_layer, nn.Embedding):
            num_conds = int(emb_layer.num_embeddings)
        else:
            raise ValueError("Gradient tracking requires a model with an embedding layer.")
        # Accumulate per-condition gradient squared sums and counts for MSE diagnostics.
        stats_dtype = torch.float32 if device.type == "mps" else torch.float64
        grad_sq_sums = torch.zeros(num_conds, dtype=stats_dtype, device=device)
        grad_sq_counts = torch.zeros(num_conds, dtype=stats_dtype, device=device)
        grad_sq_class_sums = torch.zeros(2, dtype=stats_dtype, device=device)
        grad_sq_class_counts = torch.zeros(2, dtype=stats_dtype, device=device)
    if device.type == "cuda":
        first_param = next(model.parameters(), None)
        if (first_param is not None) and (first_param.device.type != "cuda"):
            raise RuntimeError("Model parameters must be on CUDA when device is CUDA.")
        torch.cuda.reset_peak_memory_stats(device)

    init_train_loss, _ = evaluate(
        model,
        train_loader,
        device,
        condition_weights=condition_weights,
        loss_mode=loss_mode,
        base_rates=base_rates_train_t if use_le else None,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        non_blocking=non_blocking,
    )
    init_eval_loss, _ = evaluate(
        model,
        val_loader,
        device,
        condition_weights=condition_weights,
        loss_mode=loss_mode,
        base_rates=base_rates_eval_t if use_le else None,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        non_blocking=non_blocking,
    )
    train_losses.append(float(init_train_loss))
    val_losses.append(float(init_eval_loss))

    preds = None
    eval_every = int(eval_every_n_batches) if eval_every_n_batches is not None else 0
    if eval_every < 1:
        eval_every = 0
    eval_batch_losses: List[dict] = []
    eval_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        if base_lr_zero_after is not None and epoch > base_lr_zero_after:
            for group_idx in base_lr_group_indices:
                if opt.param_groups[group_idx]["lr"] != 0.0:
                    opt.param_groups[group_idx]["lr"] = 0.0
        br_tracker = None
        if use_le and base_rates_train_t is None:
            # Fallback to streaming base rates if precomputed rates are unavailable.
            num_conds_model = getattr(model, "embedding").num_embeddings if hasattr(model, "embedding") else 1
            first_param = next(model.parameters(), None)
            p_dtype = first_param.dtype if first_param is not None else torch.float32
            br_tracker = StreamingBaseRate(num_conds_model, device=device, dtype=p_dtype)
        running = 0.0
        count = 0
        verified_cuda_batch = False
        epoch_start = time.time()
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        for batch_idx, batch in enumerate(train_loader, start=1):
            x, x_cat, c, y, w = batch
            x = x.to(device, non_blocking=non_blocking, dtype=model_dtype)
            x_cat = x_cat.to(device, non_blocking=non_blocking)
            c = c.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking, dtype=model_dtype)
            w = w.to(device, non_blocking=non_blocking, dtype=model_dtype)
            if use_le and debug_le_inputs:
                _print_le_batch_inputs(epoch, batch_idx, x, x_cat, c, y, w)
            if (device.type == "cuda") and (not verified_cuda_batch):
                tensors = (x, x_cat, c, y)
                if any(t.device.type != "cuda" for t in tensors):
                    raise RuntimeError("Detected CPU tensors in the training loop while using CUDA.")
                verified_cuda_batch = True
            opt.zero_grad(set_to_none=True)
            logits = model(x, x_cat, c)
            if grad_sq_sums is not None:
                logits.retain_grad()
            if use_le:
                if br_tracker is not None:
                    br_tracker.update(y, c)
                loss = localized_entropy(
                    logits=logits,
                    targets=y,
                    conditions=c,
                    base_rates=base_rates_train_t if base_rates_train_t is not None else br_tracker.rates(),
                    condition_weights=(
                        torch.as_tensor(condition_weights, device=logits.device, dtype=logits.dtype)
                        if condition_weights is not None else None),
                    sample_weights=w,
                    debug=debug_le_inputs,
                    cross_batch_history=cross_batch_history,
                )
            elif loss_mode == "bce":
                bce_per = bce_loss(logits, y)
                w_flat = w.view(-1).to(logits.dtype)
                loss = (bce_per.view(-1) * w_flat).sum() / w_flat.sum().clamp_min(1.0)
            elif loss_mode == "focal":
                loss = focal_loss_with_logits(
                    logits=logits,
                    targets=y,
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                    sample_weights=w,
                )
            else:
                raise ValueError(f"Unsupported loss_mode: {loss_mode}")
            loss.backward()
            if grad_sq_sums is not None:
                grad = logits.grad
                if grad is None:
                    raise RuntimeError("Expected logits gradients for grad tracking.")
                # Aggregate squared gradients and counts by condition/class for diagnostics.
                grad_sq = grad.detach().view(-1).to(stats_dtype or torch.float64)
                c_flat = c.view(-1).to(torch.long)
                if grad_sq.numel() != c_flat.numel():
                    raise RuntimeError("Logits gradient size does not match conditions.")
                grad_sq = grad_sq * grad_sq
                grad_sq_sums += torch.bincount(
                    c_flat,
                    weights=grad_sq,
                    minlength=grad_sq_sums.numel(),
                )
                if grad_sq_counts is None:
                    raise RuntimeError("Expected grad_sq_counts when tracking gradients.")
                grad_sq_counts += torch.bincount(
                    c_flat,
                    minlength=grad_sq_counts.numel(),
                ).to(stats_dtype or torch.float64)
                if grad_sq_class_sums is None or grad_sq_class_counts is None:
                    raise RuntimeError("Expected grad_sq_class stats when tracking gradients.")
                y_flat = y.view(-1)
                y_int = (y_flat > 0.5).to(torch.long)
                grad_sq_class_sums += torch.bincount(
                    y_int,
                    weights=grad_sq,
                    minlength=2,
                )
                grad_sq_class_counts += torch.bincount(
                    y_int,
                    minlength=2,
                ).to(stats_dtype or torch.float64)
            if debug_gradients:
                print(f"[DEBUG] Gradients for epoch {epoch}, batch {batch_idx}")
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"[DEBUG] {name}: grad=None")
                    else:
                        print(f"[DEBUG] {name}:\n{param.grad.detach()}")
            opt.step()
            batch_weight = float(w.sum().item())
            running += float(loss.item()) * batch_weight
            count += batch_weight
            if (
                eval_every
                and (eval_batch_callback is not None or track_eval_batch_losses)
                and (batch_idx % eval_every == 0)
                and (total_batches is None or batch_idx < total_batches)
            ):
                # Mid-epoch evals support batch-level plots and loss curve diagnostics.
                mid_loss, mid_preds = evaluate(
                    model, val_loader, device,
                    condition_weights=condition_weights,
                    loss_mode=loss_mode,
                    base_rates=base_rates_eval_t if use_le else None,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    non_blocking=non_blocking,
                )
                if eval_batch_callback is not None:
                    eval_batch_callback(mid_preds, epoch, batch_idx)
                if track_eval_batch_losses:
                    eval_step += 1
                    step = (
                        (epoch - 1) * total_batches + batch_idx
                        if total_batches is not None
                        else eval_step
                    )
                    epoch_progress = None
                    if total_batches:
                        epoch_progress = (epoch - 1) + (batch_idx / total_batches)
                    eval_batch_losses.append(
                        {
                            "epoch": int(epoch),
                            "batch_idx": int(batch_idx),
                            "step": int(step),
                            "epoch_progress": epoch_progress,
                            "train_loss": float(running / max(1, count)),
                            "loss": float(mid_loss),
                        }
                    )
                model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_loss = running / max(1, count)
        val_loss, preds = evaluate(
            model, val_loader, device,
            condition_weights=condition_weights,
            loss_mode=loss_mode,
            base_rates=base_rates_eval_t if use_le else None,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            non_blocking=non_blocking,
        )
        if plot_eval_hist_epochs and preds is not None:
            plot_eval_log10p_hist(preds.astype(np.float32), epoch)
        if eval_callback is not None and preds is not None:
            eval_callback(preds, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_time = time.time() - epoch_start
        log_msg = (
            f"Epoch {epoch:3d}/{epochs} | Train {loss_label}: {train_loss:.6f} "
            f"| Eval {loss_label}: {val_loss:.6f} | wall {epoch_time:.2f}s"
        )
        if device.type == "cuda":
            mem_alloc = torch.cuda.memory_allocated(device) / 1e6
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
            log_msg += f" | cuda_mem={mem_alloc:.1f}MB (peak {peak_mem:.1f}MB)"
        print(log_msg)
        if print_embedding_table:
            _print_embedding_table(model, epoch)

    print(f"Final Train {loss_label}: {train_losses[-1]:.10f}")
    print(f"Final Eval  {loss_label}: {val_losses[-1]:.10f}")
    grad_sq_stats = None
    if grad_sq_sums is not None:
        if grad_sq_counts is None or grad_sq_class_sums is None or grad_sq_class_counts is None:
            raise RuntimeError("Expected grad_sq stats to be populated when tracking gradients.")
        grad_sq_sums_np = grad_sq_sums.detach().cpu().numpy()
        grad_sq_counts_np = grad_sq_counts.detach().cpu().numpy()
        grad_sq_means_np = np.divide(
            grad_sq_sums_np,
            grad_sq_counts_np,
            out=np.zeros_like(grad_sq_sums_np),
            where=grad_sq_counts_np > 0,
        )
        grad_sq_class_sums_np = grad_sq_class_sums.detach().cpu().numpy()
        grad_sq_class_counts_np = grad_sq_class_counts.detach().cpu().numpy()
        grad_sq_class_mse_np = np.divide(
            grad_sq_class_sums_np,
            grad_sq_class_counts_np,
            out=np.full_like(grad_sq_class_sums_np, np.nan),
            where=grad_sq_class_counts_np > 0,
        )
        class_ratio = float("nan")
        if (
            np.isfinite(grad_sq_class_mse_np).all()
            and grad_sq_class_mse_np[1] != 0.0
        ):
            class_ratio = float(grad_sq_class_mse_np[0] / grad_sq_class_mse_np[1])
        grad_sq_stats = GradSqStats(
            sum_by_condition=grad_sq_sums_np,
            mean_by_condition=grad_sq_means_np,
            count_by_condition=grad_sq_counts_np,
            class_mse=grad_sq_class_mse_np,
            class_counts=grad_sq_class_counts_np,
            class_ratio=class_ratio,
        )
    return train_losses, val_losses, grad_sq_stats, eval_batch_losses
