from typing import Callable, List, Optional, Tuple

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from localized_entropy.losses import localized_entropy
from localized_entropy.plotting import plot_eval_log10p_hist


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    condition_weights: Optional[np.ndarray] = None,
    nw_threshold: Optional[float] = None,
    nw_multiplier: float = 1.0,
    loss_mode: str = "localized_entropy",
    base_rates: Optional[np.ndarray] = None,
    non_blocking: bool = False,
) -> Tuple[float, np.ndarray]:
    """Evaluate a model and return mean loss plus predictions."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds_all = []
    verified_cuda_batch = False
    loss_mode = loss_mode.lower().strip()
    bce_loss = nn.BCEWithLogitsLoss()
    for x, x_cat, c, y, nw in loader:
        x = x.to(device, non_blocking=non_blocking)
        x_cat = x_cat.to(device, non_blocking=non_blocking)
        c = c.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        nw = nw.to(device, non_blocking=non_blocking)
        logits = model(x, x_cat, c)
        # Guard against inadvertently running evaluation on CPU when CUDA is expected.
        if (device.type == "cuda") and (not verified_cuda_batch):
            tensors = (x, x_cat, c, y, nw, logits)
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
                net_worth=nw,
                condition_weights=(
                    torch.as_tensor(condition_weights, device=logits.device, dtype=logits.dtype)
                    if condition_weights is not None else None),
                nw_threshold=nw_threshold,
                nw_multiplier=nw_multiplier,
            )
        elif loss_mode == "bce":
            loss = bce_loss(logits, y)
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
    preds_all = []
    verified_cuda_batch = False
    for x, x_cat, c, y, nw in loader:
        x = x.to(device, non_blocking=non_blocking)
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
    condition_weights: Optional[np.ndarray] = None,
    nw_threshold: Optional[float] = None,
    nw_multiplier: float = 1.0,
    loss_mode: str = "localized_entropy",
    base_rates_train: Optional[np.ndarray] = None,
    base_rates_eval: Optional[np.ndarray] = None,
    non_blocking: bool = False,
    plot_eval_hist_epochs: bool = False,
    eval_callback: Optional[Callable[[np.ndarray, int], None]] = None,
    eval_every_n_batches: Optional[int] = None,
    eval_batch_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
    track_eval_batch_losses: bool = False,
    track_grad_sq_sums: bool = False,
    debug_gradients: bool = False,
) -> Tuple[List[float], List[float], Optional[np.ndarray], List[dict]]:
    """Train a model while optionally collecting plots and diagnostics."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: List[float] = []
    val_losses: List[float] = []
    loss_mode = loss_mode.lower().strip()
    bce_loss = nn.BCEWithLogitsLoss()
    loss_label = "LE" if loss_mode == "localized_entropy" else "BCE"
    use_le = loss_mode == "localized_entropy"
    base_rates_train_t = None
    if use_le and base_rates_train is not None:
        first_param = next(model.parameters(), None)
        p_dtype = first_param.dtype if first_param is not None else torch.float32
        base_rates_train_t = torch.as_tensor(base_rates_train, device=device, dtype=p_dtype)
    grad_sq_sums = None
    if track_grad_sq_sums:
        emb_layer = getattr(model, "embedding", None)
        if isinstance(emb_layer, nn.Embedding):
            num_conds = int(emb_layer.num_embeddings)
        else:
            raise ValueError("Gradient tracking requires a model with an embedding layer.")
        # Accumulate per-condition gradient squared sums on logits for diagnostics.
        grad_sq_sums = torch.zeros(num_conds, dtype=torch.float64, device=device)
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
        nw_threshold=nw_threshold,
        nw_multiplier=nw_multiplier,
        loss_mode=loss_mode,
        base_rates=base_rates_train if use_le else None,
        non_blocking=non_blocking,
    )
    init_eval_loss, _ = evaluate(
        model,
        val_loader,
        device,
        condition_weights=condition_weights,
        nw_threshold=nw_threshold,
        nw_multiplier=nw_multiplier,
        loss_mode=loss_mode,
        base_rates=base_rates_eval if use_le else None,
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
        br_tracker = None
        if use_le and base_rates_train_t is None:
            # Track per-condition base rates within the epoch for LE denominator terms.
            num_conds_model = getattr(model, "embedding").num_embeddings if hasattr(model, "embedding") else 1
            first_param = next(model.parameters(), None)
            p_dtype = first_param.dtype if first_param is not None else torch.float32
            br_tracker = StreamingBaseRate(num_conds_model, device=device, dtype=p_dtype)
        running = 0.0
        count = 0
        verified_cuda_batch = False
        epoch_start = time.time()
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        for batch_idx, (x, x_cat, c, y, nw) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=non_blocking)
            x_cat = x_cat.to(device, non_blocking=non_blocking)
            c = c.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            nw = nw.to(device, non_blocking=non_blocking)
            if (device.type == "cuda") and (not verified_cuda_batch):
                tensors = (x, x_cat, c, y, nw)
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
                    net_worth=nw,
                    condition_weights=(
                        torch.as_tensor(condition_weights, device=logits.device, dtype=logits.dtype)
                        if condition_weights is not None else None),
                    nw_threshold=nw_threshold,
                    nw_multiplier=nw_multiplier,
                )
            elif loss_mode == "bce":
                loss = bce_loss(logits, y)
            else:
                raise ValueError(f"Unsupported loss_mode: {loss_mode}")
            loss.backward()
            if grad_sq_sums is not None:
                grad = logits.grad
                if grad is None:
                    raise RuntimeError("Expected logits gradients for grad tracking.")
                # Aggregate squared gradients by condition for BCE vs LE diagnostics.
                grad_sq = grad.detach().view(-1).to(torch.float64)
                c_flat = c.view(-1).to(torch.long)
                if grad_sq.numel() != c_flat.numel():
                    raise RuntimeError("Logits gradient size does not match conditions.")
                grad_sq = grad_sq * grad_sq
                grad_sq_sums += torch.bincount(
                    c_flat,
                    weights=grad_sq,
                    minlength=grad_sq_sums.numel(),
                )
            if debug_gradients:
                print(f"[DEBUG] Gradients for epoch {epoch}, batch {batch_idx}")
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"[DEBUG] {name}: grad=None")
                    else:
                        print(f"[DEBUG] {name}:\n{param.grad.detach()}")
            opt.step()
            running += float(loss.item()) * x.size(0)
            count += x.size(0)
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
                    nw_threshold=nw_threshold,
                    nw_multiplier=nw_multiplier,
                    loss_mode=loss_mode,
                    base_rates=base_rates_eval if use_le else None,
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
            nw_threshold=nw_threshold,
            nw_multiplier=nw_multiplier,
            loss_mode=loss_mode,
            base_rates=base_rates_eval if use_le else None,
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

    print(f"Final Train {loss_label}: {train_losses[-1]:.10f}")
    print(f"Final Eval  {loss_label}: {val_losses[-1]:.10f}")
    grad_sq_sums_np = grad_sq_sums.detach().cpu().numpy() if grad_sq_sums is not None else None
    return train_losses, val_losses, grad_sq_sums_np, eval_batch_losses
