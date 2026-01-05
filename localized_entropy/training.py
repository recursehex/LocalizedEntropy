from typing import List, Optional, Tuple

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
    non_blocking: bool = False,
) -> Tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds_all = []
    verified_cuda_batch = False
    loss_mode = loss_mode.lower().strip()
    bce_loss = nn.BCEWithLogitsLoss()
    for x, c, y, nw in loader:
        x = x.to(device, non_blocking=non_blocking)
        c = c.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        nw = nw.to(device, non_blocking=non_blocking)
        logits = model(x, c)
        if (device.type == "cuda") and (not verified_cuda_batch):
            tensors = (x, c, y, nw, logits)
            if any(t.device.type != "cuda" for t in tensors):
                raise RuntimeError("Expected CUDA tensors during evaluation but found CPU tensors.")
            verified_cuda_batch = True
        if loss_mode == "localized_entropy":
            loss = localized_entropy(
                logits=logits,
                targets=y,
                conditions=c,
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
    model.eval()
    preds_all = []
    verified_cuda_batch = False
    for x, c, y, nw in loader:
        x = x.to(device, non_blocking=non_blocking)
        c = c.to(device, non_blocking=non_blocking)
        if (device.type == "cuda") and (not verified_cuda_batch):
            tensors = (x, c)
            if any(t.device.type != "cuda" for t in tensors):
                raise RuntimeError("Expected CUDA tensors during prediction but found CPU tensors.")
            verified_cuda_batch = True
        logits = model(x, c)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        preds_all.append(p)
    if not preds_all:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(preds_all, axis=0)


class StreamingBaseRate:
    def __init__(self, num_conditions: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.num_conditions = int(num_conditions)
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.counts = torch.zeros(self.num_conditions, dtype=torch.long, device=self.device)
        self.sum_ones = torch.zeros(self.num_conditions, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def update(self, y: torch.Tensor, c: torch.Tensor):
        c = c.view(-1).to(torch.long)
        y = y.view(-1).to(self.dtype)
        cnt = torch.bincount(c, minlength=self.num_conditions)
        s1 = torch.bincount(c, weights=y, minlength=self.num_conditions)
        self.counts += cnt
        self.sum_ones += s1

    @torch.no_grad()
    def rates(self, eps: float = 1e-12) -> torch.Tensor:
        denom = self.counts.clamp_min(1).to(self.sum_ones.dtype)
        return (self.sum_ones / denom).clamp(eps, 1.0 - eps)


def train_with_epoch_plots(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    l2_weight_decay: float = 0.0,
    condition_weights: Optional[np.ndarray] = None,
    nw_threshold: Optional[float] = None,
    nw_multiplier: float = 1.0,
    loss_mode: str = "localized_entropy",
    non_blocking: bool = False,
    plot_eval_hist_epochs: bool = False,
) -> Tuple[List[float], List[float]]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight_decay)
    train_losses: List[float] = []
    val_losses: List[float] = []
    loss_mode = loss_mode.lower().strip()
    bce_loss = nn.BCEWithLogitsLoss()
    loss_label = "LE" if loss_mode == "localized_entropy" else "BCE"
    use_le = loss_mode == "localized_entropy"
    if device.type == "cuda":
        first_param = next(model.parameters(), None)
        if (first_param is not None) and (first_param.device.type != "cuda"):
            raise RuntimeError("Model parameters must be on CUDA when device is CUDA.")
        torch.cuda.reset_peak_memory_stats(device)

    preds = None
    for epoch in range(1, epochs + 1):
        model.train()
        br_tracker = None
        if use_le:
            num_conds_model = getattr(model, "embedding").num_embeddings if hasattr(model, "embedding") else 1
            first_param = next(model.parameters(), None)
            p_dtype = first_param.dtype if first_param is not None else torch.float32
            br_tracker = StreamingBaseRate(num_conds_model, device=device, dtype=p_dtype)
        running = 0.0
        count = 0
        verified_cuda_batch = False
        epoch_start = time.time()
        for x, c, y, nw in train_loader:
            x = x.to(device, non_blocking=non_blocking)
            c = c.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            nw = nw.to(device, non_blocking=non_blocking)
            if (device.type == "cuda") and (not verified_cuda_batch):
                tensors = (x, c, y, nw)
                if any(t.device.type != "cuda" for t in tensors):
                    raise RuntimeError("Detected CPU tensors in the training loop while using CUDA.")
                verified_cuda_batch = True
            opt.zero_grad(set_to_none=True)
            logits = model(x, c)
            if use_le:
                br_tracker.update(y, c)
                loss = localized_entropy(
                    logits=logits,
                    targets=y,
                    conditions=c,
                    base_rates=br_tracker.rates(),
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
            opt.step()
            running += float(loss.item()) * x.size(0)
            count += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_loss = running / max(1, count)
        val_loss, preds = evaluate(
            model, val_loader, device,
            condition_weights=condition_weights,
            nw_threshold=nw_threshold,
            nw_multiplier=nw_multiplier,
            loss_mode=loss_mode,
            non_blocking=non_blocking,
        )
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

    if plot_eval_hist_epochs and preds is not None:
        plot_eval_log10p_hist(preds.astype(np.float32), epoch)
    print(f"Final Train {loss_label}: {train_losses[-1]:.10f}")
    print(f"Final Eval  {loss_label}: {val_losses[-1]:.10f}")
    return train_losses, val_losses
