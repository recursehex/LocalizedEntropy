import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from localized_entropy.training import train_with_epoch_plots


class _PassThroughLogitModel(nn.Module):
    """Return a scaled numeric feature as the logit."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x, x_cat, c):
        del x_cat, c
        return x.view(-1) * self.scale


def test_bce_training_respects_sample_weights():
    """BCE training loss should use per-sample reduction before weighting."""
    x = torch.tensor([[-5.0], [0.2]], dtype=torch.float32)
    x_cat = torch.zeros((2, 1), dtype=torch.long)
    c = torch.zeros(2, dtype=torch.long)
    y = torch.tensor([1.0, 0.0], dtype=torch.float32)
    w = torch.tensor([10.0, 1.0], dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x, x_cat, c, y, w),
        batch_size=2,
        shuffle=False,
    )
    model = _PassThroughLogitModel()

    train_losses, _, _, _ = train_with_epoch_plots(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        lr=0.0,
        loss_mode="bce",
        debug_le_inputs=False,
    )

    logits = x.view(-1)
    bce_per = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        y,
        reduction="none",
    )
    expected_weighted = float((bce_per * w).sum() / w.sum())
    unweighted = float(bce_per.mean())
    assert abs(expected_weighted - unweighted) > 1e-3
    assert train_losses[1] == pytest.approx(expected_weighted, rel=1e-6, abs=1e-8)
