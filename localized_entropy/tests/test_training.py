import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from localized_entropy.training import evaluate, train_with_epoch_plots


class _PassThroughLogitModel(nn.Module):
    """Return a scaled numeric feature as the logit."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x, x_cat, c):
        del x_cat, c
        return x.view(-1) * self.scale


class _EmbeddingAndScaleLogitModel(nn.Module):
    """Model with embedding + base parameter to test split LR behavior."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(2, 1)
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x, x_cat, c):
        del x_cat
        emb = self.embedding(c.view(-1)).view(-1)
        return x.view(-1) * self.scale + emb


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


def test_bce_evaluate_respects_sample_weights():
    """BCE evaluation loss should apply non-uniform per-sample weights."""
    x = torch.tensor([[-5.0], [0.2], [1.5], [-0.7]], dtype=torch.float32)
    x_cat = torch.zeros((4, 1), dtype=torch.long)
    c = torch.zeros(4, dtype=torch.long)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    w = torch.tensor([10.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x, x_cat, c, y, w),
        batch_size=2,
        shuffle=False,
    )
    model = _PassThroughLogitModel()

    eval_loss, _ = evaluate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        loss_mode="bce",
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
    assert eval_loss == pytest.approx(expected_weighted, rel=1e-6, abs=1e-8)


def test_bce_evaluate_unit_weights_matches_unweighted():
    """BCE evaluation with unit weights should match the original unweighted mean."""
    x = torch.tensor([[-5.0], [0.2], [1.5], [-0.7]], dtype=torch.float32)
    x_cat = torch.zeros((4, 1), dtype=torch.long)
    c = torch.zeros(4, dtype=torch.long)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    w = torch.ones(4, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x, x_cat, c, y, w),
        batch_size=2,
        shuffle=False,
    )
    model = _PassThroughLogitModel()

    eval_loss, _ = evaluate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        loss_mode="bce",
    )

    logits = x.view(-1)
    expected_unweighted = float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="mean",
        )
    )
    assert eval_loss == pytest.approx(expected_unweighted, rel=1e-6, abs=1e-8)


def test_evaluate_empty_loader_returns_nan_and_empty_predictions():
    """Evaluation on an empty loader should not raise and should return empty preds."""
    x = torch.empty((0, 1), dtype=torch.float32)
    x_cat = torch.empty((0, 1), dtype=torch.long)
    c = torch.empty((0,), dtype=torch.long)
    y = torch.empty((0,), dtype=torch.float32)
    w = torch.empty((0,), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x, x_cat, c, y, w),
        batch_size=2,
        shuffle=False,
    )
    model = _PassThroughLogitModel()

    eval_loss, preds = evaluate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        loss_mode="bce",
    )

    assert np.isnan(eval_loss)
    assert preds.dtype == np.float32
    assert preds.shape == (0,)


def test_train_with_epoch_plots_applies_lr_decay_each_batch(monkeypatch):
    """Learning rate should decay multiplicatively after each batch step."""
    x = torch.zeros((3, 1), dtype=torch.float32)
    x_cat = torch.zeros((3, 1), dtype=torch.long)
    c = torch.zeros(3, dtype=torch.long)
    y = torch.zeros(3, dtype=torch.float32)
    w = torch.ones(3, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x, x_cat, c, y, w),
        batch_size=1,
        shuffle=False,
    )
    model = _PassThroughLogitModel()

    created_opts = []

    class _RecordingAdam:
        def __init__(self, params, lr=1e-3):
            self.param_groups = []
            if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
                for group in params:
                    group_lr = float(group.get("lr", lr))
                    self.param_groups.append(
                        {
                            "params": list(group["params"]),
                            "lr": group_lr,
                        }
                    )
            else:
                self.param_groups.append(
                    {
                        "params": list(params),
                        "lr": float(lr),
                    }
                )
            self.step_lrs = []
            created_opts.append(self)

        def zero_grad(self, set_to_none=True):
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zero_()

        def step(self):
            self.step_lrs.append([float(group["lr"]) for group in self.param_groups])

    monkeypatch.setattr(torch.optim, "Adam", _RecordingAdam)

    train_with_epoch_plots(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        lr=0.1,
        lr_decay=0.5,
        loss_mode="bce",
        debug_le_inputs=False,
    )

    assert len(created_opts) == 1
    opt = created_opts[0]
    observed_lrs = [row[0] for row in opt.step_lrs]
    assert observed_lrs == pytest.approx([0.1, 0.05, 0.025], rel=1e-9, abs=0.0)
    assert float(opt.param_groups[0]["lr"]) == pytest.approx(0.0125, rel=1e-9, abs=0.0)


def test_train_with_epoch_plots_lr_decay_does_not_change_lr_category(monkeypatch):
    """Per-batch lr_decay should not alter the category embedding lr."""
    x = torch.zeros((3, 1), dtype=torch.float32)
    x_cat = torch.zeros((3, 1), dtype=torch.long)
    c = torch.tensor([0, 1, 0], dtype=torch.long)
    y = torch.zeros(3, dtype=torch.float32)
    w = torch.ones(3, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x, x_cat, c, y, w),
        batch_size=1,
        shuffle=False,
    )
    model = _EmbeddingAndScaleLogitModel()

    created_opts = []

    class _RecordingAdam:
        def __init__(self, params, lr=1e-3):
            self.param_groups = []
            if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
                for group in params:
                    group_lr = float(group.get("lr", lr))
                    self.param_groups.append(
                        {
                            "params": list(group["params"]),
                            "lr": group_lr,
                        }
                    )
            else:
                self.param_groups.append(
                    {
                        "params": list(params),
                        "lr": float(lr),
                    }
                )
            self.step_lrs = []
            created_opts.append(self)

        def zero_grad(self, set_to_none=True):
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zero_()

        def step(self):
            self.step_lrs.append([float(group["lr"]) for group in self.param_groups])

    monkeypatch.setattr(torch.optim, "Adam", _RecordingAdam)

    train_with_epoch_plots(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        lr=0.1,
        lr_decay=0.5,
        category_lr=0.01,
        loss_mode="bce",
        debug_le_inputs=False,
    )

    assert len(created_opts) == 1
    opt = created_opts[0]
    base_lrs = [row[0] for row in opt.step_lrs]
    category_lrs = [row[1] for row in opt.step_lrs]
    assert base_lrs == pytest.approx([0.1, 0.05, 0.025], rel=1e-9, abs=0.0)
    assert category_lrs == pytest.approx([0.01, 0.01, 0.01], rel=1e-9, abs=0.0)
    assert float(opt.param_groups[0]["lr"]) == pytest.approx(0.0125, rel=1e-9, abs=0.0)
    assert float(opt.param_groups[1]["lr"]) == pytest.approx(0.01, rel=1e-9, abs=0.0)


def test_train_with_epoch_plots_lr_category_decay_only_changes_category(monkeypatch):
    """Per-batch lr_category_decay should not alter the base lr."""
    x = torch.zeros((3, 1), dtype=torch.float32)
    x_cat = torch.zeros((3, 1), dtype=torch.long)
    c = torch.tensor([0, 1, 0], dtype=torch.long)
    y = torch.zeros(3, dtype=torch.float32)
    w = torch.ones(3, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x, x_cat, c, y, w),
        batch_size=1,
        shuffle=False,
    )
    model = _EmbeddingAndScaleLogitModel()

    created_opts = []

    class _RecordingAdam:
        def __init__(self, params, lr=1e-3):
            self.param_groups = []
            if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
                for group in params:
                    group_lr = float(group.get("lr", lr))
                    self.param_groups.append(
                        {
                            "params": list(group["params"]),
                            "lr": group_lr,
                        }
                    )
            else:
                self.param_groups.append(
                    {
                        "params": list(params),
                        "lr": float(lr),
                    }
                )
            self.step_lrs = []
            created_opts.append(self)

        def zero_grad(self, set_to_none=True):
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zero_()

        def step(self):
            self.step_lrs.append([float(group["lr"]) for group in self.param_groups])

    monkeypatch.setattr(torch.optim, "Adam", _RecordingAdam)

    train_with_epoch_plots(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        lr=0.1,
        lr_decay=1.0,
        lr_category_decay=0.5,
        category_lr=0.01,
        loss_mode="bce",
        debug_le_inputs=False,
    )

    assert len(created_opts) == 1
    opt = created_opts[0]
    base_lrs = [row[0] for row in opt.step_lrs]
    category_lrs = [row[1] for row in opt.step_lrs]
    assert base_lrs == pytest.approx([0.1, 0.1, 0.1], rel=1e-9, abs=0.0)
    assert category_lrs == pytest.approx([0.01, 0.005, 0.0025], rel=1e-9, abs=0.0)
    assert float(opt.param_groups[0]["lr"]) == pytest.approx(0.1, rel=1e-9, abs=0.0)
    assert float(opt.param_groups[1]["lr"]) == pytest.approx(0.00125, rel=1e-9, abs=0.0)
