import math

import pytest
import torch

from localized_entropy.losses import localized_entropy


def test_localized_entropy_known_value_balanced():
    logits = torch.zeros(4, dtype=torch.float64)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    conds = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss = localized_entropy(logits, targets, conds)
    expected = 0.5

    assert loss.item() == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_localized_entropy_base_rates_override():
    logits = torch.zeros(4, dtype=torch.float64)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    conds = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    base_rates = torch.tensor([0.25, 0.75], dtype=torch.float64)

    loss = localized_entropy(logits, targets, conds, base_rates=base_rates)

    log2 = math.log(2.0)
    num = 2.0 * log2
    den = -math.log(0.25) - math.log(0.75)
    class_term = num / den
    expected = (class_term + class_term) / 4.0

    assert loss.item() == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_localized_entropy_single_condition_scales_bce():
    logits = torch.tensor([0.3, -0.7, 1.2, -1.5], dtype=torch.float64)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    conds = torch.zeros_like(targets, dtype=torch.long)

    loss = localized_entropy(logits, targets, conds)

    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")(logits, targets)
    ones = targets.sum()
    zeros = targets.numel() - ones
    p = (ones / targets.numel()).clamp(1e-12, 1.0 - 1e-12)
    den = ones * (-torch.log(p)) + zeros * (-torch.log1p(-p))
    expected = bce / den

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6, abs=1e-8)


def test_localized_entropy_condition_weights_scale_terms():
    logits = torch.zeros(4, dtype=torch.float64)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    conds = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    weights = torch.tensor([2.0, 0.5], dtype=torch.float64)

    loss = localized_entropy(logits, targets, conds, condition_weights=weights)
    expected = (2.0 * 1.0 + 0.5 * 1.0) / 4.0

    assert loss.item() == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_localized_entropy_extreme_logits_are_finite():
    logits = torch.tensor([1000.0, -1000.0, 1000.0, -1000.0], dtype=torch.float64)
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    conds = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss = localized_entropy(logits, targets, conds)

    assert torch.isfinite(loss).item()


def test_localized_entropy_grad_signs():
    logits = torch.tensor([2.0, -2.0], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0.0, 1.0], dtype=torch.float64)
    conds = torch.tensor([0, 1], dtype=torch.long)

    loss = localized_entropy(logits, targets, conds)
    loss.backward()

    grads = logits.grad
    assert grads is not None
    assert torch.isfinite(grads).all().item()
    assert grads[0].item() > 0.0
    assert grads[1].item() < 0.0


def test_localized_entropy_gradcheck():
    logits = torch.tensor([0.25, -0.35, 0.5, -0.75], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    conds = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    def loss_fn(z):
        return localized_entropy(z, targets, conds)

    assert torch.autograd.gradcheck(loss_fn, (logits,), eps=1e-6, atol=1e-4, rtol=1e-3)
