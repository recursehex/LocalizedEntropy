import warnings

import numpy as np
import pytest

from localized_entropy.data.synthetic import compute_negative_reweighting


def _synthetic_cfg(enabled: bool) -> dict:
    return {
        "reweighting": {
            "enabled": enabled,
            "mode": "fixed",
            "negative_removal_n": 2,
            "base_rate_log10_floor": 1e-6,
        }
    }


def test_reweighting_enabled_emits_deprecation_warning() -> None:
    labels = np.array([0, 0, 1, 0, 0, 1], dtype=np.float32)
    conds = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    rng = np.random.default_rng(0)

    with pytest.warns(FutureWarning, match="deprecated"):
        keep_idx, weights, stats = compute_negative_reweighting(
            labels,
            conds,
            _synthetic_cfg(enabled=True),
            rng=rng,
        )

    assert stats["enabled"] is True
    assert keep_idx.ndim == 1
    assert weights.shape[0] == keep_idx.shape[0]


def test_reweighting_disabled_has_no_warning() -> None:
    labels = np.array([0, 1, 0], dtype=np.float32)
    conds = np.array([0, 0, 1], dtype=np.int64)
    rng = np.random.default_rng(0)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        keep_idx, weights, stats = compute_negative_reweighting(
            labels,
            conds,
            _synthetic_cfg(enabled=False),
            rng=rng,
        )

    assert len(captured) == 0
    assert bool(stats["enabled"]) is False
    assert np.array_equal(keep_idx, np.array([0, 1, 2], dtype=np.int64))
    assert np.all(weights == 1.0)
