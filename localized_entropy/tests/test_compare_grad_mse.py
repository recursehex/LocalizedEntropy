import numpy as np
import pytest

from localized_entropy.compare import build_grad_mse_comparison, grad_mse_global_mean
from localized_entropy.training import GradSqStats


def _stats(
    *,
    sums,
    means,
    counts,
    class_mse=(1.0, 1.0),
    class_counts=(1.0, 1.0),
    class_ratio=1.0,
) -> GradSqStats:
    """Build a GradSqStats test fixture."""
    return GradSqStats(
        sum_by_condition=np.asarray(sums, dtype=np.float64),
        mean_by_condition=np.asarray(means, dtype=np.float64),
        count_by_condition=np.asarray(counts, dtype=np.float64),
        class_mse=np.asarray(class_mse, dtype=np.float64),
        class_counts=np.asarray(class_counts, dtype=np.float64),
        class_ratio=float(class_ratio),
    )


def test_grad_mse_global_mean_uses_weighted_sums():
    """Global mean should use per-condition sums/counts, not an unweighted average."""
    stats = _stats(sums=[2.0, 12.0], means=[2.0, 4.0], counts=[1.0, 3.0])
    assert grad_mse_global_mean(stats) == pytest.approx(3.5)


def test_build_grad_mse_comparison_contains_raw_and_normalized_ratios():
    """Comparison output should include raw scale and global-normalized scale diagnostics."""
    bce = _stats(sums=[2.0, 12.0], means=[2.0, 4.0], counts=[1.0, 3.0])
    le = _stats(sums=[2.0, 10.0], means=[1.0, 5.0], counts=[2.0, 2.0])
    comparison = build_grad_mse_comparison(bce, le, condition_label="Condition")

    assert comparison.bce_global_mse == pytest.approx(3.5)
    assert comparison.le_global_mse == pytest.approx(3.0)
    assert comparison.le_over_bce_global_mse == pytest.approx(3.0 / 3.5)

    frame = comparison.per_condition
    assert list(frame.columns) == [
        "Condition",
        "bce_grad_mse",
        "le_grad_mse",
        "le_over_bce_grad_mse",
        "bce_grad_mse_norm_global",
        "le_grad_mse_norm_global",
        "le_over_bce_grad_mse_norm_global",
    ]
    assert frame["le_over_bce_grad_mse"].to_numpy() == pytest.approx([0.5, 1.25])
    assert frame["bce_grad_mse_norm_global"].to_numpy() == pytest.approx(
        [2.0 / 3.5, 4.0 / 3.5]
    )
    assert frame["le_grad_mse_norm_global"].to_numpy() == pytest.approx(
        [1.0 / 3.0, 5.0 / 3.0]
    )
    expected_norm_ratio = np.array([0.5, 1.25]) / (3.0 / 3.5)
    assert frame["le_over_bce_grad_mse_norm_global"].to_numpy() == pytest.approx(
        expected_norm_ratio
    )


def test_build_grad_mse_comparison_handles_zero_global_mean():
    """If a global mean is zero, normalized ratios should return NaN instead of inf."""
    bce = _stats(sums=[0.0, 0.0], means=[0.0, 0.0], counts=[3.0, 5.0])
    le = _stats(sums=[1.0, 1.0], means=[1.0, 1.0], counts=[1.0, 1.0])
    comparison = build_grad_mse_comparison(bce, le)
    frame = comparison.per_condition

    assert np.isnan(comparison.le_over_bce_global_mse)
    assert np.isnan(frame["bce_grad_mse_norm_global"]).all()
    assert np.isnan(frame["le_over_bce_grad_mse_norm_global"]).all()


def test_build_grad_mse_comparison_validates_shape_match():
    """Mismatched per-condition arrays should fail fast."""
    bce = _stats(sums=[1.0, 2.0], means=[1.0, 2.0], counts=[1.0, 1.0])
    le = _stats(sums=[1.0], means=[1.0], counts=[1.0])
    with pytest.raises(ValueError):
        build_grad_mse_comparison(bce, le)
