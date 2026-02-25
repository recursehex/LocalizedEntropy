import numpy as np
import pandas as pd
import pytest

from localized_entropy.analysis import expected_calibration_error, per_condition_metrics
from localized_entropy.compare import build_wilcoxon_summary


def test_expected_calibration_error_supports_all_methods():
    """ECE backends should all return finite values and a standard table schema."""
    preds = np.array([0.01, 0.03, 0.05, 0.2, 0.4, 0.6, 0.85, 0.95], dtype=np.float64)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
    methods = ("custom", "adaptive", "smooth", "adaptive_lib", "smooth_lib")

    for method in methods:
        ece, table = expected_calibration_error(
            preds,
            labels,
            bins=4,
            min_count=1,
            method=method,
            smooth_grid_bins=32,
        )
        assert np.isfinite(ece)
        assert list(table.columns) == ["bin", "count", "avg_pred", "avg_label", "abs_gap"]
        assert len(table) > 0


def test_expected_calibration_error_rejects_unknown_method():
    """Unknown ECE method names should fail fast."""
    with pytest.raises(ValueError):
        expected_calibration_error(
            np.array([0.1, 0.9], dtype=np.float64),
            np.array([0, 1], dtype=np.float64),
            method="unknown",
        )


def test_per_condition_metrics_accepts_smooth_ece():
    """Per-condition metric path should accept non-default ECE backends."""
    preds = np.array([0.01, 0.02, 0.7, 0.8, 0.2, 0.3], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.float64)
    conds = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    frame = per_condition_metrics(
        preds,
        labels,
        conds,
        bins=4,
        min_count=1,
        small_prob_max=0.5,
        ece_method="smooth",
        ece_smooth_grid_bins=16,
    )

    assert len(frame) == 2
    assert np.isfinite(frame["ece"]).all()


def test_build_wilcoxon_summary_orients_one_sided_tests_for_accuracy():
    """For higher-is-better metrics, one-sided tests should still align with LE-favoring deltas."""
    bce_metrics = pd.DataFrame({"accuracy": [0.70, 0.71, 0.72, 0.73, 0.74]})
    le_metrics = pd.DataFrame({"accuracy": [0.75, 0.76, 0.77, 0.78, 0.79]})

    summary = build_wilcoxon_summary(
        bce_metrics,
        le_metrics,
        metrics={"accuracy": False},
        alternative="greater",
    )

    assert len(summary) == 1
    assert summary.iloc[0]["metric"] == "accuracy"
    assert summary.iloc[0]["mean_delta"] > 0.0
    assert summary.iloc[0]["p_value"] < 0.1


def test_build_wilcoxon_summary_all_ties_returns_non_significant_result():
    """All-zero paired deltas should return a stable non-significant Wilcoxon output."""
    bce_metrics = pd.DataFrame({"ece": [0.1, 0.1, 0.1, 0.1]})
    le_metrics = pd.DataFrame({"ece": [0.1, 0.1, 0.1, 0.1]})

    summary = build_wilcoxon_summary(
        bce_metrics,
        le_metrics,
        metrics={"ece": True},
        alternative="greater",
    )

    assert len(summary) == 1
    assert summary.iloc[0]["n_nonzero"] == 0
    assert summary.iloc[0]["p_value"] == pytest.approx(1.0)
