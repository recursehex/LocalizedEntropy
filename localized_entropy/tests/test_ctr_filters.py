import pandas as pd

from localized_entropy.data.ctr import (
    _resolve_filter_mode,
    _select_top_count_rate_mix,
    _select_values_from_stats,
)


def test_resolve_filter_mode_infers_top_count_rate_mix_from_preselect_key():
    """Infer the mixed CTR mode when no explicit mode is provided."""
    filter_cfg = {"preselect_k": 200, "high_k": 10, "middle_k": 10, "low_k": 10}
    mode = _resolve_filter_mode(filter_cfg, legacy_filter_top_k=None)
    assert mode == "top_count_rate_mix"


def test_select_top_count_rate_mix_builds_high_mid_low_selection():
    """Select high/mid/low buckets from a top-count candidate pool."""
    ids = [f"ad_{i}" for i in range(1, 11)]
    stats_df = pd.DataFrame(
        {
            "frequency": [1000 - i for i in range(10)],
            "mean": [0.01 * (i + 1) for i in range(10)],
            "median": [0.01 * (i + 1) for i in range(10)],
            "std": [0.1 for _ in range(10)],
        },
        index=ids,
    )
    selected, selected_stats = _select_top_count_rate_mix(
        stats_df,
        {"preselect_k": 10, "high_k": 2, "middle_k": 2, "low_k": 2},
    )
    assert selected == ["ad_10", "ad_9", "ad_6", "ad_5", "ad_1", "ad_2"]
    assert selected_stats is not None
    assert selected_stats.index.to_list() == selected


def test_select_values_from_stats_supports_median_metric():
    """Rank top-k selections by median when requested."""
    stats_df = pd.DataFrame(
        {
            "frequency": [100, 100, 100],
            "mean": [0.20, 0.60, 0.40],
            "median": [0.0, 1.0, 0.5],
            "std": [0.1, 0.1, 0.1],
        },
        index=["ad_low", "ad_high", "ad_mid"],
    )
    selected, selected_stats = _select_values_from_stats(
        stats_df=stats_df,
        filter_mode="top_k",
        filter_cfg={"k": 2, "metric": "median"},
        legacy_filter_top_k=None,
    )
    assert selected == ["ad_high", "ad_mid"]
    assert selected_stats is not None
    assert selected_stats.index.to_list() == selected
