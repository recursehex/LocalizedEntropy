from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from localized_entropy.data.yambda import maybe_prepare_yambda_dataset


pyarrow = pytest.importorskip("pyarrow")


def _small_multi_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": [1, 1, 2, 2, 3, 3, 4, 4],
            "item_id": [10, 11, 10, 12, 11, 13, 14, 15],
            "timestamp": [100, 101, 102, 103, 104, 105, 106, 107],
            "is_organic": [1, 0, 1, 1, 0, 1, 0, 1],
            "event_type": ["listen", "like", "dislike", "listen", "like", "unlike", "listen", "like"],
            "played_ratio_pct": [75.0, None, None, 80.0, None, None, 55.0, None],
            "track_length_seconds": [180.0, None, None, 210.0, None, None, 150.0, None],
        }
    )


def test_maybe_prepare_yambda_dataset_creates_ctr_csvs(tmp_path: Path):
    source_path = tmp_path / "multi_event.parquet"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    _small_multi_event_frame().to_parquet(source_path, index=False)
    ctr_cfg = {
        "dataset_name": "yambda",
        "auto_prepare": True,
        "download_if_missing": False,
        "source_parquet_path": str(source_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "label_col": "click",
        "yambda_positive_event_types": ["like"],
        "yambda_test_fraction": 0.25,
        "yambda_hash_mod": 8,
        "yambda_prepare_batch_size_rows": 4,
    }

    maybe_prepare_yambda_dataset(ctr_cfg)

    assert train_path.exists()
    assert test_path.exists()
    assert ctr_cfg["test_has_labels"] is True

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    assert len(full_df) == 8
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert set(full_df.columns) == {
        "click",
        "item_id",
        "timestamp",
        "played_ratio_pct",
        "track_length_seconds",
        "uid",
        "is_organic",
        "event_type",
    }
    assert full_df["click"].sum() == 3


def test_maybe_prepare_yambda_dataset_noop_for_non_yambda(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    ctr_cfg = {
        "dataset_name": "avazu",
        "auto_prepare": True,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }

    maybe_prepare_yambda_dataset(ctr_cfg)

    assert not train_path.exists()
    assert not test_path.exists()
