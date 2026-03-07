from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest

from localized_entropy.data import criteo as criteo_module
from localized_entropy.data.criteo import maybe_prepare_criteo_dataset


def _small_criteo_frame() -> pd.DataFrame:
    rows = 12
    data = {"label": [0, 1] * (rows // 2)}
    for idx in range(1, 14):
        data[f"I{idx}"] = [float(idx + r) for r in range(rows)]
    for idx in range(1, 27):
        data[f"C{idx}"] = [int((idx * 1000) + r) for r in range(rows)]
    return pd.DataFrame(data)


def test_maybe_prepare_criteo_dataset_splits_source_csv(tmp_path: Path):
    source_path = tmp_path / "source.csv"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    _small_criteo_frame().to_csv(source_path, index=False)
    ctr_cfg = {
        "dataset_name": "criteo",
        "auto_prepare": True,
        "download_if_missing": False,
        "source_csv_path": str(source_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "label_col": "click",
        "criteo_input_label_col": "label",
        "criteo_test_fraction": 0.25,
        "criteo_hash_mod": 8,
        "criteo_prepare_batch_size_rows": 5,
    }

    maybe_prepare_criteo_dataset(ctr_cfg)

    assert train_path.exists()
    assert test_path.exists()
    assert ctr_cfg["test_has_labels"] is True

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    assert len(full_df) == 12
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert "click" in full_df.columns
    assert "label" not in full_df.columns
    assert all(col in full_df.columns for col in [f"I{i}" for i in range(1, 14)])
    assert all(col in full_df.columns for col in [f"C{i}" for i in range(1, 27)])


def test_maybe_prepare_criteo_dataset_noop_for_non_criteo(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    ctr_cfg = {
        "dataset_name": "avazu",
        "auto_prepare": True,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }

    maybe_prepare_criteo_dataset(ctr_cfg)

    assert not train_path.exists()
    assert not test_path.exists()


def test_maybe_prepare_criteo_dataset_falls_back_to_hf_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    archive_path = tmp_path / "Criteo_x1.zip"
    train_raw = _small_criteo_frame()
    valid_raw = _small_criteo_frame().head(4).copy()
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nested/train.csv", train_raw.to_csv(index=False))
        zf.writestr("nested/valid.csv", valid_raw.to_csv(index=False))

    def _fake_download_from_hf(*, repo_id: str, subfolder: str, filename: str, local_dir: Path) -> Path:
        if filename == "Criteo_x1.zip":
            return archive_path
        raise FileNotFoundError(filename)

    monkeypatch.setattr(criteo_module, "_download_from_hf", _fake_download_from_hf)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    ctr_cfg = {
        "dataset_name": "criteo",
        "auto_prepare": True,
        "download_if_missing": True,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "label_col": "click",
        "criteo_input_label_col": "label",
        "use_valid_as_test": True,
        "hf_repo_id": "reczoo/Criteo_x1",
        "hf_subfolder": "",
        "hf_train_filename": "train.csv",
        "hf_test_filename": "test.csv",
        "hf_valid_filename": "valid.csv",
        "hf_archive_filename": "Criteo_x1.zip",
    }

    maybe_prepare_criteo_dataset(ctr_cfg)

    assert train_path.exists()
    assert test_path.exists()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert "click" in train_df.columns
    assert "label" not in train_df.columns
    assert len(train_df) == len(train_raw)
    assert len(test_df) == len(valid_raw)
