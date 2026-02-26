from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from localized_entropy.config import load_and_resolve
from localized_entropy.data.pipeline import prepare_data
from localized_entropy.experiments import resolve_eval_bundle


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.json"


def _small_synthetic_cfg(*, use_test_set: bool, test_split: float) -> dict:
    cfg = load_and_resolve(str(CONFIG_PATH))
    cfg = deepcopy(cfg)
    cfg["data"]["source"] = "synthetic"
    cfg["data"]["train_split"] = 0.8
    cfg["data"]["standardize"] = False
    cfg["data"]["shuffle_test"] = False
    cfg["data"]["use_test_set"] = use_test_set

    cfg["device"]["move_dataset_to_cuda"] = False
    cfg["device"]["allow_dataloader_workers"] = False

    cfg["training"]["batch_size"] = 8

    cfg["synthetic"]["min_samples_per_condition"] = 10
    cfg["synthetic"]["max_samples_per_condition"] = 10
    cfg["synthetic"]["test_split"] = test_split
    cfg["synthetic"]["reweighting"]["enabled"] = False
    return cfg


def _small_ctr_cfg(tmp_path: Path, *, use_test_set: bool, test_has_labels: bool) -> dict:
    cfg = load_and_resolve(str(CONFIG_PATH))
    cfg = deepcopy(cfg)
    cfg["data"]["source"] = "ctr"
    cfg["data"]["ctr_dataset"] = "avazu"
    cfg["data"]["train_split"] = 0.8
    cfg["data"]["standardize"] = False
    cfg["data"]["shuffle_test"] = False
    cfg["data"]["use_test_set"] = use_test_set

    cfg["device"]["move_dataset_to_cuda"] = False
    cfg["device"]["allow_dataloader_workers"] = False

    cfg["training"]["batch_size"] = 4

    cfg["ctr"]["warn_root_csv"] = False
    dataset_cfg = cfg["ctr"]["datasets"]["avazu"]
    dataset_cfg["train_path"] = str(tmp_path / "train.csv")
    dataset_cfg["test_path"] = str(tmp_path / "test.csv")
    dataset_cfg["read_rows"] = None
    dataset_cfg["numeric_cols"] = ["num"]
    dataset_cfg["categorical_cols"] = ["feature"]
    dataset_cfg["categorical_max_values"] = None
    dataset_cfg["derived_time"] = False
    dataset_cfg["device_counters"] = False
    dataset_cfg["condition_col"] = "ad"
    dataset_cfg["label_col"] = "click"
    dataset_cfg["max_conditions"] = None
    dataset_cfg["drop_na"] = True
    dataset_cfg["test_has_labels"] = test_has_labels
    dataset_cfg["plot_sample_size"] = 0
    dataset_cfg["balance_by_condition"] = False
    dataset_cfg["filter"] = {"enabled": False, "mode": "none"}
    dataset_cfg["filter_col"] = None
    dataset_cfg["filter_top_k"] = 0
    return cfg


def test_prepare_data_synthetic_builds_test_split_when_enabled():
    cfg = _small_synthetic_cfg(use_test_set=True, test_split=0.25)

    bundle = prepare_data(cfg, device=torch.device("cpu"), use_cuda=False, use_mps=False)
    splits = bundle.splits

    assert splits.x_test is not None
    assert splits.y_test is not None
    assert splits.c_test is not None
    assert len(splits.y_test) == 10
    assert len(splits.y_train) + len(splits.y_eval) == 30
    assert splits.test_labels_available is True
    assert bundle.loaders.test_loader is not None


def test_prepare_data_synthetic_disables_test_split_when_disabled():
    cfg = _small_synthetic_cfg(use_test_set=False, test_split=0.25)

    bundle = prepare_data(cfg, device=torch.device("cpu"), use_cuda=False, use_mps=False)
    splits = bundle.splits

    assert splits.x_test is None
    assert splits.y_test is None
    assert splits.c_test is None
    assert splits.test_labels_available is False
    assert bundle.loaders.test_loader is None


def test_prepare_data_ctr_respects_use_test_set_flag(tmp_path):
    train_df = pd.DataFrame(
        {
            "click": [0, 1, 0, 1, 0, 0, 1, 1],
            "ad": [1, 1, 2, 2, 3, 3, 4, 4],
            "num": [0.1, 0.5, 0.2, 0.7, 0.9, 0.3, 0.4, 0.8],
            "feature": ["a", "b", "a", "b", "c", "a", "b", "c"],
        }
    )
    test_df = pd.DataFrame(
        {
            "click": [0, 1, 0, 1],
            "ad": [1, 2, 3, 4],
            "num": [0.2, 0.6, 0.1, 0.7],
            "feature": ["a", "b", "c", "a"],
        }
    )
    train_df.to_csv(tmp_path / "train.csv", index=False)
    test_df.to_csv(tmp_path / "test.csv", index=False)

    cfg = _small_ctr_cfg(tmp_path, use_test_set=True, test_has_labels=True)
    enabled_bundle = prepare_data(cfg, device=torch.device("cpu"), use_cuda=False, use_mps=False)
    assert enabled_bundle.splits.x_test is not None
    assert enabled_bundle.splits.test_labels_available is True
    assert enabled_bundle.loaders.test_loader is not None

    cfg = _small_ctr_cfg(tmp_path, use_test_set=False, test_has_labels=True)
    disabled_bundle = prepare_data(cfg, device=torch.device("cpu"), use_cuda=False, use_mps=False)
    assert disabled_bundle.splits.x_test is None
    assert disabled_bundle.splits.y_test is None
    assert disabled_bundle.splits.test_labels_available is False
    assert disabled_bundle.loaders.test_loader is None


def test_resolve_eval_bundle_respects_test_label_availability_flag():
    labels_test = np.array([0.0, 1.0], dtype=np.float32)
    conds_test = np.array([0, 1], dtype=np.int64)

    splits = SimpleNamespace(
        y_train=np.array([1.0], dtype=np.float32),
        y_eval=np.array([0.0], dtype=np.float32),
        y_test=labels_test,
        c_train=np.array([0], dtype=np.int64),
        c_eval=np.array([0], dtype=np.int64),
        c_test=conds_test,
        test_labels_available=False,
    )
    loaders = SimpleNamespace(train_loader="train", eval_loader="eval", test_loader="test")
    cfg = {
        "data": {"source": "ctr"},
        "ctr": {"test_has_labels": True},
        "evaluation": {"split": "test", "use_test_labels": True},
    }

    eval_split, _, eval_labels, eval_conds, eval_name = resolve_eval_bundle(cfg, splits, loaders)
    assert eval_split == "test"
    assert eval_labels is None
    assert np.array_equal(eval_conds, conds_test)
    assert eval_name == "Test"

    splits.test_labels_available = True
    eval_split, _, eval_labels, _, _ = resolve_eval_bundle(cfg, splits, loaders)
    assert eval_split == "test"
    assert np.array_equal(eval_labels, labels_test)
