from types import SimpleNamespace

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

from localized_entropy.analysis import expected_calibration_error
from localized_entropy.experiments import _score_epoch_preds, resolve_train_eval_bundle


def _dummy_loader():
    x = torch.zeros((2, 1), dtype=torch.float32)
    x_cat = torch.zeros((2, 0), dtype=torch.long)
    c = torch.zeros(2, dtype=torch.long)
    y = torch.zeros(2, dtype=torch.float32)
    w = torch.ones(2, dtype=torch.float32)
    return DataLoader(TensorDataset(x, x_cat, c, y, w), batch_size=2, shuffle=False)


def test_score_epoch_preds_ece_small_matches_reference():
    preds = np.array([1e-4, 5e-4, 0.02, 0.4], dtype=np.float64)
    labels = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    mask = preds <= 0.01
    expected, _ = expected_calibration_error(preds[mask], labels[mask], bins=5, min_count=1)
    score = _score_epoch_preds(
        "ece_small",
        preds,
        labels,
        ece_bins=5,
        ece_min_count=1,
        small_prob_max=0.01,
        small_prob_quantile=0.1,
    )
    assert score == expected


def test_resolve_train_eval_bundle_train_split_shape_and_values():
    loader = _dummy_loader()
    splits = SimpleNamespace(
        y_eval=np.array([0.0, 1.0], dtype=np.float64),
        c_eval=np.array([0, 1], dtype=np.int64),
        y_train=np.array([1.0, 0.0], dtype=np.float64),
        c_train=np.array([1, 0], dtype=np.int64),
    )
    loaders = SimpleNamespace(train_loader=loader, eval_loader=loader)
    train_eval_loader, train_eval_conds, train_eval_name = resolve_train_eval_bundle(
        "train",
        loader,
        eval_labels=None,
        eval_conds=None,
        eval_name="Eval",
        loaders=loaders,
        splits=splits,
    )
    assert train_eval_loader is loader
    assert train_eval_name == "Train"
    np.testing.assert_array_equal(train_eval_conds, splits.c_train)
