from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def build_condition_encoder(train_series: pd.Series, max_conditions: Optional[int]):
    """Build a mapping from condition values to integer IDs."""
    counts = train_series.value_counts()
    if (max_conditions is not None) and (max_conditions > 0):
        top = counts.nlargest(max_conditions - 1).index
        mapping = {k: i for i, k in enumerate(top)}
    else:
        mapping = {k: i for i, k in enumerate(counts.index)}
    other_id = len(mapping)
    num_conditions = other_id + 1
    return mapping, other_id, num_conditions


def encode_conditions(series: pd.Series, mapping: Dict, other_id: int) -> np.ndarray:
    """Encode condition values into IDs with an 'other' bucket."""
    return series.map(mapping).fillna(other_id).astype(np.int64).to_numpy()


def train_eval_split(n_total: int, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into train/eval partitions with a RNG permutation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    split = int(train_ratio * n_total)
    return idx[:split], idx[split:]


def standardize_features(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    x_test: Optional[np.ndarray],
    eps: float,
):
    """Standardize features using train mean/std and apply to eval/test."""
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < eps] = 1.0
    x_train_n = (x_train - mu) / sd
    x_eval_n = (x_eval - mu) / sd
    x_test_n = None
    if x_test is not None:
        x_test_n = (x_test - mu) / sd
    return x_train_n, x_eval_n, x_test_n, {"mean": mu, "std": sd}
