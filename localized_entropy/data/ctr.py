from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from localized_entropy.utils import dedupe
from localized_entropy.data.common import build_condition_encoder, encode_conditions


def _safe_nrows(read_rows: Optional[int]):
    if read_rows is None:
        return None
    read_rows = int(read_rows)
    return None if read_rows <= 0 else read_rows


def load_ctr_frames(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[list]]:
    label_col = cfg.get("label_col", "click")
    condition_col = cfg["condition_col"]
    numeric_cols = cfg["numeric_cols"]
    filter_col = cfg.get("filter_col")
    weight_col = cfg.get("weight_col")

    filter_cols = [filter_col] if filter_col else []
    train_usecols = dedupe([label_col, condition_col, *numeric_cols, *filter_cols] + ([weight_col] if weight_col else []))
    test_usecols = dedupe([condition_col, *numeric_cols, *filter_cols] + ([weight_col] if weight_col else []))

    print("Loading CTR dataset...")
    train_df = pd.read_csv(cfg["train_path"], usecols=train_usecols, nrows=_safe_nrows(cfg.get("read_rows")))
    test_df = pd.read_csv(cfg["test_path"], usecols=test_usecols, nrows=_safe_nrows(cfg.get("read_rows")))
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    if cfg.get("drop_na", True):
        train_df = train_df.dropna(subset=train_usecols)
        test_df = test_df.dropna(subset=test_usecols)

    stats_df = None
    top_values = None
    filter_top_k = cfg.get("filter_top_k")
    filter_test = bool(cfg.get("filter_test", True))
    if filter_col and filter_top_k:
        top_counts = train_df[filter_col].value_counts()
        top_values = top_counts.head(int(filter_top_k)).index.to_list()
        print(f"Filtering to top {filter_top_k} values of {filter_col}: {top_values}")
        train_df = train_df[train_df[filter_col].isin(top_values)].copy()
        if filter_test:
            test_unfiltered = test_df
            test_df = test_df[test_df[filter_col].isin(top_values)].copy()
            if len(test_df) == 0 and len(test_unfiltered) > 0:
                print("[WARN] Filtered test set is empty; keeping unfiltered test rows.")
                test_df = test_unfiltered
        print(f"Filtered Train rows: {len(train_df):,} | Filtered Test rows: {len(test_df):,}")

        stats_df = (
            train_df.groupby(filter_col)[label_col]
            .agg(
                frequency="size",
                mean="mean",
                std=lambda s: float(np.std(s.to_numpy(), ddof=0)),
            )
            .reindex(top_values)
        )

    return train_df, test_df, stats_df, top_values


def build_ctr_arrays(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Dict) -> Dict:
    label_col = cfg.get("label_col", "click")
    condition_col = cfg["condition_col"]
    numeric_cols = cfg["numeric_cols"]
    weight_col = cfg.get("weight_col")
    max_conditions = cfg.get("max_conditions")

    labels = train_df[label_col].to_numpy(dtype=np.float32)
    cond_map, other_id, num_conditions = build_condition_encoder(
        train_df[condition_col], max_conditions
    )
    conds = encode_conditions(train_df[condition_col], cond_map, other_id)
    conds_test = encode_conditions(test_df[condition_col], cond_map, other_id)

    xnum = train_df[numeric_cols].to_numpy(dtype=np.float32)
    xnum_test = test_df[numeric_cols].to_numpy(dtype=np.float32)

    if weight_col:
        net_worth = train_df[weight_col].to_numpy(dtype=np.float32)
        net_worth_test = test_df[weight_col].to_numpy(dtype=np.float32)
    else:
        net_worth = np.zeros_like(labels, dtype=np.float32)
        net_worth_test = np.zeros((len(test_df),), dtype=np.float32)

    probs = np.clip(labels, 1e-6, 1.0 - 1e-6)

    return {
        "xnum": xnum,
        "xnum_test": xnum_test,
        "labels": labels,
        "conds": conds,
        "conds_test": conds_test,
        "net_worth": net_worth,
        "net_worth_test": net_worth_test,
        "probs": probs,
        "num_conditions": num_conditions,
        "feature_names": list(numeric_cols),
    }
