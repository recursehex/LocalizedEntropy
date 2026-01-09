from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from localized_entropy.utils import dedupe
from localized_entropy.data.common import build_condition_encoder, encode_conditions


def _safe_nrows(read_rows: Optional[int]):
    if read_rows is None:
        return None
    read_rows = int(read_rows)
    return None if read_rows <= 0 else read_rows


def _extract_hour_parts(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    hour_str = series.astype(str).str.zfill(8)
    date_str = hour_str.str.slice(0, 6)
    hh_str = hour_str.str.slice(6, 8)
    return date_str, hh_str


def _add_derived_time_features(df: pd.DataFrame, hour_col: str = "hour") -> None:
    if hour_col not in df.columns:
        return
    date_str, hh_str = _extract_hour_parts(df[hour_col])
    df[hour_col] = pd.to_numeric(hh_str, errors="coerce").fillna(0).astype(int)
    dates = pd.to_datetime(date_str, format="%y%m%d", errors="coerce")
    wd = dates.dt.dayofweek.fillna(-1).astype(int).astype(str)
    df["wd"] = wd
    df["wd_hour"] = wd + "_" + hh_str


def _cap_counts(series: pd.Series, cap: Optional[int]) -> pd.Series:
    if cap is None or cap <= 0:
        return series
    return series.clip(upper=cap)


def _add_device_counters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    device_ip_col: str = "device_ip",
    device_id_col: str = "device_id",
    cap: Optional[int] = 8,
) -> None:
    if device_ip_col not in train_df.columns or device_id_col not in train_df.columns:
        return
    ip_counts = train_df[device_ip_col].value_counts()
    id_counts = train_df[device_id_col].value_counts()
    train_df["device_ip_count"] = _cap_counts(
        train_df[device_ip_col].map(ip_counts).fillna(0).astype(int), cap
    )
    train_df["device_id_count"] = _cap_counts(
        train_df[device_id_col].map(id_counts).fillna(0).astype(int), cap
    )
    test_df["device_ip_count"] = _cap_counts(
        test_df[device_ip_col].map(ip_counts).fillna(0).astype(int), cap
    )
    test_df["device_id_count"] = _cap_counts(
        test_df[device_id_col].map(id_counts).fillna(0).astype(int), cap
    )


def _build_categorical_mapping(series: pd.Series, max_values: Optional[int]) -> Tuple[Dict[str, int], int]:
    series = series.astype(str)
    counts = series.value_counts()
    if (max_values is not None) and (max_values > 0):
        top = counts.nlargest(max_values - 1).index
        mapping = {k: i for i, k in enumerate(top)}
    else:
        mapping = {k: i for i, k in enumerate(counts.index)}
    other_id = len(mapping)
    return mapping, other_id


def _encode_categorical(series: pd.Series, mapping: Dict[str, int], other_id: int) -> np.ndarray:
    series = series.astype(str)
    return series.map(mapping).fillna(other_id).astype(np.int64).to_numpy()


def load_ctr_frames(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[list]]:
    label_col = cfg.get("label_col", "click")
    condition_col = cfg["condition_col"]
    numeric_cols = cfg["numeric_cols"]
    categorical_cols = cfg.get("categorical_cols", []) or []
    filter_col = cfg.get("filter_col")
    weight_col = cfg.get("weight_col")
    derived_time = bool(cfg.get("derived_time", False))
    device_counters = bool(cfg.get("device_counters", False))
    test_has_labels = bool(cfg.get("test_has_labels", False))

    filter_cols = [filter_col] if filter_col else []
    extra_cols: List[str] = []
    if derived_time and "hour" not in numeric_cols and "hour" not in categorical_cols and condition_col != "hour":
        extra_cols.append("hour")
    if device_counters:
        extra_cols.extend(["device_ip", "device_id"])
    train_usecols = dedupe(
        [label_col, condition_col, *numeric_cols, *categorical_cols, *filter_cols, *extra_cols]
        + ([weight_col] if weight_col else [])
    )
    test_usecols = dedupe(
        [condition_col, *numeric_cols, *categorical_cols, *filter_cols, *extra_cols]
        + ([label_col] if test_has_labels else [])
        + ([weight_col] if weight_col else [])
    )

    print("Loading CTR dataset...")
    train_df = pd.read_csv(cfg["train_path"], usecols=train_usecols, nrows=_safe_nrows(cfg.get("read_rows")))
    print("TRAIN DATA HEAD:\n")
    print(train_df.head())
    test_df = pd.read_csv(cfg["test_path"], usecols=test_usecols, nrows=_safe_nrows(cfg.get("read_rows")))
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    if cfg.get("drop_na", True):
        train_df = train_df.dropna(subset=train_usecols)
        test_df = test_df.dropna(subset=test_usecols)

    if derived_time:
        _add_derived_time_features(train_df)
        _add_derived_time_features(test_df)

    if device_counters:
        cap = cfg.get("device_counter_cap", 8)
        if cap is not None:
            cap = int(cap)
        _add_device_counters(
            train_df,
            test_df,
            cap=cap,
        )

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
    numeric_cols = list(cfg["numeric_cols"])
    categorical_cols = cfg.get("categorical_cols", []) or []
    weight_col = cfg.get("weight_col")
    max_conditions = cfg.get("max_conditions")
    derived_time = bool(cfg.get("derived_time", False))
    device_counters = bool(cfg.get("device_counters", False))
    test_has_labels = bool(cfg.get("test_has_labels", False))

    labels = train_df[label_col].to_numpy(dtype=np.float32)
    cond_map, other_id, num_conditions = build_condition_encoder(
        train_df[condition_col], max_conditions
    )
    conds = encode_conditions(train_df[condition_col], cond_map, other_id)
    conds_test = encode_conditions(test_df[condition_col], cond_map, other_id)

    if device_counters:
        for col in ("device_ip_count", "device_id_count"):
            if col not in numeric_cols:
                numeric_cols.append(col)
    if derived_time:
        for col in ("wd", "wd_hour"):
            if col not in categorical_cols:
                categorical_cols.append(col)
    categorical_cols = [
        col for col in dedupe(categorical_cols)
        if col not in numeric_cols and col != condition_col
    ]

    xnum = train_df[numeric_cols].to_numpy(dtype=np.float32)
    xnum_test = test_df[numeric_cols].to_numpy(dtype=np.float32)

    if categorical_cols:
        cat_maps = []
        cat_sizes = []
        max_values = cfg.get("categorical_max_values")
        if max_values is not None:
            max_values = int(max_values)
        for col in categorical_cols:
            mapping, cat_other_id = _build_categorical_mapping(train_df[col], max_values)
            cat_maps.append((mapping, cat_other_id))
            cat_sizes.append(cat_other_id + 1)
        xcat = np.stack(
            [
                _encode_categorical(train_df[col], mapping, cat_other_id)
                for col, (mapping, cat_other_id) in zip(categorical_cols, cat_maps)
            ],
            axis=1,
        )
        xcat_test = np.stack(
            [
                _encode_categorical(test_df[col], mapping, cat_other_id)
                for col, (mapping, cat_other_id) in zip(categorical_cols, cat_maps)
            ],
            axis=1,
        )
    else:
        xcat = np.empty((len(train_df), 0), dtype=np.int64)
        xcat_test = np.empty((len(test_df), 0), dtype=np.int64)
        cat_sizes = []

    if weight_col:
        net_worth = train_df[weight_col].to_numpy(dtype=np.float32)
        net_worth_test = test_df[weight_col].to_numpy(dtype=np.float32)
    else:
        net_worth = np.zeros_like(labels, dtype=np.float32)
        net_worth_test = np.zeros((len(test_df),), dtype=np.float32)

    probs = np.clip(labels, 1e-6, 1.0 - 1e-6)
    labels_test = None
    if test_has_labels and (label_col in test_df.columns):
        labels_test = test_df[label_col].to_numpy(dtype=np.float32)

    return {
        "xnum": xnum,
        "xnum_test": xnum_test,
        "xcat": xcat,
        "xcat_test": xcat_test,
        "labels": labels,
        "labels_test": labels_test,
        "conds": conds,
        "conds_test": conds_test,
        "net_worth": net_worth,
        "net_worth_test": net_worth_test,
        "probs": probs,
        "num_conditions": num_conditions,
        "feature_names": list(numeric_cols),
        "cat_sizes": cat_sizes,
        "cat_cols": list(categorical_cols),
    }
