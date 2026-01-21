from pathlib import Path
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
    # Split hour into hour-of-day and day-of-week categorical features.
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
    # Count device IDs in train and apply the same counts to test for consistency.
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


def _normalize_filter_mode(mode: Optional[str]) -> str:
    if not mode:
        return ""
    mode = mode.strip().lower()
    if mode in ("ids", "id_list", "list", "include"):
        return "ids"
    if mode in ("top", "top_k", "highest", "most"):
        return "top_k"
    if mode in ("bottom", "bottom_k", "lowest", "least"):
        return "bottom_k"
    if mode in ("none", "off", "disabled"):
        return "none"
    return mode


def _normalize_filter_metric(metric: Optional[str]) -> str:
    if not metric:
        return "frequency"
    metric = metric.strip().lower()
    if metric in ("count", "frequency", "impressions", "rows"):
        return "frequency"
    if metric in ("mean", "avg", "rate", "click_rate", "clickrate"):
        return "mean"
    print(f"[WARN] Unknown filter metric '{metric}'; defaulting to frequency.")
    return "frequency"


def _normalize_filter_order(order: Optional[str]) -> Optional[bool]:
    if not order:
        return None
    order = order.strip().lower()
    if order in ("asc", "ascending", "low", "lowest"):
        return True
    if order in ("desc", "descending", "high", "highest"):
        return False
    return None


def _coerce_filter_values(values: Optional[List], series: pd.Series) -> List:
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set, np.ndarray)):
        values = [values]
    values = dedupe(values)
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return [str(v) for v in values]
    if pd.api.types.is_integer_dtype(series):
        coerced = []
        for v in values:
            try:
                coerced.append(int(v))
            except (TypeError, ValueError):
                print(f"[WARN] Could not coerce filter value '{v}' to int; skipping.")
        return coerced
    if pd.api.types.is_numeric_dtype(series):
        coerced = []
        for v in values:
            try:
                coerced.append(float(v))
            except (TypeError, ValueError):
                print(f"[WARN] Could not coerce filter value '{v}' to float; skipping.")
        return coerced
    return values


def _build_filter_stats(train_df: pd.DataFrame, filter_col: str, label_col: str) -> pd.DataFrame:
    return (
        train_df.groupby(filter_col)[label_col]
        .agg(
            frequency="size",
            mean="mean",
            std=lambda s: float(np.std(s.to_numpy(), ddof=0)),
        )
    )


def _sample_series(path: Path, col: str) -> Optional[pd.Series]:
    try:
        sample_df = pd.read_csv(path, usecols=[col], nrows=1000)
    except Exception as exc:
        print(f"[WARN] Failed to sample {col} from {path}: {exc}")
        return None
    if col not in sample_df.columns or sample_df.empty:
        return None
    return sample_df[col]


def _stream_filter_stats(
    path: Path,
    filter_col: str,
    label_col: str,
    *,
    read_rows: Optional[int],
    chunksize: int,
) -> pd.DataFrame:
    counts: Dict = {}
    sums: Dict = {}
    for chunk in pd.read_csv(
        path,
        usecols=[filter_col, label_col],
        chunksize=chunksize,
        nrows=read_rows,
    ):
        grouped = chunk.groupby(filter_col)[label_col].agg(["size", "sum"])
        for idx, row in grouped.iterrows():
            counts[idx] = counts.get(idx, 0) + int(row["size"])
            sums[idx] = sums.get(idx, 0.0) + float(row["sum"])
    if not counts:
        return pd.DataFrame(columns=["frequency", "mean"])
    stats_df = pd.DataFrame(
        {
            "frequency": pd.Series(counts, dtype="int64"),
            "sum": pd.Series(sums, dtype="float64"),
        }
    )
    stats_df["mean"] = stats_df["sum"] / stats_df["frequency"].replace(0, np.nan)
    return stats_df.drop(columns=["sum"])


def _filter_csv_to_ids(
    input_path: Path,
    output_path: Path,
    *,
    filter_col: str,
    ids: List,
    read_rows: Optional[int],
    chunksize: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = False
    row_count = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize, nrows=read_rows):
        filtered = chunk[chunk[filter_col].isin(ids)]
        if filtered.empty:
            continue
        row_count += len(filtered)
        filtered.to_csv(
            output_path,
            mode="w" if not wrote else "a",
            header=not wrote,
            index=False,
        )
        wrote = True
    if not wrote:
        pd.read_csv(input_path, nrows=0).to_csv(output_path, index=False)
    return row_count


def maybe_cache_filtered_ctr(ctr_cfg: Dict) -> None:
    filter_cfg = ctr_cfg.get("filter") or {}
    cache_cfg = filter_cfg.get("cache") or {}
    if not cache_cfg.get("enabled", False):
        return

    legacy_filter_top_k = ctr_cfg.get("filter_top_k")
    if legacy_filter_top_k is not None:
        try:
            legacy_filter_top_k = int(legacy_filter_top_k)
        except (TypeError, ValueError):
            legacy_filter_top_k = None
    if legacy_filter_top_k is not None and legacy_filter_top_k <= 0:
        legacy_filter_top_k = None

    filter_col = (
        filter_cfg.get("col")
        or filter_cfg.get("column")
        or ctr_cfg.get("filter_col")
        or ctr_cfg.get("condition_col")
    )
    if not filter_col:
        print("[WARN] Filter cache enabled but filter column is missing.")
        return

    filter_mode = _normalize_filter_mode(filter_cfg.get("mode"))
    if not filter_mode:
        if filter_cfg.get("ids") or filter_cfg.get("values"):
            filter_mode = "ids"
        elif filter_cfg.get("k") or filter_cfg.get("top_k") or legacy_filter_top_k:
            filter_mode = "top_k"
        else:
            filter_mode = "none"
    filter_enabled = filter_cfg.get("enabled")
    if filter_enabled is None:
        filter_enabled = filter_mode not in ("", "none")
    if not filter_enabled or filter_mode == "none":
        print("[WARN] Filter cache enabled but filter mode is disabled.")
        return

    read_rows = _safe_nrows(ctr_cfg.get("read_rows"))
    chunksize = int(cache_cfg.get("chunksize", 1_000_000))
    label_col = ctr_cfg.get("label_col", "click")
    train_in = Path(ctr_cfg["train_path"])
    test_in = Path(ctr_cfg["test_path"])

    selected_ids: List = []
    if filter_mode == "ids":
        ids = filter_cfg.get("ids") or filter_cfg.get("values") or []
        if not ids:
            print("[WARN] Filter cache enabled but ids list is empty.")
            return
        sample_series = _sample_series(train_in, filter_col)
        if sample_series is not None:
            ids = _coerce_filter_values(ids, sample_series)
        selected_ids = dedupe(ids)
    else:
        stats_df = _stream_filter_stats(
            train_in,
            filter_col,
            label_col,
            read_rows=read_rows,
            chunksize=chunksize,
        )
        if stats_df.empty:
            print("[WARN] Filter cache stats are empty; skipping cache.")
            return
        min_count = filter_cfg.get("min_count")
        if min_count is not None:
            try:
                min_count = int(min_count)
            except (TypeError, ValueError):
                min_count = None
        if min_count is not None and min_count > 0:
            stats_df = stats_df[stats_df["frequency"] >= min_count]
        if stats_df.empty:
            print("[WARN] Filter cache stats empty after min_count; skipping cache.")
            return
        metric = _normalize_filter_metric(filter_cfg.get("metric"))
        order_override = _normalize_filter_order(filter_cfg.get("order"))
        default_ascending = filter_mode == "bottom_k"
        ascending = order_override if order_override is not None else default_ascending
        k = filter_cfg.get("k")
        if k is None and filter_mode == "bottom_k":
            k = filter_cfg.get("bottom_k")
        if k is None:
            k = filter_cfg.get("top_k")
        if k is None:
            k = legacy_filter_top_k
        if k is not None:
            try:
                k = int(k)
            except (TypeError, ValueError):
                k = None
        if k is None or k <= 0:
            print("[WARN] Filter cache enabled but k is missing.")
            return
        selected_ids = (
            stats_df.sort_values(metric, ascending=ascending).head(k).index.to_list()
        )

    selected_ids = dedupe(selected_ids)
    if not selected_ids:
        print("[WARN] Filter cache enabled but no ids selected.")
        return

    filter_cfg["mode"] = "ids"
    filter_cfg["ids"] = selected_ids
    filter_cfg["col"] = filter_col
    ctr_cfg["filter"] = filter_cfg

    train_out = Path(cache_cfg.get("train_path", "data/train_filtered.csv"))
    test_out = Path(cache_cfg.get("test_path", "data/test_filtered.csv"))
    overwrite = bool(cache_cfg.get("overwrite", False))

    if overwrite or not train_out.exists():
        train_rows = _filter_csv_to_ids(
            train_in,
            train_out,
            filter_col=filter_col,
            ids=selected_ids,
            read_rows=read_rows,
            chunksize=chunksize,
        )
    else:
        train_rows = None
        print(f"Using cached train file: {train_out}")

    if train_rows == 0:
        print("[WARN] Cached train filter produced 0 rows; keeping original train_path.")
    else:
        ctr_cfg["train_path"] = str(train_out)
        if train_rows is not None:
            print(f"Cached filtered train rows: {train_rows:,}")

    apply_to_test = bool(filter_cfg.get("apply_to_test", ctr_cfg.get("filter_test", True)))
    if apply_to_test:
        if overwrite or not test_out.exists():
            test_rows = _filter_csv_to_ids(
                test_in,
                test_out,
                filter_col=filter_col,
                ids=selected_ids,
                read_rows=read_rows,
                chunksize=chunksize,
            )
        else:
            test_rows = None
            print(f"Using cached test file: {test_out}")
        if test_rows == 0:
            print("[WARN] Cached test filter produced 0 rows; keeping original test_path.")
        else:
            ctr_cfg["test_path"] = str(test_out)
            if test_rows is not None:
                print(f"Cached filtered test rows: {test_rows:,}")


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
    weight_col = cfg.get("weight_col")
    derived_time = bool(cfg.get("derived_time", False))
    device_counters = bool(cfg.get("device_counters", False))
    test_has_labels = bool(cfg.get("test_has_labels", False))

    filter_cfg = cfg.get("filter") or {}
    legacy_filter_col = cfg.get("filter_col")
    legacy_filter_top_k = cfg.get("filter_top_k")
    if legacy_filter_top_k is not None:
        try:
            legacy_filter_top_k = int(legacy_filter_top_k)
        except (TypeError, ValueError):
            legacy_filter_top_k = None
    if legacy_filter_top_k is not None and legacy_filter_top_k <= 0:
        legacy_filter_top_k = None

    filter_col = filter_cfg.get("col") or filter_cfg.get("column") or legacy_filter_col
    if filter_col is None and filter_cfg:
        filter_col = condition_col

    filter_mode = _normalize_filter_mode(filter_cfg.get("mode"))
    if not filter_mode:
        if filter_cfg.get("ids") or filter_cfg.get("values"):
            filter_mode = "ids"
        elif filter_cfg.get("k") or filter_cfg.get("top_k") or legacy_filter_top_k:
            filter_mode = "top_k"
        else:
            filter_mode = "none"

    filter_enabled = filter_cfg.get("enabled")
    if filter_enabled is None:
        filter_enabled = filter_mode not in ("", "none") and filter_col is not None
    if not filter_enabled:
        filter_mode = "none"

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
    filter_test = bool(filter_cfg.get("apply_to_test", cfg.get("filter_test", True)))
    if filter_mode != "none" and filter_col:
        selected_values = []
        selected_stats = None
        if filter_mode == "ids":
            ids = filter_cfg.get("ids") or filter_cfg.get("values") or []
            selected_values = _coerce_filter_values(ids, train_df[filter_col])
        else:
            metric = _normalize_filter_metric(filter_cfg.get("metric"))
            min_count = filter_cfg.get("min_count")
            if min_count is not None:
                try:
                    min_count = int(min_count)
                except (TypeError, ValueError):
                    min_count = None
            order_override = _normalize_filter_order(filter_cfg.get("order"))
            if filter_mode == "bottom_k":
                default_ascending = True
            else:
                default_ascending = False
            ascending = order_override if order_override is not None else default_ascending
            k = filter_cfg.get("k")
            if k is None:
                if filter_mode == "bottom_k":
                    k = filter_cfg.get("bottom_k")
            if k is None:
                k = filter_cfg.get("top_k")
            if k is None:
                k = legacy_filter_top_k
            if k is not None:
                try:
                    k = int(k)
                except (TypeError, ValueError):
                    k = None
            if k is None or k <= 0:
                print("[WARN] Filter configured but k is missing; skipping filter.")
            else:
                stats_all = _build_filter_stats(train_df, filter_col, label_col)
                if min_count is not None and min_count > 0:
                    stats_all = stats_all[stats_all["frequency"] >= min_count]
                if stats_all.empty:
                    print("[WARN] Filter stats are empty; skipping filter.")
                else:
                    stats_sorted = stats_all.sort_values(metric, ascending=ascending)
                    selected_stats = stats_sorted.head(k)
                    selected_values = selected_stats.index.to_list()

        selected_values = dedupe(selected_values)
        if selected_values:
            filter_label = f"{filter_mode} {len(selected_values)} values of {filter_col}"
            print(f"Filtering to {filter_label}: {selected_values}")
            train_df = train_df[train_df[filter_col].isin(selected_values)].copy()
            present = set(train_df[filter_col].unique())
            missing = [v for v in selected_values if v not in present]
            if missing:
                print(f"[WARN] Missing {len(missing)} filter values in training data: {missing}")
                selected_values = [v for v in selected_values if v in present]
            if filter_test:
                test_unfiltered = test_df
                test_df = test_df[test_df[filter_col].isin(selected_values)].copy()
                if len(test_df) == 0 and len(test_unfiltered) > 0:
                    print("[WARN] Filtered test set is empty; keeping unfiltered test rows.")
                    test_df = test_unfiltered
            print(f"Filtered Train rows: {len(train_df):,} | Filtered Test rows: {len(test_df):,}")
            if selected_stats is None:
                stats_df = _build_filter_stats(train_df, filter_col, label_col).reindex(selected_values)
            else:
                stats_df = selected_stats.reindex(selected_values)
            top_values = selected_values
        else:
            print("[WARN] Filter configured but no values selected; skipping filter.")

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
    max_cond = int(conds.max()) if conds.size > 0 else -1
    if conds_test.size > 0:
        max_cond = max(max_cond, int(conds_test.max()))
    if max_cond >= 0:
        num_conditions = max_cond + 1

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
