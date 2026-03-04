from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


YAMBDA_MULTI_EVENT_COLUMNS = [
    "uid",
    "item_id",
    "timestamp",
    "is_organic",
    "event_type",
    "played_ratio_pct",
    "track_length_seconds",
]


def _as_bool(value, default: bool = False) -> bool:
    """Coerce a config value to bool with a fallback default."""
    if value is None:
        return default
    return bool(value)


def _normalize_event_values(values: Iterable[str] | None, default: Iterable[str]) -> set[str]:
    """Normalize event-type values to a lowercase set."""
    if values is None:
        values = default
    normalized = {str(v).strip().lower() for v in values if str(v).strip()}
    if not normalized:
        normalized = {str(v).strip().lower() for v in default if str(v).strip()}
    return normalized


def _resolve_source_parquet_path(ctr_cfg: Dict) -> Path:
    """Resolve the source parquet path for Yambda multi-event data."""
    source_path = ctr_cfg.get("source_parquet_path", "data/yambda/multi_event.parquet")
    return Path(str(source_path))


def _download_source_parquet_if_missing(source_path: Path, ctr_cfg: Dict) -> Path:
    """Download Yambda multi-event parquet from Hugging Face when requested."""
    if source_path.exists():
        return source_path

    subfolder = str(ctr_cfg.get("hf_subfolder", "flat/50m"))
    filename = str(ctr_cfg.get("hf_filename", "multi_event.parquet"))
    cached_local_path = source_path.parent / subfolder / filename
    if cached_local_path.exists():
        return cached_local_path

    if not _as_bool(ctr_cfg.get("download_if_missing", False), default=False):
        raise FileNotFoundError(
            f"Yambda source parquet not found at {source_path}. "
            "Set ctr.datasets.yambda.download_if_missing=true or provide source_parquet_path."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download yandex/yambda automatically. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    repo_id = str(ctr_cfg.get("hf_repo_id", "yandex/yambda"))
    source_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        subfolder=subfolder,
        local_dir=str(source_path.parent),
    )
    return Path(downloaded)


def _build_test_mask(df: pd.DataFrame, *, test_fraction: float, hash_mod: int) -> pd.Series:
    """Create a deterministic hash split mask for test rows."""
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("yambda_test_fraction must be in (0, 1).")
    if hash_mod <= 1:
        raise ValueError("yambda_hash_mod must be > 1.")

    threshold = int(round(test_fraction * hash_mod))
    threshold = max(1, min(threshold, hash_mod - 1))

    uid = pd.to_numeric(df["uid"], errors="coerce").fillna(0).astype(np.uint64)
    item_id = pd.to_numeric(df["item_id"], errors="coerce").fillna(0).astype(np.uint64)
    timestamp = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.uint64)

    mixed = (
        uid * np.uint64(11400714819323198485)
        + item_id * np.uint64(14029467366897019727)
        + timestamp * np.uint64(1609587929392839161)
    )
    hashed = mixed % np.uint64(hash_mod)
    return hashed < np.uint64(threshold)


def _prepare_multi_event_batch(df: pd.DataFrame, ctr_cfg: Dict) -> pd.DataFrame:
    """Transform a raw Yambda multi-event batch into CTR-style rows."""
    missing = [col for col in YAMBDA_MULTI_EVENT_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required Yambda columns: {missing}")

    work = df[YAMBDA_MULTI_EVENT_COLUMNS].copy()
    work["event_type"] = work["event_type"].astype(str).str.strip().str.lower()

    work["uid"] = pd.to_numeric(work["uid"], errors="coerce").fillna(-1).astype(np.int64)
    work["item_id"] = pd.to_numeric(work["item_id"], errors="coerce").fillna(-1).astype(np.int64)
    work["timestamp"] = pd.to_numeric(work["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    work["is_organic"] = (
        pd.to_numeric(work["is_organic"], errors="coerce").fillna(0).clip(lower=0, upper=1).astype(np.int8)
    )
    work["played_ratio_pct"] = (
        pd.to_numeric(work["played_ratio_pct"], errors="coerce").fillna(0).clip(lower=0, upper=100).astype(np.float32)
    )
    work["track_length_seconds"] = (
        pd.to_numeric(work["track_length_seconds"], errors="coerce").fillna(0).clip(lower=0).astype(np.float32)
    )

    positive_event_types = _normalize_event_values(
        ctr_cfg.get("yambda_positive_event_types"),
        default=["like"],
    )
    listen_min_played_ratio_pct = float(ctr_cfg.get("listen_min_played_ratio_pct", 0.0) or 0.0)

    label = work["event_type"].isin(positive_event_types)
    if "listen" in positive_event_types and listen_min_played_ratio_pct > 0.0:
        listen_mask = work["event_type"].eq("listen")
        label = label & (~listen_mask | (work["played_ratio_pct"] >= listen_min_played_ratio_pct))

    label_col = str(ctr_cfg.get("label_col", "click"))
    work[label_col] = label.astype(np.int8)

    out_cols = [
        label_col,
        "item_id",
        "timestamp",
        "played_ratio_pct",
        "track_length_seconds",
        "uid",
        "is_organic",
        "event_type",
    ]
    return work[out_cols]


def _convert_multi_event_to_ctr_csvs(source_parquet: Path, train_path: Path, test_path: Path, ctr_cfg: Dict) -> None:
    """Convert Yambda multi-event parquet into train/test CSV files."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read yambda multi_event.parquet. "
            "Install it with: pip install pyarrow"
        ) from exc

    batch_size_rows = int(ctr_cfg.get("yambda_prepare_batch_size_rows", 1_000_000) or 1_000_000)
    test_fraction = float(ctr_cfg.get("yambda_test_fraction", 0.1) or 0.1)
    hash_mod = int(ctr_cfg.get("yambda_hash_mod", 1000) or 1000)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_train_header = False
    wrote_test_header = False
    train_rows = 0
    test_rows = 0
    total_rows = 0

    parquet = pq.ParquetFile(source_parquet)
    for batch_idx, batch in enumerate(
        parquet.iter_batches(columns=YAMBDA_MULTI_EVENT_COLUMNS, batch_size=batch_size_rows),
        start=1,
    ):
        raw_df = batch.to_pandas()
        chunk_df = _prepare_multi_event_batch(raw_df, ctr_cfg)
        test_mask = _build_test_mask(chunk_df, test_fraction=test_fraction, hash_mod=hash_mod)
        train_df = chunk_df.loc[~test_mask]
        test_df = chunk_df.loc[test_mask]

        if not train_df.empty:
            train_df.to_csv(train_path, mode="a", index=False, header=not wrote_train_header)
            wrote_train_header = True
            train_rows += len(train_df)
        if not test_df.empty:
            test_df.to_csv(test_path, mode="a", index=False, header=not wrote_test_header)
            wrote_test_header = True
            test_rows += len(test_df)
        total_rows += len(chunk_df)

        print(
            f"[INFO] Yambda chunk {batch_idx}: processed={len(chunk_df):,} "
            f"(train={len(train_df):,}, test={len(test_df):,})."
        )

    if train_rows == 0 or test_rows == 0:
        raise RuntimeError(
            f"Yambda conversion produced an empty split (train_rows={train_rows}, test_rows={test_rows}). "
            "Adjust yambda_test_fraction or yambda_hash_mod."
        )
    print(
        f"[INFO] Prepared Yambda CSVs from {source_parquet}: "
        f"total={total_rows:,}, train={train_rows:,}, test={test_rows:,}."
    )


def maybe_prepare_yambda_dataset(ctr_cfg: Dict) -> None:
    """Ensure Yambda 50M flat multi-event data is ready as train/test CSV files."""
    dataset_name = str(ctr_cfg.get("dataset_name", "")).lower().strip()
    if dataset_name != "yambda":
        return
    if not _as_bool(ctr_cfg.get("auto_prepare", False), default=False):
        return

    train_path = Path(str(ctr_cfg["train_path"]))
    test_path = Path(str(ctr_cfg["test_path"]))
    if train_path.exists() and test_path.exists():
        ctr_cfg["test_has_labels"] = True
        return

    source_parquet = _resolve_source_parquet_path(ctr_cfg)
    source_parquet = _download_source_parquet_if_missing(source_parquet, ctr_cfg)
    print(
        "[INFO] Preparing yandex/yambda flat/50m multi_event parquet "
        f"for pandas CSV loading -> train={train_path}, test={test_path}."
    )
    _convert_multi_event_to_ctr_csvs(source_parquet, train_path, test_path, ctr_cfg)
    ctr_cfg["test_has_labels"] = True
