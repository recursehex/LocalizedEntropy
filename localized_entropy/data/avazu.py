from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


# Header emitted by the Avazu CTR competition `train` file.
AVAZU_LABEL_COL = "click"


def _as_bool(value, default: bool = False) -> bool:
    """Coerce a config value to bool with a fallback default."""
    if value is None:
        return default
    return bool(value)


def _coerce_split_fraction(value: object, default: float = 0.1) -> float:
    """Parse and validate a split fraction in (0, 1)."""
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        fraction = default
    if not (0.0 < fraction < 1.0):
        raise ValueError("avazu_test_fraction must be in (0, 1).")
    return fraction


def _coerce_hash_mod(value: object, default: int = 1000) -> int:
    """Parse and validate a positive hash modulus > 1."""
    try:
        hash_mod = int(value)
    except (TypeError, ValueError):
        hash_mod = default
    if hash_mod <= 1:
        raise ValueError("avazu_hash_mod must be > 1.")
    return hash_mod


def _detect_compression(path: Path) -> Optional[str]:
    """Return 'gzip' when the file has gzip magic bytes, else None."""
    try:
        with path.open("rb") as handle:
            magic = handle.read(2)
    except OSError:
        return None
    if magic == b"\x1f\x8b":
        return "gzip"
    return None


def _find_source_file(base_dir: Path, filenames: Iterable[str]) -> Optional[Path]:
    """Return the first matching file under base_dir (recursive fallback)."""
    for name in filenames:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    # Fall back to a recursive search for any matching basename.
    names = {str(n).lower() for n in filenames}
    for candidate in sorted(base_dir.rglob("*")):
        if candidate.is_file() and candidate.name.lower() in names:
            return candidate
    return None


def _download_competition(handle: str) -> Path:
    """Download an Avazu competition bundle via kagglehub and return its dir."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download the Avazu competition automatically. "
            "Install it with: pip install kagglehub"
        ) from exc

    try:
        path = kagglehub.competition_download(handle)
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message.
        raise RuntimeError(
            f"kagglehub failed to download competition '{handle}'. This usually means "
            "Kaggle credentials are missing or the competition rules have not been "
            "accepted. Provide a ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY "
            "env vars) and accept the rules at "
            "https://www.kaggle.com/competitions/avazu-ctr-prediction/rules, or set a "
            "local ctr.datasets.avazu.source_csv_path instead."
        ) from exc
    return Path(path)


def _resolve_source_train_file(ctr_cfg: Dict) -> Path:
    """Locate a labeled Avazu train file (local override or Kaggle download)."""
    train_candidates = ["train.gz", "train.csv", "train"]

    source_csv_path = ctr_cfg.get("source_csv_path")
    if source_csv_path:
        source = Path(str(source_csv_path))
        if source.exists():
            return source
        raise FileNotFoundError(
            f"Configured ctr.datasets.avazu.source_csv_path does not exist: {source}."
        )

    if not _as_bool(ctr_cfg.get("download_if_missing", False), default=False):
        raise FileNotFoundError(
            "Avazu train file is missing. Set ctr.datasets.avazu.download_if_missing=true "
            "to fetch it via kagglehub, or provide ctr.datasets.avazu.source_csv_path."
        )

    handle = str(ctr_cfg.get("kaggle_competition", "avazu-ctr-prediction"))
    download_dir = _download_competition(handle)
    source = _find_source_file(download_dir, train_candidates)
    if source is None:
        raise FileNotFoundError(
            f"Could not locate an Avazu train file ({train_candidates}) under the "
            f"kagglehub download directory: {download_dir}."
        )
    print(f"[INFO] Using Avazu source train file from kagglehub: {source}.")
    return source


def _read_source_chunks(source: Path, chunk_rows: int):
    """Yield pandas chunks from a (possibly gzipped) Avazu source CSV."""
    compression = _detect_compression(source)
    if compression is None and source.suffix == ".gz":
        compression = "gzip"
    return pd.read_csv(
        source,
        chunksize=chunk_rows,
        compression=compression if compression else "infer",
        dtype=str,
        keep_default_na=False,
    )


def _split_labeled_source(source: Path, train_path: Path, test_path: Path, ctr_cfg: Dict) -> None:
    """Deterministically split a labeled Avazu source CSV into train/test CSVs."""
    chunk_rows = int(ctr_cfg.get("avazu_prepare_batch_size_rows", 1_000_000) or 1_000_000)
    label_col = str(ctr_cfg.get("label_col", AVAZU_LABEL_COL))
    test_fraction = _coerce_split_fraction(ctr_cfg.get("avazu_test_fraction", 0.1), default=0.1)
    hash_mod = _coerce_hash_mod(ctr_cfg.get("avazu_hash_mod", 1000), default=1000)
    threshold = int(round(test_fraction * hash_mod))
    threshold = max(1, min(threshold, hash_mod - 1))

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    for existing in (train_path, test_path):
        if existing.exists():
            existing.unlink()

    wrote_train_header = False
    wrote_test_header = False
    train_rows = 0
    test_rows = 0
    global_row = 0

    for chunk_idx, chunk in enumerate(_read_source_chunks(source, chunk_rows), start=1):
        if label_col not in chunk.columns:
            raise KeyError(
                f"Avazu source {source} is missing the label column '{label_col}'. "
                "The Kaggle competition test set is unlabeled; point at the labeled "
                "train file (ctr.datasets.avazu.source_csv_path) instead."
            )
        idx = np.arange(global_row, global_row + len(chunk), dtype=np.int64)
        global_row += len(chunk)
        test_mask = (idx % hash_mod) < threshold
        train_df = chunk.loc[~test_mask]
        test_df = chunk.loc[test_mask]

        if not train_df.empty:
            train_df.to_csv(train_path, mode="a", index=False, header=not wrote_train_header)
            wrote_train_header = True
            train_rows += len(train_df)
        if not test_df.empty:
            test_df.to_csv(test_path, mode="a", index=False, header=not wrote_test_header)
            wrote_test_header = True
            test_rows += len(test_df)

        print(
            f"[INFO] Avazu split chunk {chunk_idx}: processed={len(chunk):,} "
            f"(train={len(train_df):,}, test={len(test_df):,})."
        )

    if train_rows == 0 or test_rows == 0:
        raise RuntimeError(
            f"Avazu split produced an empty split (train_rows={train_rows}, test_rows={test_rows}). "
            "Adjust avazu_test_fraction or avazu_hash_mod."
        )
    print(f"[INFO] Prepared Avazu CSV splits: train={train_rows:,}, test={test_rows:,}.")


def maybe_prepare_avazu_dataset(ctr_cfg: Dict) -> None:
    """Ensure Avazu dataset files exist, optionally downloading via kagglehub.

    The Kaggle Avazu competition ships a labeled ``train`` file and an unlabeled
    ``test`` file. To produce a labeled, per-condition-evaluable test set that
    shares conditions with the training split (matching the Criteo/Yambda
    preparation flow), this splits the labeled train file deterministically into
    ``train_path`` / ``test_path`` by row-index modulo.
    """
    dataset_name = str(ctr_cfg.get("dataset_name", "")).lower().strip()
    if dataset_name != "avazu":
        return
    if not _as_bool(ctr_cfg.get("auto_prepare", False), default=False):
        return

    train_path = Path(str(ctr_cfg["train_path"]))
    test_path = Path(str(ctr_cfg["test_path"]))
    if train_path.exists() and test_path.exists():
        ctr_cfg["test_has_labels"] = True
        return

    source = _resolve_source_train_file(ctr_cfg)
    print(
        f"[INFO] Preparing Avazu train/test CSVs from source {source} -> "
        f"train={train_path}, test={test_path}."
    )
    _split_labeled_source(source, train_path, test_path, ctr_cfg)
    ctr_cfg["test_has_labels"] = True
