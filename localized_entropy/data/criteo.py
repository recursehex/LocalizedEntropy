from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
import shutil
import zipfile

import numpy as np
import pandas as pd


CRITEO_NUMERIC_COLS = [f"I{i}" for i in range(1, 14)]
CRITEO_CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]


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
        raise ValueError("criteo_test_fraction must be in (0, 1).")
    return fraction


def _coerce_hash_mod(value: object, default: int = 1000) -> int:
    """Parse and validate a positive hash modulus > 1."""
    try:
        hash_mod = int(value)
    except (TypeError, ValueError):
        hash_mod = default
    if hash_mod <= 1:
        raise ValueError("criteo_hash_mod must be > 1.")
    return hash_mod


def _resolve_label_columns(ctr_cfg: Dict) -> tuple[str, list[str]]:
    """Return output label column and accepted input aliases."""
    out_label_col = str(ctr_cfg.get("label_col", "click"))
    preferred_in = str(ctr_cfg.get("criteo_input_label_col", "label"))
    aliases = [preferred_in, "label", "click"]
    aliases = [col for idx, col in enumerate(aliases) if col and col not in aliases[:idx]]
    return out_label_col, aliases


def _normalize_criteo_chunk(df: pd.DataFrame, ctr_cfg: Dict) -> pd.DataFrame:
    """Normalize Criteo columns to the configured output schema."""
    out_label_col, label_aliases = _resolve_label_columns(ctr_cfg)
    label_in_col: Optional[str] = None
    for candidate in label_aliases:
        if candidate in df.columns:
            label_in_col = candidate
            break
    if label_in_col is None:
        raise KeyError(f"Criteo source missing label column. Tried aliases: {label_aliases}")

    missing_numeric = [col for col in CRITEO_NUMERIC_COLS if col not in df.columns]
    missing_cat = [col for col in CRITEO_CATEGORICAL_COLS if col not in df.columns]
    if missing_numeric or missing_cat:
        raise KeyError(
            "Criteo source missing required columns. "
            f"Missing numeric={missing_numeric}, missing categorical={missing_cat}"
        )

    work = df[[label_in_col, *CRITEO_NUMERIC_COLS, *CRITEO_CATEGORICAL_COLS]].copy()
    if label_in_col != out_label_col:
        work.rename(columns={label_in_col: out_label_col}, inplace=True)
    return work[[out_label_col, *CRITEO_NUMERIC_COLS, *CRITEO_CATEGORICAL_COLS]]


def _write_stream_normalized_csv(source_csv: Path, target_csv: Path, ctr_cfg: Dict) -> int:
    """Stream-normalize a Criteo CSV and write it to target CSV."""
    chunk_rows = int(ctr_cfg.get("criteo_prepare_batch_size_rows", 1_000_000) or 1_000_000)
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    row_count = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(source_csv, chunksize=chunk_rows), start=1):
        out = _normalize_criteo_chunk(chunk, ctr_cfg)
        out.to_csv(target_csv, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        row_count += len(out)
        print(f"[INFO] Criteo normalize chunk {chunk_idx}: rows={len(out):,}.")
    return row_count


def _split_single_source_csv(source_csv: Path, train_path: Path, test_path: Path, ctr_cfg: Dict) -> None:
    """Split one Criteo CSV into deterministic train/test CSVs."""
    chunk_rows = int(ctr_cfg.get("criteo_prepare_batch_size_rows", 1_000_000) or 1_000_000)
    test_fraction = _coerce_split_fraction(ctr_cfg.get("criteo_test_fraction", 0.1), default=0.1)
    hash_mod = _coerce_hash_mod(ctr_cfg.get("criteo_hash_mod", 1000), default=1000)
    threshold = int(round(test_fraction * hash_mod))
    threshold = max(1, min(threshold, hash_mod - 1))

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_train_header = False
    wrote_test_header = False
    train_rows = 0
    test_rows = 0
    global_row = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(source_csv, chunksize=chunk_rows), start=1):
        out = _normalize_criteo_chunk(chunk, ctr_cfg)
        idx = np.arange(global_row, global_row + len(out), dtype=np.int64)
        global_row += len(out)
        test_mask = (idx % hash_mod) < threshold
        train_df = out.loc[~test_mask]
        test_df = out.loc[test_mask]

        if not train_df.empty:
            train_df.to_csv(train_path, mode="a", index=False, header=not wrote_train_header)
            wrote_train_header = True
            train_rows += len(train_df)
        if not test_df.empty:
            test_df.to_csv(test_path, mode="a", index=False, header=not wrote_test_header)
            wrote_test_header = True
            test_rows += len(test_df)

        print(
            f"[INFO] Criteo split chunk {chunk_idx}: processed={len(out):,} "
            f"(train={len(train_df):,}, test={len(test_df):,})."
        )

    if train_rows == 0 or test_rows == 0:
        raise RuntimeError(
            f"Criteo split produced an empty split (train_rows={train_rows}, test_rows={test_rows}). "
            "Adjust criteo_test_fraction or criteo_hash_mod."
        )
    print(f"[INFO] Prepared Criteo CSV splits: train={train_rows:,}, test={test_rows:,}.")


def _download_from_hf(
    *,
    repo_id: str,
    subfolder: str,
    filename: str,
    local_dir: Path,
) -> Path:
    """Download one file from Hugging Face datasets hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download reczoo/Criteo_x1 automatically. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        subfolder=subfolder,
        local_dir=str(local_dir),
    )
    return Path(downloaded)


def _find_existing_candidate(base_dir: Path, subfolder: str, filenames: Iterable[str]) -> Optional[Path]:
    """Return the first existing candidate path under local cache roots."""
    roots = [base_dir]
    if subfolder:
        roots.append(base_dir / subfolder)
    for root in roots:
        for name in filenames:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def _prepare_from_hf_split_files(train_path: Path, test_path: Path, ctr_cfg: Dict) -> bool:
    """Prepare train/test CSVs by downloading split files from Hugging Face."""
    if not _as_bool(ctr_cfg.get("download_if_missing", False), default=False):
        return False

    repo_id = str(ctr_cfg.get("hf_repo_id", "reczoo/Criteo_x1"))
    subfolder = str(ctr_cfg.get("hf_subfolder", "") or "")
    hf_train_filename = str(ctr_cfg.get("hf_train_filename", "train.csv"))
    hf_test_filename = str(ctr_cfg.get("hf_test_filename", "test.csv"))
    hf_valid_filename = str(ctr_cfg.get("hf_valid_filename", "valid.csv"))
    use_valid_as_test = _as_bool(ctr_cfg.get("use_valid_as_test", False), default=False)

    local_dir = train_path.parent
    local_dir.mkdir(parents=True, exist_ok=True)

    train_source = _find_existing_candidate(local_dir, subfolder, [hf_train_filename])
    if train_source is None:
        try:
            train_source = _download_from_hf(
                repo_id=repo_id,
                subfolder=subfolder,
                filename=hf_train_filename,
                local_dir=local_dir,
            )
        except Exception:
            train_source = None
    if train_source is None:
        return False

    test_candidates = [hf_test_filename]
    if use_valid_as_test:
        test_candidates.insert(0, hf_valid_filename)
    else:
        test_candidates.append(hf_valid_filename)
    test_source = _find_existing_candidate(local_dir, subfolder, test_candidates)
    if test_source is None:
        for name in test_candidates:
            try:
                test_source = _download_from_hf(
                    repo_id=repo_id,
                    subfolder=subfolder,
                    filename=name,
                    local_dir=local_dir,
                )
                break
            except Exception:
                test_source = None
        if test_source is None:
            return False

    if train_source.resolve() == train_path.resolve():
        # Normalize in-place would overwrite while reading; use temp target first.
        tmp_train = train_path.with_suffix(".tmp.csv")
        if tmp_train.exists():
            tmp_train.unlink()
        _write_stream_normalized_csv(train_source, tmp_train, ctr_cfg)
        shutil.move(str(tmp_train), str(train_path))
    else:
        if train_path.exists():
            train_path.unlink()
        _write_stream_normalized_csv(train_source, train_path, ctr_cfg)

    if test_source.resolve() == test_path.resolve():
        tmp_test = test_path.with_suffix(".tmp.csv")
        if tmp_test.exists():
            tmp_test.unlink()
        _write_stream_normalized_csv(test_source, tmp_test, ctr_cfg)
        shutil.move(str(tmp_test), str(test_path))
    else:
        if test_path.exists():
            test_path.unlink()
        _write_stream_normalized_csv(test_source, test_path, ctr_cfg)

    print(
        f"[INFO] Prepared Criteo CSVs from Hugging Face ({repo_id}): "
        f"train={train_path}, test={test_path}."
    )
    return True


def _find_archive_member(members: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Select one archive member by exact path, suffix path, or basename."""
    member_list = [str(m).replace("\\", "/") for m in members]
    for candidate in candidates:
        if not candidate:
            continue
        cand = str(candidate).replace("\\", "/")
        exact = next((m for m in member_list if m == cand), None)
        if exact is not None:
            return exact
        suffix = "/" + cand
        by_suffix = next((m for m in member_list if m.endswith(suffix)), None)
        if by_suffix is not None:
            return by_suffix
        base = Path(cand).name
        by_base = next((m for m in member_list if Path(m).name == base), None)
        if by_base is not None:
            return by_base
    return None


def _extract_archive_member(zip_path: Path, member_name: str, out_path: Path) -> None:
    """Extract one ZIP member to an output file path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="r") as zf:
        with zf.open(member_name, mode="r") as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _prepare_from_hf_archive(train_path: Path, test_path: Path, ctr_cfg: Dict) -> bool:
    """Prepare train/test CSVs from a downloaded Criteo archive file."""
    if not _as_bool(ctr_cfg.get("download_if_missing", False), default=False):
        return False

    repo_id = str(ctr_cfg.get("hf_repo_id", "reczoo/Criteo_x1"))
    subfolder = str(ctr_cfg.get("hf_subfolder", "") or "")
    hf_archive_filename = str(ctr_cfg.get("hf_archive_filename", "Criteo_x1.zip"))
    hf_train_filename = str(ctr_cfg.get("hf_train_filename", "train.csv"))
    hf_test_filename = str(ctr_cfg.get("hf_test_filename", "test.csv"))
    hf_valid_filename = str(ctr_cfg.get("hf_valid_filename", "valid.csv"))
    use_valid_as_test = _as_bool(ctr_cfg.get("use_valid_as_test", False), default=False)
    train_archive_name = str(ctr_cfg.get("hf_archive_train_filename", hf_train_filename))
    test_archive_name = str(ctr_cfg.get("hf_archive_test_filename", hf_test_filename))
    valid_archive_name = str(ctr_cfg.get("hf_archive_valid_filename", hf_valid_filename))

    local_dir = train_path.parent
    local_dir.mkdir(parents=True, exist_ok=True)

    archive_source = _find_existing_candidate(
        local_dir,
        subfolder,
        [hf_archive_filename, Path(hf_archive_filename).name],
    )
    if archive_source is None:
        try:
            archive_source = _download_from_hf(
                repo_id=repo_id,
                subfolder=subfolder,
                filename=hf_archive_filename,
                local_dir=local_dir,
            )
        except Exception:
            return False
    if archive_source is None or not archive_source.exists():
        return False

    try:
        with zipfile.ZipFile(archive_source, mode="r") as zf:
            members = zf.namelist()
    except Exception:
        return False

    train_member = _find_archive_member(
        members,
        [train_archive_name, hf_train_filename, "train.csv"],
    )
    if use_valid_as_test:
        test_candidates = [
            valid_archive_name,
            hf_valid_filename,
            "valid.csv",
            test_archive_name,
            hf_test_filename,
            "test.csv",
        ]
    else:
        test_candidates = [
            test_archive_name,
            hf_test_filename,
            "test.csv",
            valid_archive_name,
            hf_valid_filename,
            "valid.csv",
        ]
    test_member = _find_archive_member(members, test_candidates)
    if train_member is None or test_member is None:
        return False

    train_raw = local_dir / "__hf_criteo_train_raw.csv"
    test_raw = local_dir / "__hf_criteo_test_raw.csv"
    if train_raw.exists():
        train_raw.unlink()
    if test_raw.exists():
        test_raw.unlink()

    _extract_archive_member(archive_source, train_member, train_raw)
    _extract_archive_member(archive_source, test_member, test_raw)
    try:
        if train_path.exists():
            train_path.unlink()
        if test_path.exists():
            test_path.unlink()
        _write_stream_normalized_csv(train_raw, train_path, ctr_cfg)
        _write_stream_normalized_csv(test_raw, test_path, ctr_cfg)
    finally:
        if train_raw.exists():
            train_raw.unlink()
        if test_raw.exists():
            test_raw.unlink()

    print(
        f"[INFO] Prepared Criteo CSVs from Hugging Face archive ({repo_id}): "
        f"train={train_path}, test={test_path}."
    )
    return True


def maybe_prepare_criteo_dataset(ctr_cfg: Dict) -> None:
    """Ensure Criteo dataset files exist, optionally via HF download/preparation."""
    dataset_name = str(ctr_cfg.get("dataset_name", "")).lower().strip()
    if dataset_name != "criteo":
        return
    if not _as_bool(ctr_cfg.get("auto_prepare", False), default=False):
        return

    train_path = Path(str(ctr_cfg["train_path"]))
    test_path = Path(str(ctr_cfg["test_path"]))
    if train_path.exists() and test_path.exists():
        return

    # 1) Prefer splitting a locally cached source CSV when provided.
    source_csv_path = ctr_cfg.get("source_csv_path")
    if source_csv_path:
        source_csv = Path(str(source_csv_path))
        if source_csv.exists():
            print(
                "[INFO] Preparing Criteo train/test CSVs from local source CSV "
                f"{source_csv} -> train={train_path}, test={test_path}."
            )
            _split_single_source_csv(source_csv, train_path, test_path, ctr_cfg)
            ctr_cfg["test_has_labels"] = True
            return

    # 2) Try using Hugging Face split files (train/test or train/valid fallback).
    prepared = _prepare_from_hf_split_files(train_path, test_path, ctr_cfg)
    if not prepared:
        prepared = _prepare_from_hf_archive(train_path, test_path, ctr_cfg)
    if prepared:
        ctr_cfg["test_has_labels"] = True
        return

    raise FileNotFoundError(
        "Unable to prepare Criteo dataset. Provide existing train/test CSVs, "
        "or set valid HF download settings under ctr.datasets.criteo.* "
        "(split files or archive fallback)."
    )
