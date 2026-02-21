from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from localized_entropy.data.common import standardize_features, train_eval_split
from localized_entropy.data.ctr import build_ctr_arrays, load_ctr_frames, maybe_cache_filtered_ctr
from localized_entropy.data.datasets import ConditionDataset, TensorBatchLoader
from localized_entropy.data.synthetic import build_features, compute_negative_reweighting, make_dataset
from localized_entropy.utils import is_notebook


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    x_eval: np.ndarray
    x_test: Optional[np.ndarray]
    x_cat_train: np.ndarray
    x_cat_eval: np.ndarray
    x_cat_test: Optional[np.ndarray]
    y_train: np.ndarray
    y_eval: np.ndarray
    y_test: Optional[np.ndarray]
    c_train: np.ndarray
    c_eval: np.ndarray
    c_test: Optional[np.ndarray]
    p_train: Optional[np.ndarray]
    p_eval: Optional[np.ndarray]
    w_train: np.ndarray
    w_eval: np.ndarray
    w_test: Optional[np.ndarray]
    feature_names: list
    num_conditions: int
    cat_sizes: list
    cat_cols: list


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    eval_loader: DataLoader
    test_loader: Optional[DataLoader]
    loader_note: str


@dataclass
class PreparedData:
    splits: DatasetSplits
    loaders: LoaderBundle
    normalizer: Dict[str, np.ndarray]
    plot_data: Dict


def _instantiate_loader(dataset: Dataset, *, shuffle: bool, loader_common: Dict, worker_kwargs: Dict) -> DataLoader:
    """Instantiate a PyTorch DataLoader with shared options."""
    kwargs = dict(loader_common)
    if worker_kwargs:
        kwargs.update(worker_kwargs)
    kwargs["shuffle"] = shuffle
    return DataLoader(dataset, **kwargs)


def _build_loader_with_fallback(dataset: Dataset, *, shuffle: bool, role: str, loader_common: Dict, worker_kwargs: Dict) -> DataLoader:
    """Build a DataLoader and fall back to num_workers=0 on failure."""
    if not worker_kwargs:
        return _instantiate_loader(dataset, shuffle=shuffle, loader_common=loader_common, worker_kwargs={})
    test_iter = None
    try:
        # Probe a worker-backed loader to catch multiprocessing issues early.
        test_loader = _instantiate_loader(dataset, shuffle=shuffle, loader_common=loader_common, worker_kwargs=worker_kwargs)
        test_iter = iter(test_loader)
        next(test_iter)
    except Exception as exc:
        if test_iter is not None and hasattr(test_iter, "_shutdown_workers"):
            test_iter._shutdown_workers()
        print(f"[WARN] {role} DataLoader workers failed ({exc}); falling back to workers=0.")
        return _instantiate_loader(dataset, shuffle=shuffle, loader_common=loader_common, worker_kwargs={})
    else:
        if test_iter is not None and hasattr(test_iter, "_shutdown_workers"):
            test_iter._shutdown_workers()
        return _instantiate_loader(dataset, shuffle=shuffle, loader_common=loader_common, worker_kwargs=worker_kwargs)


def _apply_indices(arr: Optional[np.ndarray], idx: np.ndarray) -> Optional[np.ndarray]:
    """Apply a shared index array to an optional ndarray."""
    if arr is None:
        return None
    return arr[idx]


def _balance_indices_by_condition(
    conds: np.ndarray,
    num_conditions: int,
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], int, np.ndarray]:
    """Downsample each condition to the minimum non-zero count."""
    c = np.asarray(conds, dtype=np.int64).reshape(-1)
    counts = np.bincount(c, minlength=int(num_conditions))
    nonzero_counts = counts[counts > 0]
    if nonzero_counts.size == 0:
        return None, 0, counts
    min_count = int(nonzero_counts.min())
    keep_idx = []
    for cond_id in range(int(num_conditions)):
        cond_idx = np.flatnonzero(c == cond_id)
        if cond_idx.size == 0:
            continue
        # Downsample each condition to the smallest non-zero count.
        if cond_idx.size > min_count:
            cond_idx = rng.choice(cond_idx, size=min_count, replace=False)
        keep_idx.append(cond_idx)
    if not keep_idx:
        return None, min_count, counts
    keep_idx = np.concatenate(keep_idx)
    keep_idx = rng.permutation(keep_idx)
    return keep_idx, min_count, counts


def build_dataloaders(
    splits: DatasetSplits,
    cfg: Dict,
    device: torch.device,
    use_cuda: bool,
    use_mps: bool = False,
    batch_size: Optional[int] = None,
) -> LoaderBundle:
    """Construct train/eval/test loaders from prepared splits."""
    if batch_size is None:
        batch_size = int(cfg["training"]["batch_size"])
    else:
        batch_size = int(batch_size)
    move_dataset_to_cuda = bool(cfg["device"]["move_dataset_to_cuda"])
    tensor_loader_shuffle_on_cpu = bool(cfg["device"].get("tensor_loader_shuffle_on_cpu", True))
    tensor_loader_deterministic_shuffle = bool(
        cfg["device"].get("tensor_loader_deterministic_shuffle", True)
    )
    tensor_loader_shuffle_seed = None
    if tensor_loader_deterministic_shuffle:
        tensor_loader_shuffle_seed = int(cfg.get("project", {}).get("seed", 0))
    allow_dataloader_workers = cfg["device"].get("allow_dataloader_workers")
    env_var = cfg["device"].get("num_workers_env", "LOCALIZED_ENTROPY_NUM_WORKERS")

    use_accelerator = use_cuda or use_mps
    device_label = "CUDA" if use_cuda else "MPS" if use_mps else "CPU"
    use_tensor_loader = use_accelerator and move_dataset_to_cuda
    if use_tensor_loader:
        # Stage datasets on the GPU to avoid per-batch host-to-device copies.
        print(f"Staging datasets directly on {device_label} for batch sampling.")
        train_tensors = (
            torch.as_tensor(splits.x_train, dtype=torch.float32, device=device),
            torch.as_tensor(splits.x_cat_train, dtype=torch.long, device=device),
            torch.as_tensor(splits.c_train, dtype=torch.long, device=device),
            torch.as_tensor(splits.y_train, dtype=torch.float32, device=device),
            torch.as_tensor(splits.w_train, dtype=torch.float32, device=device),
        )
        eval_tensors = (
            torch.as_tensor(splits.x_eval, dtype=torch.float32, device=device),
            torch.as_tensor(splits.x_cat_eval, dtype=torch.long, device=device),
            torch.as_tensor(splits.c_eval, dtype=torch.long, device=device),
            torch.as_tensor(splits.y_eval, dtype=torch.float32, device=device),
            torch.as_tensor(splits.w_eval, dtype=torch.float32, device=device),
        )
        train_loader = TensorBatchLoader(
            train_tensors,
            batch_size=batch_size,
            shuffle=True,
            shuffle_seed=tensor_loader_shuffle_seed,
            shuffle_on_cpu=tensor_loader_shuffle_on_cpu,
        )
        eval_loader = TensorBatchLoader(eval_tensors, batch_size=batch_size, shuffle=False)
        test_loader = None
        if splits.x_test is not None:
            test_tensors = (
                torch.as_tensor(splits.x_test, dtype=torch.float32, device=device),
                torch.as_tensor(splits.x_cat_test, dtype=torch.long, device=device),
                torch.as_tensor(splits.c_test, dtype=torch.long, device=device),
                torch.as_tensor(splits.y_test, dtype=torch.float32, device=device),
                torch.as_tensor(splits.w_test, dtype=torch.float32, device=device),
            )
            test_loader = TensorBatchLoader(test_tensors, batch_size=batch_size, shuffle=False)
        loader_note = (
            f"TensorBatchLoader on {device_label} (batches per epoch: {len(train_loader)} / {len(eval_loader)})."
        )
        if tensor_loader_deterministic_shuffle:
            loader_note += (
                " | deterministic shuffle enabled"
                f" (seed={tensor_loader_shuffle_seed}, cpu_perm={tensor_loader_shuffle_on_cpu})"
            )
        if test_loader is not None:
            loader_note += f" | Test batches: {len(test_loader)}"
        return LoaderBundle(train_loader, eval_loader, test_loader, loader_note)

    train_ds = ConditionDataset(
        splits.x_train,
        splits.c_train,
        splits.y_train,
        x_cat=splits.x_cat_train,
        weights=splits.w_train,
    )
    eval_ds = ConditionDataset(
        splits.x_eval,
        splits.c_eval,
        splits.y_eval,
        x_cat=splits.x_cat_eval,
        weights=splits.w_eval,
    )
    test_ds = None
    if splits.x_test is not None:
        test_ds = ConditionDataset(
            splits.x_test,
            splits.c_test,
            splits.y_test,
            x_cat=splits.x_cat_test,
            weights=splits.w_test,
        )

    loader_common = dict(batch_size=batch_size, drop_last=False, pin_memory=use_cuda)
    max_workers = os.cpu_count() or 1
    env_override = os.environ.get(env_var)
    # Decide on worker count from config/env, defaulting to single-process in notebooks.
    if allow_dataloader_workers is None:
        allow_workers = not is_notebook()
    else:
        allow_workers = bool(allow_dataloader_workers)
    if env_override is not None and (allow_dataloader_workers is not False):
        allow_workers = True

    if not allow_workers:
        num_workers = 0
    elif env_override is not None:
        try:
            num_workers = max(0, min(int(env_override), max_workers))
        except ValueError:
            num_workers = 0
    else:
        num_workers = 0 if use_cuda else min(2, max_workers)

    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs = dict(num_workers=num_workers, persistent_workers=False, prefetch_factor=2)

    train_loader = _build_loader_with_fallback(
        train_ds, shuffle=True, role="Train", loader_common=loader_common, worker_kwargs=worker_kwargs
    )
    eval_loader = _build_loader_with_fallback(
        eval_ds, shuffle=False, role="Eval", loader_common=loader_common, worker_kwargs=worker_kwargs
    )
    test_loader = (
        _build_loader_with_fallback(test_ds, shuffle=False, role="Test", loader_common=loader_common, worker_kwargs=worker_kwargs)
        if test_ds is not None else None
    )

    train_workers = getattr(train_loader, "num_workers", 0)
    eval_workers = getattr(eval_loader, "num_workers", 0)
    loader_note = (
        f"Train/Eval DataLoader workers: {train_workers}/{eval_workers} (pin_memory={loader_common.get('pin_memory', False)})"
    )
    if test_loader is not None:
        test_workers = getattr(test_loader, "num_workers", 0)
        loader_note += f" | Test workers: {test_workers}"
    if use_cuda and train_workers == 0:
        loader_note += " | Multiprocessing disabled for CUDA stability; set LOCALIZED_ENTROPY_NUM_WORKERS>0 to retry."
    if (not allow_workers) and (train_workers == 0):
        loader_note += " | DataLoader workers disabled (set allow_dataloader_workers=true to enable)."

    return LoaderBundle(train_loader, eval_loader, test_loader, loader_note)


def _use_cpu_permutation_for_le(cfg: Dict, source: str) -> bool:
    """Return True when LE cross-batch should force CPU permutations."""
    train_cfg = cfg.get("training", {})
    by_loss = train_cfg.get("by_loss", {}) if isinstance(train_cfg, dict) else {}
    le_cfg = by_loss.get("localized_entropy", {}) if isinstance(by_loss, dict) else {}
    if not isinstance(le_cfg, dict):
        return False
    source_cfg = le_cfg.get("by_source", {}) if isinstance(le_cfg.get("by_source", {}), dict) else {}
    source_le = source_cfg.get(source, {}) if isinstance(source_cfg, dict) else {}
    if not isinstance(source_le, dict):
        return False
    cross_batch = source_le.get("cross_batch")
    if not isinstance(cross_batch, dict):
        return False
    return bool(cross_batch.get("enabled", False))


def build_dataloaders_for_loss(
    splits: DatasetSplits,
    cfg: Dict,
    device: torch.device,
    use_cuda: bool,
    use_mps: bool = False,
    *,
    loss_mode: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> LoaderBundle:
    """Build dataloaders with optional loss-specific loader overrides."""
    cfg_local = cfg
    source = str(cfg.get("data", {}).get("source", "synthetic")).lower().strip()
    if str(loss_mode).lower().strip() == "localized_entropy":
        if _use_cpu_permutation_for_le(cfg, source):
            cfg_local = dict(cfg)
            device_cfg = dict(cfg.get("device", {}))
            # LE cross-batch is highly order-sensitive; use CPU permutation for
            # stable cross-backend behavior when enabled.
            device_cfg["tensor_loader_shuffle_on_cpu"] = True
            cfg_local["device"] = device_cfg
    return build_dataloaders(
        splits,
        cfg_local,
        device,
        use_cuda,
        use_mps,
        batch_size=batch_size,
    )


def prepare_data(cfg: Dict, device: torch.device, use_cuda: bool, use_mps: bool = False) -> PreparedData:
    """Load/prepare data arrays and return splits, loaders, and plot data."""
    seed = int(cfg["project"]["seed"])
    data_cfg = cfg["data"]
    source = data_cfg["source"].lower().strip()
    train_ratio = float(data_cfg.get("train_split", 0.9))
    std_eps = float(data_cfg.get("standardize_eps", 1e-6))
    standardize = bool(data_cfg.get("standardize", True))

    plot_data = {}
    labels_test = None
    plot_sample_size = 0
    balance_by_condition = False
    if source == "ctr":
        maybe_cache_filtered_ctr(cfg["ctr"])
        train_df, test_df, stats_df, top_values = load_ctr_frames(cfg["ctr"])
        arrays = build_ctr_arrays(train_df, test_df, cfg["ctr"])
        xnum = arrays["xnum"]
        xcat = arrays["xcat"]
        labels = arrays["labels"]
        conds = arrays["conds"]
        probs = arrays["probs"]
        xnum_test = arrays["xnum_test"]
        xcat_test = arrays["xcat_test"]
        conds_test = arrays["conds_test"]
        labels_test = arrays.get("labels_test")
        feature_names = arrays["feature_names"]
        num_conditions = arrays["num_conditions"]
        cat_sizes = arrays["cat_sizes"]
        cat_cols = arrays["cat_cols"]
        balance_by_condition = bool(cfg.get("ctr", {}).get("balance_by_condition", False))
        plot_sample_size = int(cfg["ctr"].get("plot_sample_size", 0) or 0)
        if stats_df is not None:
            filter_label = (
                cfg.get("ctr", {}).get("filter", {}).get("col")
                or cfg.get("ctr", {}).get("filter_col")
                or cfg.get("ctr", {}).get("condition_col")
            )
            plot_data["ctr_stats"] = {
                "stats_df": stats_df,
                "labels": [str(v) for v in stats_df.index.to_list()],
                "filter_col": filter_label,
            }
    elif source == "synthetic":
        dataset = make_dataset(cfg["synthetic"], seed=seed)
        feature_payload = build_features(dataset, cfg["synthetic"])
        xnum = feature_payload["xnum"]
        feature_names = feature_payload["feature_names"]
        labels = dataset["labels"]
        conds = dataset["conds"]
        probs = dataset["probs"]
        xnum_test = None
        xcat = np.empty((len(labels), 0), dtype=np.int64)
        xcat_test = None
        conds_test = None
        num_conditions = dataset["num_conditions"]
        cat_sizes = []
        cat_cols = []
        plot_data["synthetic"] = {
            "net_worth": dataset["net_worth"],
            "ages": dataset["ages"],
            "probs": probs,
            "conds": conds,
            "num_conditions": num_conditions,
        }
    else:
        raise ValueError(f"Unsupported data source: {source}")

    n_total = len(labels)
    train_idx, eval_idx = train_eval_split(n_total, train_ratio, seed)

    x_train = xnum[train_idx]
    x_eval = xnum[eval_idx]
    x_cat_train = xcat[train_idx]
    x_cat_eval = xcat[eval_idx]
    c_train = conds[train_idx]
    c_eval = conds[eval_idx]
    y_train = labels[train_idx]
    y_eval = labels[eval_idx]
    p_train = probs[train_idx] if probs is not None else None
    p_eval = probs[eval_idx] if probs is not None else None

    if balance_by_condition:
        # Optionally downsample to equalize per-condition representation in training.
        balance_rng = np.random.default_rng(seed)
        keep_idx, min_count, counts = _balance_indices_by_condition(
            c_train,
            num_conditions,
            balance_rng,
        )
        if keep_idx is None:
            print("[WARN] balance_by_condition is enabled but no training samples are available; skipping.")
        else:
            before = len(c_train)
            x_train = x_train[keep_idx]
            x_cat_train = x_cat_train[keep_idx]
            c_train = c_train[keep_idx]
            y_train = y_train[keep_idx]
            p_train = _apply_indices(p_train, keep_idx)
            active_conditions = int((counts > 0).sum())
            print(
                "Balanced training data by condition: "
                f"min_count={min_count:,} across {active_conditions} conditions; "
                f"kept {len(c_train):,} of {before:,} rows."
            )

    w_train = np.ones((len(y_train),), dtype=np.float32)
    w_eval = np.ones((len(y_eval),), dtype=np.float32)
    if source == "synthetic":
        reweight_cfg = cfg.get("synthetic", {}).get("reweighting") or {}
        if bool(reweight_cfg.get("enabled", False)):
            reweight_rng = np.random.default_rng(seed)
            keep_idx, weights_kept, stats = compute_negative_reweighting(
                y_train,
                c_train,
                cfg["synthetic"],
                probs=p_train,
                rng=reweight_rng,
            )
            before = len(y_train)
            x_train = x_train[keep_idx]
            x_cat_train = x_cat_train[keep_idx]
            c_train = c_train[keep_idx]
            y_train = y_train[keep_idx]
            p_train = _apply_indices(p_train, keep_idx)
            w_train = weights_kept.astype(np.float32, copy=False)
            removed = before - len(y_train)
            mode = stats.get("mode", "unknown")
            print(
                "Synthetic reweighting enabled (DEPRECATED): "
                f"mode={mode} removed={removed:,} negatives; "
                f"kept={len(y_train):,} of {before:,} training rows."
            )

    if source == "ctr" and plot_sample_size != 0:
        if len(y_train) == 0:
            print("[WARN] CTR plot sample requested but training data is empty.")
        else:
            rng = np.random.default_rng(seed)
            n_plot = min(plot_sample_size, len(y_train))
            idx = rng.choice(len(y_train), size=n_plot, replace=False)
            plot_data["ctr_distributions"] = {
                "xnum": x_train[idx],
                "conds": c_train[idx],
                "labels": y_train[idx],
                "feature_names": feature_names,
                "num_conditions": num_conditions,
            }

    y_test = None
    if xnum_test is not None:
        if labels_test is not None:
            y_test = labels_test
        else:
            y_test = np.zeros((len(xnum_test),), dtype=np.float32)
    else:
        xcat_test = None
    w_test = np.ones((len(y_test),), dtype=np.float32) if y_test is not None else None

    shuffle_test = bool(data_cfg.get("shuffle_test", True))
    if shuffle_test and xnum_test is not None:
        rng = np.random.default_rng(seed)
        test_idx = rng.permutation(len(xnum_test))
        xnum_test = xnum_test[test_idx]
        if xcat_test is not None:
            xcat_test = xcat_test[test_idx]
        if conds_test is not None:
            conds_test = conds_test[test_idx]
        if y_test is not None:
            y_test = y_test[test_idx]
        if w_test is not None:
            w_test = w_test[test_idx]

    if standardize:
        x_train_n, x_eval_n, x_test_n, normalizer = standardize_features(
            x_train, x_eval, xnum_test, eps=std_eps
        )
    else:
        x_train_n, x_eval_n, x_test_n = x_train, x_eval, xnum_test
        normalizer = {"mean": None, "std": None}

    splits = DatasetSplits(
        x_train=x_train_n,
        x_eval=x_eval_n,
        x_test=x_test_n,
        x_cat_train=x_cat_train,
        x_cat_eval=x_cat_eval,
        x_cat_test=xcat_test,
        y_train=y_train,
        y_eval=y_eval,
        y_test=y_test,
        c_train=c_train,
        c_eval=c_eval,
        c_test=conds_test,
        p_train=p_train,
        p_eval=p_eval,
        w_train=w_train,
        w_eval=w_eval,
        w_test=w_test,
        feature_names=feature_names,
        num_conditions=num_conditions,
        cat_sizes=cat_sizes,
        cat_cols=cat_cols,
    )

    loaders = build_dataloaders(splits, cfg, device, use_cuda, use_mps)
    return PreparedData(splits=splits, loaders=loaders, normalizer=normalizer, plot_data=plot_data)
