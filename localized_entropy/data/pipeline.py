from dataclasses import dataclass
from typing import Dict, Optional

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from localized_entropy.data.common import standardize_features, train_eval_split
from localized_entropy.data.ctr import build_ctr_arrays, load_ctr_frames
from localized_entropy.data.datasets import ConditionDataset, TensorBatchLoader
from localized_entropy.data.synthetic import build_features, make_dataset
from localized_entropy.utils import is_notebook


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    x_eval: np.ndarray
    x_test: Optional[np.ndarray]
    y_train: np.ndarray
    y_eval: np.ndarray
    y_test: Optional[np.ndarray]
    c_train: np.ndarray
    c_eval: np.ndarray
    c_test: Optional[np.ndarray]
    nw_train: np.ndarray
    nw_eval: np.ndarray
    nw_test: Optional[np.ndarray]
    p_train: Optional[np.ndarray]
    p_eval: Optional[np.ndarray]
    feature_names: list
    num_conditions: int


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
    kwargs = dict(loader_common)
    if worker_kwargs:
        kwargs.update(worker_kwargs)
    kwargs["shuffle"] = shuffle
    return DataLoader(dataset, **kwargs)


def _build_loader_with_fallback(dataset: Dataset, *, shuffle: bool, role: str, loader_common: Dict, worker_kwargs: Dict) -> DataLoader:
    if not worker_kwargs:
        return _instantiate_loader(dataset, shuffle=shuffle, loader_common=loader_common, worker_kwargs={})
    test_iter = None
    try:
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


def build_dataloaders(
    splits: DatasetSplits,
    cfg: Dict,
    device: torch.device,
    use_cuda: bool,
) -> LoaderBundle:
    batch_size = int(cfg["training"]["batch_size"])
    move_dataset_to_cuda = bool(cfg["device"]["move_dataset_to_cuda"])
    allow_dataloader_workers = cfg["device"].get("allow_dataloader_workers")
    env_var = cfg["device"].get("num_workers_env", "LOCALIZED_ENTROPY_NUM_WORKERS")

    use_tensor_loader = use_cuda and move_dataset_to_cuda
    if use_tensor_loader:
        print("Staging datasets directly on CUDA for batch sampling.")
        train_tensors = (
            torch.as_tensor(splits.x_train, dtype=torch.float32, device=device),
            torch.as_tensor(splits.c_train, dtype=torch.long, device=device),
            torch.as_tensor(splits.y_train, dtype=torch.float32, device=device),
            torch.as_tensor(splits.nw_train, dtype=torch.float32, device=device),
        )
        eval_tensors = (
            torch.as_tensor(splits.x_eval, dtype=torch.float32, device=device),
            torch.as_tensor(splits.c_eval, dtype=torch.long, device=device),
            torch.as_tensor(splits.y_eval, dtype=torch.float32, device=device),
            torch.as_tensor(splits.nw_eval, dtype=torch.float32, device=device),
        )
        train_loader = TensorBatchLoader(train_tensors, batch_size=batch_size, shuffle=True)
        eval_loader = TensorBatchLoader(eval_tensors, batch_size=batch_size, shuffle=False)
        test_loader = None
        if splits.x_test is not None:
            test_tensors = (
                torch.as_tensor(splits.x_test, dtype=torch.float32, device=device),
                torch.as_tensor(splits.c_test, dtype=torch.long, device=device),
                torch.as_tensor(splits.y_test, dtype=torch.float32, device=device),
                torch.as_tensor(splits.nw_test, dtype=torch.float32, device=device),
            )
            test_loader = TensorBatchLoader(test_tensors, batch_size=batch_size, shuffle=False)
        loader_note = (
            f"TensorBatchLoader on CUDA (batches per epoch: {len(train_loader)} / {len(eval_loader)})."
        )
        if test_loader is not None:
            loader_note += f" | Test batches: {len(test_loader)}"
        return LoaderBundle(train_loader, eval_loader, test_loader, loader_note)

    train_ds = ConditionDataset(splits.x_train, splits.c_train, splits.y_train, net_worth=splits.nw_train)
    eval_ds = ConditionDataset(splits.x_eval, splits.c_eval, splits.y_eval, net_worth=splits.nw_eval)
    test_ds = None
    if splits.x_test is not None:
        test_ds = ConditionDataset(splits.x_test, splits.c_test, splits.y_test, net_worth=splits.nw_test)

    loader_common = dict(batch_size=batch_size, drop_last=False, pin_memory=use_cuda)
    max_workers = os.cpu_count() or 1
    env_override = os.environ.get(env_var)
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


def prepare_data(cfg: Dict, device: torch.device, use_cuda: bool) -> PreparedData:
    seed = int(cfg["project"]["seed"])
    data_cfg = cfg["data"]
    source = data_cfg["source"].lower().strip()
    train_ratio = float(data_cfg.get("train_split", 0.9))
    std_eps = float(data_cfg.get("standardize_eps", 1e-6))
    standardize = bool(data_cfg.get("standardize", True))

    plot_data = {}
    if source == "ctr":
        train_df, test_df, stats_df, top_values = load_ctr_frames(cfg["ctr"])
        arrays = build_ctr_arrays(train_df, test_df, cfg["ctr"])
        xnum = arrays["xnum"]
        labels = arrays["labels"]
        conds = arrays["conds"]
        net_worth = arrays["net_worth"]
        probs = arrays["probs"]
        xnum_test = arrays["xnum_test"]
        conds_test = arrays["conds_test"]
        net_worth_test = arrays["net_worth_test"]
        feature_names = arrays["feature_names"]
        num_conditions = arrays["num_conditions"]
        if stats_df is not None:
            plot_data["ctr_stats"] = {
                "stats_df": stats_df,
                "labels": [str(v) for v in stats_df.index.to_list()],
                "filter_col": cfg["ctr"].get("filter_col"),
            }
        plot_sample_size = int(cfg["ctr"].get("plot_sample_size", 0) or 0)
        if plot_sample_size != 0:
            rng = np.random.default_rng(seed)
            n_plot = min(plot_sample_size, len(labels))
            idx = rng.choice(len(labels), size=n_plot, replace=False)
            plot_data["ctr_distributions"] = {
                "xnum": xnum[idx],
                "conds": conds[idx],
                "labels": labels[idx],
                "feature_names": feature_names,
                "num_conditions": num_conditions,
            }
    elif source == "synthetic":
        dataset = make_dataset(cfg["synthetic"], seed=seed)
        feature_payload = build_features(dataset, cfg["synthetic"])
        xnum = feature_payload["xnum"]
        feature_names = feature_payload["feature_names"]
        labels = dataset["labels"]
        conds = dataset["conds"]
        net_worth = dataset["net_worth"]
        probs = dataset["probs"]
        xnum_test = None
        conds_test = None
        net_worth_test = None
        num_conditions = dataset["num_conditions"]
        plot_data["synthetic"] = {
            "net_worth": net_worth,
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
    c_train = conds[train_idx]
    c_eval = conds[eval_idx]
    y_train = labels[train_idx]
    y_eval = labels[eval_idx]
    nw_train = net_worth[train_idx]
    nw_eval = net_worth[eval_idx]
    p_train = probs[train_idx] if probs is not None else None
    p_eval = probs[eval_idx] if probs is not None else None

    y_test = None
    if xnum_test is not None:
        y_test = np.zeros((len(xnum_test),), dtype=np.float32)

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
        y_train=y_train,
        y_eval=y_eval,
        y_test=y_test,
        c_train=c_train,
        c_eval=c_eval,
        c_test=conds_test,
        nw_train=nw_train,
        nw_eval=nw_eval,
        nw_test=net_worth_test,
        p_train=p_train,
        p_eval=p_eval,
        feature_names=feature_names,
        num_conditions=num_conditions,
    )

    loaders = build_dataloaders(splits, cfg, device, use_cuda)
    return PreparedData(splits=splits, loaders=loaders, normalizer=normalizer, plot_data=plot_data)
