from typing import Iterable, List

import numpy as np
import torch


def set_seed(seed: int, use_cuda: bool) -> None:
    """Seed NumPy and PyTorch RNGs (including CUDA when enabled)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def dedupe(seq: Iterable) -> List:
    """Return a list with duplicates removed while preserving order."""
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def is_notebook() -> bool:
    """Return True when running inside a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def init_device(verbose: bool = True, use_mps: bool = True):
    """Initialize CUDA/MPS/CPU device selection and non_blocking flag."""
    use_cuda = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    use_mps = bool(use_mps) and mps_available
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    non_blocking = device.type == "cuda"
    if use_cuda:
        if verbose:
            gpu_name = torch.cuda.get_device_name(device)
            print(f"Using CUDA device: {gpu_name}")
        torch.backends.cudnn.benchmark = True
    elif use_mps:
        if verbose:
            print("Using MPS device (Apple Silicon GPU).")
    else:
        if verbose:
            print("No GPU backend available; defaulting to CPU.")
    return device, use_cuda, use_mps, non_blocking
