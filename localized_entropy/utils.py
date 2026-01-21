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


def init_device(verbose: bool = True):
    """Initialize CUDA/CPU device and non_blocking flag."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    non_blocking = use_cuda
    if use_cuda:
        if verbose:
            gpu_name = torch.cuda.get_device_name(device)
            print(f"Using CUDA device: {gpu_name}")
        torch.backends.cudnn.benchmark = True
    else:
        if verbose:
            print("CUDA not available, defaulting to CPU.")
    return device, use_cuda, non_blocking
