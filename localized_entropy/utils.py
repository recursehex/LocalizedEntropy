import random
from typing import Iterable, List

import numpy as np
import torch


def set_seed(seed: int, use_cuda: bool, deterministic: bool = False) -> None:
    """Seed RNGs and optionally force deterministic PyTorch behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except TypeError:
        # Older torch versions may not accept warn_only.
        torch.use_deterministic_algorithms(bool(deterministic))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic and hasattr(torch.backends, "cuda"):
        matmul_backend = getattr(torch.backends.cuda, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = False
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False


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


def init_device(verbose: bool = True, use_mps: bool = True, deterministic: bool = False):
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
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = not bool(deterministic)
            if deterministic:
                torch.backends.cudnn.deterministic = True
    elif use_mps:
        if verbose:
            print("Using MPS device (Apple Silicon GPU).")
    else:
        if verbose:
            print("No GPU backend available; defaulting to CPU.")
    return device, use_cuda, use_mps, non_blocking
