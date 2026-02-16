from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class ConditionDataset(Dataset):
    def __init__(
        self,
        x_num,
        conds,
        labels,
        x_cat=None,
        weights=None,
        share_memory: bool = False,
    ):
        """Wrap numeric/categorical features with condition/label targets."""
        assert x_num.ndim == 2 and x_num.shape[1] >= 1
        assert len(x_num) == len(conds) == len(labels)
        self.x = torch.as_tensor(x_num, dtype=torch.float32).contiguous()
        if x_cat is None:
            x_cat = torch.empty((len(labels), 0), dtype=torch.long)
        x_cat = torch.as_tensor(x_cat, dtype=torch.long).contiguous()
        assert x_cat.ndim == 2 and x_cat.shape[0] == len(labels)
        self.x_cat = x_cat
        self.c = torch.as_tensor(conds, dtype=torch.long).contiguous()
        self.y = torch.as_tensor(labels, dtype=torch.float32).contiguous()
        if weights is None:
            weights = torch.ones((len(labels),), dtype=torch.float32)
        self.w = torch.as_tensor(weights, dtype=torch.float32).contiguous()
        assert self.w.ndim == 1 and self.w.shape[0] == len(labels)
        if share_memory:
            for tensor in (self.x, self.x_cat, self.c, self.y, self.w):
                if tensor.device.type == "cpu":
                    tensor.share_memory_()

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.y.numel()

    def __getitem__(self, idx: int):
        """Return a single training example tuple."""
        return (
            self.x[idx],
            self.x_cat[idx],
            self.c[idx],
            self.y[idx],
            self.w[idx],
        )


class TensorBatchLoader:
    def __init__(
        self,
        tensors: Tuple[torch.Tensor, ...],
        batch_size: int,
        shuffle: bool,
        shuffle_seed: Optional[int] = None,
        shuffle_on_cpu: bool = False,
    ):
        """Batch tensors already staged on a device."""
        assert len(tensors) > 0
        n = tensors[0].shape[0]
        for t in tensors[1:]:
            assert t.shape[0] == n, "All tensors must share the first dimension."
        self.tensors = tensors
        self.batch_size = int(max(1, batch_size))
        self.shuffle = shuffle
        self.length = n
        self.device = tensors[0].device
        self.shuffle_on_cpu = bool(shuffle_on_cpu)
        self._cpu_generator: Optional[torch.Generator] = None
        if shuffle_seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(shuffle_seed))
            self._cpu_generator = gen

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return (self.length + self.batch_size - 1) // self.batch_size

    @property
    def num_workers(self) -> int:
        """Expose a DataLoader-compatible num_workers value."""
        return 0

    def __iter__(self):
        """Yield batches of tensors by index selection."""
        # Build indices on the same device as tensors to avoid host/device sync.
        indices = torch.arange(self.length, device=self.device, dtype=torch.long)
        if self.shuffle:
            # For cross-backend reproducibility, optionally build permutation on CPU.
            use_cpu_perm = self.shuffle_on_cpu or (self._cpu_generator is not None)
            if use_cpu_perm:
                perm_cpu = torch.randperm(
                    self.length,
                    generator=self._cpu_generator,
                    device="cpu",
                )
                if self.device.type == "cpu":
                    indices = perm_cpu
                else:
                    indices = perm_cpu.to(
                        self.device,
                        non_blocking=(self.device.type == "cuda"),
                    )
            else:
                indices = indices[torch.randperm(self.length, device=self.device)]
        for start in range(0, self.length, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield tuple(t.index_select(0, batch_idx) for t in self.tensors)
