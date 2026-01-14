from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn


_LayerValue = Union[str, float, int, None]
_LayerSpec = Union[_LayerValue, Sequence[_LayerValue]]


def _expand_per_layer(value: _LayerSpec, num_layers: int, name: str) -> List[_LayerValue]:
    if num_layers < 0:
        raise ValueError(f"{name} requires a non-negative layer count.")
    if num_layers == 0:
        if isinstance(value, (list, tuple)) and len(value) != 0:
            raise ValueError(f"{name} has {len(value)} entries but hidden_sizes is empty.")
        return []
    if value is None:
        return [None] * num_layers
    if isinstance(value, (list, tuple)):
        if len(value) != num_layers:
            raise ValueError(
                f"{name} has {len(value)} entries but hidden_sizes has {num_layers} layers."
            )
        return list(value)
    return [value] * num_layers


def _resolve_activation(name: Optional[str]) -> Optional[nn.Module]:
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in {"", "none", "identity", "linear"}:
        return None
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key == "tanh":
        return nn.Tanh()
    if key in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(negative_slope=0.01)
    if key == "elu":
        return nn.ELU()
    if key == "selu":
        return nn.SELU()
    if key == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation '{name}'.")


def _resolve_norm(name: Optional[str], dim: int) -> Optional[nn.Module]:
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in {"", "none", "identity"}:
        return None
    if key in {"batch_norm", "batchnorm", "batch_norm1d", "bn"}:
        return nn.BatchNorm1d(dim)
    if key in {"layer_norm", "layernorm", "ln"}:
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported norm '{name}'.")


def _resolve_dropout(value: _LayerValue) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        if value.strip().lower() in {"", "none", "null"}:
            return 0.0
    prob = float(value)
    if prob <= 0.0:
        return 0.0
    if prob >= 1.0:
        raise ValueError("Dropout probability must be in [0, 1).")
    return prob


class ConditionProbNet(nn.Module):
    def __init__(
        self,
        num_conditions: int,
        num_numeric: int = 2,
        embed_dim: int = 16,
        cat_dims: Optional[Sequence[int]] = None,
        cat_embed_dim: Optional[int] = None,
        hidden_sizes: Optional[Tuple[int, ...]] = None,
        p_drop: _LayerSpec = 0.3,
        activation: _LayerSpec = "relu",
        norm: _LayerSpec = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = (256, 256, 128, 64)
        if cat_dims is None:
            cat_dims = []
        if cat_embed_dim is None:
            cat_embed_dim = embed_dim
        self.embedding = nn.Embedding(num_conditions, embed_dim)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(int(dim), int(cat_embed_dim)) for dim in cat_dims]
        )
        layers: List[nn.Module] = []
        in_dim = embed_dim + num_numeric + (int(cat_embed_dim) * len(cat_dims))
        if isinstance(hidden_sizes, (int, float)):
            hidden_sizes = (int(hidden_sizes),)
        else:
            hidden_sizes = tuple(int(v) for v in hidden_sizes)
        activations = _expand_per_layer(activation, len(hidden_sizes), "activation")
        norms = _expand_per_layer(norm, len(hidden_sizes), "norm")
        dropouts = _expand_per_layer(p_drop, len(hidden_sizes), "dropout")
        for idx, hidden in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, hidden))
            norm_layer = _resolve_norm(norms[idx], hidden)
            if norm_layer is not None:
                layers.append(norm_layer)
            act_layer = _resolve_activation(activations[idx])
            if act_layer is not None:
                layers.append(act_layer)
            drop_prob = _resolve_dropout(dropouts[idx])
            if drop_prob > 0.0:
                layers.append(nn.Dropout(p=drop_prob))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(cond)
        parts = [x_num, emb]
        if self.cat_embeddings:
            cat_parts = [
                emb_layer(x_cat[:, idx])
                for idx, emb_layer in enumerate(self.cat_embeddings)
            ]
            parts.append(torch.cat(cat_parts, dim=-1))
        x = torch.cat(parts, dim=-1)
        logits = self.net(x).squeeze(-1)
        return logits
