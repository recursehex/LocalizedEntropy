from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn


class ConditionProbNet(nn.Module):
    def __init__(
        self,
        num_conditions: int,
        num_numeric: int = 2,
        embed_dim: int = 16,
        cat_dims: Optional[Sequence[int]] = None,
        cat_embed_dim: Optional[int] = None,
        hidden_sizes: Optional[Tuple[int, ...]] = None,
        p_drop: float = 0.3,
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
        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(p=p_drop),
            ])
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
