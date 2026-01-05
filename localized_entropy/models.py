from typing import List, Optional, Tuple

import torch
from torch import nn


class ConditionProbNet(nn.Module):
    def __init__(
        self,
        num_conditions: int,
        num_numeric: int = 2,
        embed_dim: int = 16,
        hidden_sizes: Optional[Tuple[int, ...]] = None,
        p_drop: float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = (256, 256, 128, 64)
        self.embedding = nn.Embedding(num_conditions, embed_dim)
        layers: List[nn.Module] = []
        in_dim = embed_dim + num_numeric
        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(p=p_drop),
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(cond)
        x = torch.cat([x_num, emb], dim=-1)
        logits = self.net(x).squeeze(-1)
        return logits
