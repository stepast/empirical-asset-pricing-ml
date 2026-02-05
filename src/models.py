from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class MLPReturnPredictor(nn.Module):
    """MLP for return prediction following Gu, Kelly, and Xiu (2020)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())

            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
