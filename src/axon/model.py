"""PyTorch MLP for HEP binary classification.

Two variants:
  v1 — small  (28 → 64 → 32 → 1),  ~3K params
  v2 — deep   (28 → 128 → 128 → 64 → 1), ~28K params
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_v1(n_features: int = 28) -> MLP:
    return MLP([n_features, 64, 32, 1])


def build_v2(n_features: int = 28) -> MLP:
    return MLP([n_features, 128, 128, 64, 1])
