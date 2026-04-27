"""Residual value head + training helper for sim-to-real PPO co-training.

See docs/superpowers/specs/2026-04-26-residual-value-cotrain-design.md.
"""
from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class ResidualValueNet(nn.Module):
    """Bounded residual head in normalized value space.

    Input: obs ⊕ sg(V_sim_norm). Output: c·tanh(raw/c) — bounded to [-c, c]
    in normalized units. Final linear layer zero-initialized so output ≡ 0
    at construction.
    """

    def __init__(
        self,
        obs_dim: int,
        value_size: int,
        c: float,
        hidden_units: List[int],
        activation: str = "elu",
    ):
        super().__init__()
        assert c > 0
        assert len(hidden_units) >= 1
        assert activation in _ACTIVATIONS

        self.c = float(c)
        self.value_size = int(value_size)

        in_dim = obs_dim + value_size
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(_ACTIVATIONS[activation]())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, value_size)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor, v_sim_norm: torch.Tensor) -> torch.Tensor:
        """V_residual_norm in [-c, c]. v_sim_norm is detached internally."""
        x = torch.cat([obs, v_sim_norm.detach()], dim=-1)
        raw = self.head(self.trunk(x))
        return self.c * torch.tanh(raw / self.c)
