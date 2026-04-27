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


class ResidualValueTrain(nn.Module):
    """CentralValueTrain analog for the residual head.

    Owns: live residual net, deep-copied snapshot for collection-time forwards,
    optimizer, soft alpha schedule keyed to cumulative real-transition count,
    output L2 penalty, and the mean-zero monitor. Multi-GPU: callers must
    all-reduce the n_real_steps_seen counter (handled in A2CResidualAgent).
    """

    def __init__(
        self,
        obs_dim: int,
        value_size: int,
        c: float,
        lambda_l2: float,
        hidden_units: List[int],
        activation: str,
        learning_rate: float,
        mini_epochs: int,
        warmup_real_steps: int,
        ramp_real_steps: int,
        device: str,
        mixed_precision: bool = False,
    ):
        super().__init__()
        assert lambda_l2 >= 0
        assert warmup_real_steps >= 0
        assert ramp_real_steps > 0
        assert mini_epochs >= 1

        self.lambda_l2 = float(lambda_l2)
        self.warmup_real_steps = int(warmup_real_steps)
        self.ramp_real_steps = int(ramp_real_steps)
        self.mini_epochs = int(mini_epochs)
        self.device = device
        self.mixed_precision = bool(mixed_precision)

        self.live = ResidualValueNet(
            obs_dim=obs_dim, value_size=value_size, c=c,
            hidden_units=hidden_units, activation=activation,
        ).to(device)

        self.snapshot = copy.deepcopy(self.live).to(device)
        for p in self.snapshot.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.live.parameters(), lr=float(learning_rate))

    def alpha(self, n_real_steps: int) -> float:
        progress = (n_real_steps - self.warmup_real_steps) / float(self.ramp_real_steps)
        return float(max(0.0, min(1.0, progress)))

    def refresh_snapshot(self) -> None:
        """Deep-copy live params into snapshot (called once per PPO iteration)."""
        self.snapshot.load_state_dict(self.live.state_dict())

    def compute_loss(self, obs, v_sim_norm, target_resid_norm):
        out = self.live(obs, v_sim_norm)
        mse = ((out - target_resid_norm) ** 2).mean()
        l2 = (out ** 2).mean()
        return mse + self.lambda_l2 * l2

    def train_step(self, obs, v_sim_norm, target_resid_norm):
        loss = self.compute_loss(obs, v_sim_norm, target_resid_norm)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_net(self, obs_real, v_sim_real, target_real, minibatch_size):
        """Run mini_epochs passes over the real-only batch. Caller pre-filters
        to is_real==True rows. Returns aggregate stats dict."""
        n = obs_real.shape[0]
        if n == 0:
            return {"residual_loss": 0.0, "residual_mean_abs": 0.0, "n_real_samples": 0}
        losses = []
        for _ in range(self.mini_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, minibatch_size):
                idx = perm[start:start + minibatch_size]
                losses.append(self.train_step(
                    obs_real[idx], v_sim_real[idx], target_real[idx],
                ))
        return {
            "residual_loss": float(sum(losses) / max(1, len(losses))),
            "residual_mean_abs": self.get_mean_abs(obs_real, v_sim_real),
            "n_real_samples": int(n),
        }

    def get_mean_abs(self, obs, v_sim_norm) -> float:
        """|E[V_residual]| — mean-zero monitor. Sigma-only fusion presumes
        mean-zero; flag if this exceeds 0.1 in normalized units."""
        with torch.no_grad():
            out = self.live(obs, v_sim_norm)
        return float(out.mean().abs().item())

    def get_state(self) -> dict:
        return {
            "live": self.live.state_dict(),
            "snapshot": self.snapshot.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state(self, state: Optional[dict]) -> None:
        """Strict-load when present; else leave at zero-init for baseline-checkpoint compat."""
        if state is None:
            return
        self.live.load_state_dict(state["live"])
        self.snapshot.load_state_dict(state["snapshot"])
        self.optimizer.load_state_dict(state["optimizer"])
