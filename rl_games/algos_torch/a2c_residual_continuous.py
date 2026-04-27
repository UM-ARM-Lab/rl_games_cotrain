"""A2CResidualAgent — PPO with V_sim + V_residual additive critic.

See docs/superpowers/specs/2026-04-26-residual-value-cotrain-design.md.

NOTE for maintenance: play_steps mirrors a substantial portion of
ContinuousA2CBase.play_steps. If upstream rl_games changes that body,
re-sync this override.
"""
from __future__ import annotations

import copy
import time
from typing import Optional

import torch
import torch.distributed as dist

from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.residual_value import ResidualValueTrain


def compute_v_used_raw(
    v_sim_raw: torch.Tensor,
    v_residual_norm: torch.Tensor,
    sigma: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """V_used_raw = V_sim_raw + alpha * sigma * V_residual_norm.

    Sigma scaling lifts V_residual from normalized units back to raw units
    so the additive sum is well-defined. No mu offset because V_residual is
    presumed mean-zero (softly enforced via L2 + monitored at runtime).
    """
    return v_sim_raw + alpha * sigma * v_residual_norm


class A2CResidualAgent(A2CAgent):
    """PPO agent with an additional residual value head.

    Hard incompatibilities (asserted at construction):
      - is_rnn=True (cotrain FilterConsumer doesn't reshuffle RNN states).
      - has_central_value=True with use_experimental_cv=False (the central
        value path redirects value_mean_std and disables the standard value
        loss that V_sim relies on).
    """

    def __init__(self, base_name, params):
        A2CAgent.__init__(self, base_name, params)

        assert not self.is_rnn, (
            "A2CResidualAgent does not support is_rnn in v1 — cotrain "
            "FilterConsumer does not reshuffle RNN states. Use a2c_continuous."
        )
        assert not (self.has_central_value and not self.use_experimental_cv), (
            "A2CResidualAgent: central_value with use_experimental_cv=False "
            "redirects value_mean_std and disables the standard value loss "
            "that V_sim relies on. Set use_experimental_cv=True or disable "
            "central_value."
        )
        assert self.value_mean_std is self.model.value_mean_std, (
            "A2CResidualAgent: value_mean_std must be the model's, not the "
            "central value net's, so the residual head shares normalization."
        )

        rcfg = self.config["residual_value_config"]
        self.residual_value_net = ResidualValueTrain(
            obs_dim=self._infer_obs_dim(),
            value_size=self.value_size,
            c=rcfg["c"],
            lambda_l2=rcfg["lambda_l2"],
            hidden_units=list(rcfg["hidden_units"]),
            activation=rcfg["activation"],
            learning_rate=rcfg["learning_rate"],
            mini_epochs=rcfg["mini_epochs"],
            warmup_real_steps=rcfg["warmup_real_steps"],
            ramp_real_steps=rcfg["ramp_real_steps"],
            device=self.ppo_device,
            mixed_precision=rcfg.get("mixed_precision", False),
        )
        self.n_real_steps_seen = 0

    def _infer_obs_dim(self) -> int:
        shape = getattr(self, "obs_shape", None)
        assert shape is not None
        if isinstance(shape, (tuple, list)):
            assert len(shape) == 1
            return int(shape[0])
        return int(shape)

    def current_alpha(self) -> float:
        n = self.n_real_steps_seen
        if self.multi_gpu and dist.is_available() and dist.is_initialized():
            t = torch.tensor([n], device=self.ppo_device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            n = int(t.item())
        return self.residual_value_net.alpha(n)

    def get_full_state_weights(self):
        state = A2CAgent.get_full_state_weights(self)
        state["residual_value"] = self.residual_value_net.get_state()
        state["n_real_steps_seen"] = self.n_real_steps_seen
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        A2CAgent.set_full_state_weights(self, weights, set_epoch=set_epoch)
        self.residual_value_net.load_state(weights.get("residual_value"))
        self.n_real_steps_seen = int(weights.get("n_real_steps_seen", 0))
