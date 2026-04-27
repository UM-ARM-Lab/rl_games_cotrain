"""A2CResidualAgent — PPO with V_sim + V_residual additive critic.

See docs/superpowers/specs/2026-04-26-residual-value-cotrain-design.md.

NOTE for maintenance: play_steps mirrors a substantial portion of
ContinuousA2CBase.play_steps. If upstream rl_games changes that body,
re-sync this override.
"""
from __future__ import annotations

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

    def _sigma(self) -> torch.Tensor:
        """Sigma from value_mean_std.running_var."""
        return self.value_mean_std.running_var.sqrt()

    def _normalize_v_sim_raw(self, v_sim_raw: torch.Tensor) -> torch.Tensor:
        """Convert raw V_sim values to normalized space using value_mean_std
        in eval mode (no stats update). value_mean_std defaults to eval mode
        after construction; calling forward normalizes (subtracts mean, divides
        by std) without updating the running stats."""
        if not self.normalize_value:
            return v_sim_raw
        return self.value_mean_std(v_sim_raw)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize obs using the same running stats as V_sim's input
        normalization. This makes V_residual's input distribution match what
        V_sim sees, per spec §5 (input is `obs ⊕ sg(V_sim_norm)` in
        normalized obs space)."""
        if not getattr(self, "normalize_input", False):
            return obs
        rms = getattr(self.model, "running_mean_std", None)
        if rms is None:
            return obs
        return rms(obs)  # eval mode → normalize without update

    def init_tensors(self):
        # A2CAgent.init_tensors constructs a CotrainExperienceBuffer via
        # cotrain_cfg, but doesn't pass residual_enabled / real_env_idx.
        # We let it run, then REPLACE the buffer with a residual-enabled one
        # constructed with the same kwargs plus the residual fields.
        A2CAgent.init_tensors(self)

        # Pull the cotrain config; real_env_idx must have been injected by
        # the experiment setup (dynamics_cotrain/experiments/exp_cotrain_ppo.py).
        cot = self.cotrain_cfg
        real_env_idx = cot.get("real_env_idx")
        assert real_env_idx is not None, (
            "A2CResidualAgent requires cotrain.real_env_idx to be injected at "
            "experiment setup. Update dynamics_cotrain/experiments/exp_cotrain_ppo.py "
            "to inject env.unwrapped.idx_train_real.clone() symmetric to sim_env_idx."
        )

        from rl_games.common.cotrain_experience import CotrainExperienceBuffer
        algo_info = {
            "num_actors": self.num_actors,
            "horizon_length": self.horizon_length,
            "has_central_value": self.has_central_value,
            "use_action_masks": getattr(self, "use_action_masks", False),
        }
        self.experience_buffer = CotrainExperienceBuffer(
            self.env_info,
            algo_info,
            self.ppo_device,
            scorer=cot.get("scorer_object", None),
            threshold=cot.get("scorer_threshold", None),
            sim_env_idx=cot.get("sim_env_idx", None),
            real_env_idx=real_env_idx,                     # NEW (residual mode)
            residual_enabled=True,                         # NEW (residual mode)
            scoring_mode=cot.get("scoring_mode", "binary"),
            temperature=cot.get("temperature", None),
            score_direction=cot.get("score_direction", "le"),
            plot_dir=cot.get("plot_dir", None),
            plot_every=cot.get("plot_every", 1),
            plot_num_trajectories=cot.get("plot_num_trajectories", 10),
            mod_method=cot.get("mod_method", "filter"),
            mod_cfg=cot.get("mod", None),
            writer=getattr(self, "writer", None),
        )

        # Now extend update_list / tensor_list so per-step writes flow through.
        for key in ("values_sim", "residual_values_norm", "rewards_sim"):
            if key not in self.update_list:
                self.update_list.append(key)
            if key not in self.tensor_list:
                self.tensor_list.append(key)
        # is_real is painted per-rollout (not per-step), but still flows in tensor_list.
        if "is_real" not in self.tensor_list:
            self.tensor_list.append("is_real")

        td = self.experience_buffer.tensor_dict
        for key in ("values_sim", "residual_values_norm", "rewards_sim", "is_real"):
            assert key in td, f"experience_buffer is missing {key!r} after init"

    def get_action_values(self, obs):
        res = A2CAgent.get_action_values(self, obs)
        v_sim_raw = res["values"]
        obs_tensor = obs["obs"] if isinstance(obs, dict) else obs
        # Normalize obs and V_sim to feed the residual head (spec §5).
        obs_norm = self._normalize_obs(obs_tensor)
        v_sim_norm = self._normalize_v_sim_raw(v_sim_raw)
        v_residual_norm = self.residual_value_net.snapshot(obs_norm, v_sim_norm)
        sigma = self._sigma()
        alpha = self.current_alpha()
        res["values_sim"] = v_sim_raw.clone()
        res["residual_values_norm"] = v_residual_norm
        res["values"] = compute_v_used_raw(v_sim_raw, v_residual_norm, sigma, alpha)
        return res

    def get_values(self, obs):
        v_sim_raw = A2CAgent.get_values(self, obs)
        obs_tensor = obs["obs"] if isinstance(obs, dict) else obs
        obs_norm = self._normalize_obs(obs_tensor)
        v_sim_norm = self._normalize_v_sim_raw(v_sim_raw)
        v_residual_norm = self.residual_value_net.snapshot(obs_norm, v_sim_norm)
        sigma = self._sigma()
        alpha = self.current_alpha()
        return compute_v_used_raw(v_sim_raw, v_residual_norm, sigma, alpha)

    def get_sim_values(self, obs) -> torch.Tensor:
        """V_sim alone — used for the V_sim-only GAE pass bootstrap."""
        return A2CAgent.get_values(self, obs)
