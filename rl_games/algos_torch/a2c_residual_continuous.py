"""A2CResidualAgent — PPO with V_sim + V_residual additive critic.

See docs/superpowers/specs/2026-04-26-residual-value-cotrain-design.md.

NOTE for maintenance: play_steps mirrors a substantial portion of
ContinuousA2CBase.play_steps. If upstream rl_games changes that body,
re-sync this override.
"""
from __future__ import annotations

import time

import torch
import torch.distributed as dist
from tqdm import trange

from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.residual_value import ResidualValueTrain
from rl_games.common.a2c_common import (
    _filter_train_done_indices,
    _slice_batch_dict_to_train,
    swap_and_flatten01,
)


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
        assert self.cotrain_enabled, (
            "A2CResidualAgent requires cotrain.enabled=True (residual mode is "
            "built on the cotrain experience buffer)."
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
        """Sigma matching RunningMeanStd.forward: sqrt(running_var + eps).
        Cast to float32 to match buffer dtype and avoid silent promotion."""
        var = self.value_mean_std.running_var
        eps = self.value_mean_std.epsilon
        return (var + eps).sqrt().to(torch.float32)

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

        # values_sim and residual_values_norm flow from get_action_values (per-step
        # writes), so they go in update_list. rewards_sim is computed AFTER env_step
        # (from shaped_rewards) and is written explicitly via update_data, so it
        # only needs to be in tensor_list (so get_transformed_list picks it up
        # for batch_dict). is_real is painted per-rollout, also tensor_list only.
        for key in ("values_sim", "residual_values_norm"):
            if key not in self.update_list:
                self.update_list.append(key)
            if key not in self.tensor_list:
                self.tensor_list.append(key)
        for key in ("rewards_sim", "is_real"):
            if key not in self.tensor_list:
                self.tensor_list.append(key)

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

    def play_steps(self):
        # Mirror of ContinuousA2CBase.play_steps with four marked insertions
        # for the residual-value two-pass GAE split. Re-sync if upstream
        # changes the base body (a2c_common.py:877-986).
        self.experience_buffer.paint_is_real()  # Insertion #1

        update_list = self.update_list

        step_time = 0.0

        for n in trange(self.horizon_length, leave=False, desc="Playing steps"):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            step_time_start = time.time()
            # Val envs use mean actions for deterministic rollout (comparable to dedicated eval)
            step_actions = res_dict["actions"]
            if self.num_val > 0:
                step_actions = step_actions.clone()
                step_actions[self.num_train:] = res_dict["mus"][self.num_train:]
            self.obs, rewards, self.dones, infos = self.env_step(step_actions)

            step_time_end = time.time()

            step_time += step_time_end - step_time_start

            # Insertion #2 — two-stream value-bootstrap split.
            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and "time_outs" in infos:
                time_outs = self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                rewards_used, rewards_sim = apply_value_bootstrap_split(
                    shaped_rewards, time_outs,
                    res_dict["values"],          # V_used (overwritten in get_action_values)
                    res_dict["values_sim"],      # V_sim alone
                    self.gamma,
                )
            else:
                rewards_used = shaped_rewards
                rewards_sim = shaped_rewards.clone()
            self.experience_buffer.update_data("rewards", n, rewards_used)
            self.experience_buffer.update_data("rewards_sim", n, rewards_sim)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[:: self.num_agents]

            # For train/val setups, exclude val env done episodes from reward tracking.
            # Val envs (large hole, easier) would inflate mean_rewards and bias checkpoint saves.
            # process_infos uses env-level aggregates from the env itself, so it keeps full indices.
            train_done_indices = (
                _filter_train_done_indices(env_done_indices, self.num_train)
                if self.env_has_train_val
                else env_done_indices
            )

            self.game_rewards.update(self.current_rewards[train_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[train_done_indices])
            self.game_lengths.update(self.current_lengths[train_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)
        last_values_sim = self.get_sim_values(self.obs)  # Insertion #3

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_masks = self.experience_buffer.tensor_dict.get("mask", None)

        # mod_method=rew_low/rew_sub mutates mb_rewards in-place here so GAE
        # propagates the penalty backward. mod_method=filter is a no-op.
        if self.cotrain_enabled:
            self.experience_buffer.apply_pre_gae(self.experience_buffer.tensor_dict)

        if mb_masks is not None:
            mb_advs = self.discount_values_masks(
                fdones, last_values, mb_fdones, mb_values, mb_rewards, mb_masks.float()
            )
        else:
            mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        # Insertion #4 (a) — second GAE pass with V_sim alone — produces
        # returns_sim for V_sim's regression target in prepare_dataset.
        mb_values_sim = self.experience_buffer.tensor_dict["values_sim"]
        mb_rewards_sim = self.experience_buffer.tensor_dict["rewards_sim"]
        if mb_masks is not None:
            mb_advs_sim = self.discount_values_masks(
                fdones, last_values_sim, mb_fdones, mb_values_sim, mb_rewards_sim, mb_masks.float(),
            )
        else:
            mb_advs_sim = self.discount_values(
                fdones, last_values_sim, mb_fdones, mb_values_sim, mb_rewards_sim,
            )
        mb_returns_sim = mb_advs_sim + mb_values_sim

        batch_dict = {}
        if self.cotrain_enabled:
            self.experience_buffer.update_data_full("returns", mb_returns)
            self.experience_buffer.update_data_full("returns_sim", mb_returns_sim)  # Insertion #4 (b)
            self.experience_buffer.apply_post_gae(
                td=self.experience_buffer.tensor_dict,
                rnn_states_raw=None,
                seq_length=self.horizon_length,
            )
            # Insertion #4 (d) — alpha-schedule counter increment.
            n_real_this_rollout = (
                self.horizon_length * int(self.experience_buffer.real_env_idx.numel())
            )
            self.n_real_steps_seen += n_real_this_rollout
            tensor_list_with_return = self.tensor_list + ["returns", "returns_sim"]  # Insertion #4 (c)
            batch_dict = self.experience_buffer.get_transformed_list(
                swap_and_flatten01, tensor_list_with_return
            )
        else:        # old method
            batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
            batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time

        # Slice out val env rows so prepare_dataset normalizes only over train envs.
        # After swap_and_flatten01, layout is [train_rows | val_rows] with boundary at batch_size.
        if self.env_has_train_val:
            batch_dict = _slice_batch_dict_to_train(batch_dict, self.batch_size)

        return batch_dict


def apply_value_bootstrap_split(
    shaped_rewards: torch.Tensor,
    time_outs: torch.Tensor,
    v_used_raw: torch.Tensor,
    v_sim_raw: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two reward tensors for the two-GAE-pass split.

    On time-out cells (time_outs != 0), value_bootstrap adds gamma*V*time_outs
    to the reward. We need:
      - rewards_used: bootstraps with V_used (for policy GAE)
      - rewards_sim:  bootstraps with V_sim alone (for V_sim-only GAE)
    """
    rewards_used = shaped_rewards + gamma * v_used_raw * time_outs
    rewards_sim = shaped_rewards + gamma * v_sim_raw * time_outs
    return rewards_used, rewards_sim
