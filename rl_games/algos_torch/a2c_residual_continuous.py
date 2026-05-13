"""A2CResidualAgent — PPO with V_sim + V_residual additive critic.

See docs/superpowers/specs/2026-04-26-residual-value-cotrain-design.md.

NOTE for maintenance: play_steps mirrors a substantial portion of
ContinuousA2CBase.play_steps. If upstream rl_games changes that body,
re-sync this override.
"""
from __future__ import annotations

import hashlib
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import trange

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.residual_value import ResidualValueTrain
from rl_games.common import common_losses
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
        # Populated by prepare_dataset (called inside base train_epoch). The
        # train_epoch override below reads it after delegating to base.
        self._last_dataset = None
        # Cached alpha for the current rollout. Refreshed at the start of
        # play_steps so the H per-step get_action_values calls + bootstrap
        # don't each issue a redundant dist.all_reduce under multi-GPU.
        self._cached_alpha = None
        # Per-rollout diagnostic metrics computed in play_steps after both
        # GAE passes, logged from train_epoch alongside residual training stats.
        self._rollout_diag: dict[str, float] = {}
        # Per-minibatch accumulators populated by calc_gradients overrides.
        # Reset at the start of every train_epoch.
        self._mb_ratio_extreme_count = 0
        self._mb_ratio_total = 0
        self._mb_grad_norm_max = 0.0
        self._mb_grad_norm_nonfinite = False

    def _infer_obs_dim(self) -> int:
        shape = getattr(self, "obs_shape", None)
        assert shape is not None
        if isinstance(shape, (tuple, list)):
            assert len(shape) == 1
            return int(shape[0])
        return int(shape)

    def current_alpha(self) -> float:
        if self._cached_alpha is not None:
            return self._cached_alpha
        return self._compute_alpha()

    def _compute_alpha(self) -> float:
        n = self.n_real_steps_seen
        if self.multi_gpu and dist.is_available() and dist.is_initialized():
            t = torch.tensor([n], device=self.ppo_device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            n = int(t.item())
        return self.residual_value_net.alpha(n)

    def _refresh_alpha_cache(self) -> None:
        """Compute and cache alpha once per rollout (called at start of play_steps)."""
        self._cached_alpha = self._compute_alpha()

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
        # Cache alpha once per rollout. n_real_steps_seen only changes once
        # per rollout (post-loop), so the H per-step current_alpha() reads
        # below all see the same value — the cache eliminates H+ redundant
        # dist.all_reduce collectives under multi-GPU.
        self._refresh_alpha_cache()

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

        # Insertion #4 (e) — rollout-level diagnostic metrics. Computed here
        # because we have all tensors (V_used vs V_sim, both GAE passes,
        # is_real mask) in their pre-flatten layout. Logged in train_epoch.
        self._compute_rollout_diag(
            mb_values=mb_values,
            mb_values_sim=mb_values_sim,
            mb_advs=mb_advs,
            mb_advs_sim=mb_advs_sim,
            mb_returns=mb_returns,
            mb_returns_sim=mb_returns_sim,
            mb_residual_norm=self.experience_buffer.tensor_dict["residual_values_norm"],
            mb_is_real=self.experience_buffer.tensor_dict["is_real"].bool(),
        )

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
            # Cache is now stale — invalidate so post-rollout current_alpha()
            # (e.g. _log_residual_stats) recomputes from the updated counter.
            self._cached_alpha = None
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

    def prepare_dataset(self, batch_dict):
        """RESIDUAL-MODE override of ContinuousA2CBase.prepare_dataset.

        Mirrors the base body verbatim except:
          - value_mean_std updates run against values_sim + returns_sim (not
            V_used) so V_sim's normalization tracks V_sim's distribution.
          - dataset["old_values"] and dataset["returns"] are V_sim's normalized
            tensors (the critic loss regresses V_sim).
          - advantages still come from V_used GAE (policy-side).
          - Adds residual-specific dataset entries: is_real, target_resid_norm,
            values_sim, residual_values_norm.
          - Stashes dataset_dict on self._last_dataset for train_epoch.
        """
        obses = batch_dict["obses"]
        returns = batch_dict["returns"]                 # V_used GAE returns
        dones = batch_dict["dones"]
        values = batch_dict["values"]                   # V_used per-step
        actions = batch_dict["actions"]
        neglogpacs = batch_dict["neglogpacs"]
        mus = batch_dict["mus"]
        sigmas = batch_dict["sigmas"]
        rnn_states = batch_dict.get("rnn_states", None)
        rnn_masks = batch_dict.get("rnn_masks", None)

        # NEW: residual-specific tensors.
        returns_sim = batch_dict["returns_sim"]
        values_sim = batch_dict["values_sim"]
        residual_values_norm = batch_dict["residual_values_norm"]
        is_real = batch_dict["is_real"]

        # advantages from V_used (policy-side) — UNCHANGED from base.
        advantages = returns - values

        # MOD #1: route the base's two-update pattern through V_sim tensors.
        # Base does train(), forward(values), forward(returns), then disables
        # training. We do train(True), forward(values_sim), forward(returns_sim),
        # train(False). At alpha=0, values_sim == values and returns_sim ==
        # returns, so the two RMS updates are bit-identical to base. Using
        # train(True/False) instead of the named-mode helpers to avoid hook flags.
        if self.normalize_value:
            self.value_mean_std.train(True)
            values_sim_norm = self.value_mean_std(values_sim)
            returns_sim_norm = self.value_mean_std(returns_sim)
            self.value_mean_std.train(False)
        else:
            values_sim_norm = values_sim
            returns_sim_norm = returns_sim

        # MOD #2: residual target in normalized space (stop-grad).
        target_resid_norm = (returns_sim_norm - values_sim_norm).detach()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # MOD #3: dataset_dict — old_values and returns OVERRIDDEN to V_sim
        # normalized tensors. The critic loss regresses V_sim toward returns_sim,
        # NOT V_used toward returns_used. advantages stays from V_used GAE.
        dataset_dict = {}
        dataset_dict["old_values"] = values_sim_norm           # OVERRIDE: V_sim normalized
        dataset_dict["old_logp_actions"] = neglogpacs
        dataset_dict["advantages"] = advantages                 # V_used GAE (policy)
        dataset_dict["returns"] = returns_sim_norm              # OVERRIDE: V_sim normalized
        dataset_dict["actions"] = actions
        dataset_dict["obs"] = obses
        dataset_dict["dones"] = dones
        dataset_dict["rnn_states"] = rnn_states
        dataset_dict["rnn_masks"] = rnn_masks
        dataset_dict["mu"] = mus
        dataset_dict["sigma"] = sigmas
        # NEW residual-specific keys (used by train_epoch in Task 8):
        dataset_dict["is_real"] = is_real
        dataset_dict["target_resid_norm"] = target_resid_norm
        dataset_dict["values_sim"] = values_sim
        dataset_dict["residual_values_norm"] = residual_values_norm

        self.dataset.update_values_dict(dataset_dict)

        # Preserve the central_value branch from the base — UNCHANGED in
        # semantics, but renamed to dataset_dict_cv to avoid clobbering the
        # dataset_dict we stash on self._last_dataset below. The base body
        # reuses the variable name because it doesn't need the original;
        # we do.
        if self.has_central_value:
            dataset_dict_cv = {}
            dataset_dict_cv["old_values"] = values_sim_norm    # V_sim normalized
            dataset_dict_cv["advantages"] = advantages
            dataset_dict_cv["returns"] = returns_sim_norm
            dataset_dict_cv["actions"] = actions
            dataset_dict_cv["obs"] = batch_dict["states"]
            dataset_dict_cv["dones"] = dones
            dataset_dict_cv["rnn_masks"] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict_cv)

        # MOD #4: stash for the residual training step in train_epoch (Task 8).
        self._last_dataset = dataset_dict

    def train_epoch(self):
        """Refresh snapshot → run base train_epoch → train residual on real-only batch.

        Base train_epoch internally calls play_steps, prepare_dataset, and
        the PPO inner loop. We delegate to it (no copy of the inner-loop
        body) and slot residual training in afterward.
        """
        # Reset per-iter metric accumulators populated by calc_gradients
        # and trancate_gradients_and_step. Their values flush in
        # _log_residual_stats at the end of this method.
        self._mb_ratio_extreme_count = 0
        self._mb_ratio_total = 0
        self._mb_grad_norm_max = 0.0
        self._mb_grad_norm_nonfinite = False
        # Capture RNG state at the start of this iteration. Logged at end.
        rng_fingerprint = self._compute_rng_fingerprint()

        self.residual_value_net.refresh_snapshot()
        result = A2CAgent.train_epoch(self)

        ds = self._last_dataset
        assert ds is not None, "train_epoch called before prepare_dataset has run"
        is_real = ds["is_real"].bool()
        if is_real.any():
            # Match the rollout-time obs space: residual head saw normalized
            # obs in get_action_values, so training input must also be
            # normalized obs to keep the input distribution consistent.
            obs_real = self._normalize_obs(ds["obs"][is_real])
            v_sim_real = ds["values_sim"][is_real]
            with torch.no_grad():
                v_sim_norm_real = (
                    self.value_mean_std(v_sim_real) if self.normalize_value else v_sim_real
                )
            target_real = ds["target_resid_norm"][is_real]
            # Install a backward hook on actor params that fires if any
            # gradient is accumulated into them during V_residual.train_step.
            # An accumulated grad here would mean the residual loss has an
            # accidental autograd path into the actor (Mode 2 hypothesis H_B).
            hook_handles, leak_flag = self._install_actor_grad_leak_hook()
            try:
                stats = self.residual_value_net.train_net(
                    obs_real=obs_real,
                    v_sim_real=v_sim_norm_real,
                    target_real=target_real,
                    minibatch_size=self.minibatch_size,
                )
            finally:
                for h in hook_handles:
                    h.remove()
            stats["actor_grad_leaked"] = bool(leak_flag[0])
        else:
            # No real data this rollout — log zero-sample marker so the alpha
            # curve has no gaps and misconfiguration (e.g., empty real_env_idx)
            # is visible in TensorBoard.
            stats = {
                "residual_loss": 0.0,
                "residual_mean_abs": 0.0,
                "n_real_samples": 0,
                "actor_grad_leaked": False,
            }
        self._log_residual_stats(stats, rng_fingerprint=rng_fingerprint)
        return result

    def _log_residual_stats(self, stats: dict, *, rng_fingerprint: int) -> None:
        if not hasattr(self, "writer") or self.writer is None:
            return
        # Match write_stats' post-increment frame so residual/* and losses/*
        # scalars line up at the same TB step for the same iteration.
        frame = self.frame + self.curr_frames
        w = self.writer
        w.add_scalar("residual/loss", stats["residual_loss"], frame)
        w.add_scalar("residual/mean_abs", stats["residual_mean_abs"], frame)
        w.add_scalar("residual/n_real_samples", stats["n_real_samples"], frame)
        w.add_scalar("residual/alpha", self.current_alpha(), frame)
        # Mode-1 dataset-split metrics (computed in _compute_rollout_diag).
        for k, v in self._rollout_diag.items():
            w.add_scalar(k, v, frame)
        # Mode-2 per-minibatch accumulators flushed from calc_gradients /
        # trancate_gradients_and_step.
        ratio_frac = (
            self._mb_ratio_extreme_count / max(1, self._mb_ratio_total)
        )
        w.add_scalar("ppo/ratio_extreme_frac", ratio_frac, frame)
        w.add_scalar(
            "ppo/grad_norm_pre_clip_max",
            float("inf") if self._mb_grad_norm_nonfinite else self._mb_grad_norm_max,
            frame,
        )
        w.add_scalar("debug/rng_fingerprint", float(rng_fingerprint), frame)
        w.add_scalar("debug/actor_grad_leaked", float(stats["actor_grad_leaked"]), frame)
        if stats["residual_mean_abs"] > 0.1:
            print(
                f"[residual] WARN: mean|V_residual|={stats['residual_mean_abs']:.4f} "
                f"exceeds 0.1 — sigma-only fusion presumes mean-zero."
            )
        if stats["actor_grad_leaked"]:
            print(
                "[residual] ERROR: actor parameter received a gradient during "
                "V_residual.train_step — the residual loss has an autograd path "
                "into the actor. This is a Mode 2 H_B failure."
            )

    def _compute_rollout_diag(
        self,
        *,
        mb_values: torch.Tensor,
        mb_values_sim: torch.Tensor,
        mb_advs: torch.Tensor,
        mb_advs_sim: torch.Tensor,
        mb_returns: torch.Tensor,
        mb_returns_sim: torch.Tensor,
        mb_residual_norm: torch.Tensor,
        mb_is_real: torch.Tensor,
    ) -> None:
        """Populate self._rollout_diag with sim/real-split metrics.

        Shapes: (T, N, value_size) for value tensors and advantages,
        (T, N) for mb_is_real. Splits along the (T, N) flat axis.
        """
        with torch.no_grad():
            real = mb_is_real
            sim = ~real
            n_sim = int(sim.sum().item())
            n_real = int(real.sum().item())

            res = mb_residual_norm
            # Flatten value_size by mean — value_size is 1 in our config so this
            # is a no-op, but written symbolically to be robust.
            res_per_cell = res.mean(dim=-1)
            res_abs = res_per_cell.abs()

            c = float(self.residual_value_net.live.c)
            sigma = self._sigma()
            alpha = self.current_alpha()

            # Mode-1 metrics.
            mean_sim = res_per_cell[sim].mean().item() if n_sim > 0 else 0.0
            mean_real = res_per_cell[real].mean().item() if n_real > 0 else 0.0
            std_sim = res_per_cell[sim].std().item() if n_sim > 1 else 0.0
            frac_sat = (res_abs > 0.9 * c).float().mean().item()
            # advantage shift = (adv_used - adv_sim) averaged over sim cells.
            adv_diff = mb_advs - mb_advs_sim
            adv_diff_per_cell = adv_diff.mean(dim=-1)
            adv_shift_sim = (
                adv_diff_per_cell[sim].mean().item() if n_sim > 0 else 0.0
            )
            # Sign flip: sign(adv_used) != sign(adv_sim) per cell, sim only.
            sign_used = mb_advs.mean(dim=-1).sign()
            sign_sim = mb_advs_sim.mean(dim=-1).sign()
            flips = (sign_used != sign_sim)
            sign_flip_frac = (
                flips[sim].float().mean().item() if n_sim > 0 else 0.0
            )

            # Mode-2: bit-identity check. At alpha=0, V_used == V_sim exactly
            # and both GAE passes should give identical advantages/returns.
            ident = max(
                float((mb_values - mb_values_sim).abs().max().item()),
                float((mb_advs - mb_advs_sim).abs().max().item()),
                float((mb_returns - mb_returns_sim).abs().max().item()),
            )

        self._rollout_diag = {
            "residual/V_residual_norm_mean_sim": mean_sim,
            "residual/V_residual_norm_mean_sim_minus_real": mean_sim - mean_real,
            "residual/V_residual_norm_std_sim": std_sim,
            "residual/frac_saturated": frac_sat,
            "residual/advantage_shift_sim_raw": adv_shift_sim,
            "residual/sigma": float(sigma.mean().item()),
            "residual/sim_adv_sign_flip_frac": sign_flip_frac,
            "debug/alpha0_value_identity_max": ident if alpha == 0.0 else 0.0,
            "debug/value_identity_max_always": ident,
        }

    def _compute_rng_fingerprint(self) -> int:
        """Small deterministic checksum of CPU + CUDA RNG state.

        Used to detect non-bit-identity at alpha=0 between residual-mode and
        baseline runs (Mode 2). A divergence at iter N versus a baseline run
        before _step 1671 supports H_B (residual code perturbs the trajectory).
        """
        h = hashlib.blake2b(digest_size=8)
        h.update(torch.get_rng_state().numpy().tobytes())
        if torch.cuda.is_available():
            h.update(torch.cuda.get_rng_state(self.ppo_device).numpy().tobytes())
        # blake2b 8-byte digest fits in int64; cast to signed for TB-safe scalar.
        return int.from_bytes(h.digest(), "little", signed=True)

    def _install_actor_grad_leak_hook(self):
        """Install backward hooks on actor params asserting no gradient flows
        through them during the V_residual training step.

        Returns (handles, leak_flag). Caller must remove handles after the
        residual training step completes.
        """
        leak_flag = [False]

        def _fire(_grad):
            leak_flag[0] = True

        handles = []
        for p in self.model.parameters():
            if p.requires_grad:
                handles.append(p.register_hook(_fire))
        return handles, leak_flag

    def calc_gradients(self, input_dict):
        """Override of A2CAgent.calc_gradients — verbatim except for the
        ratio_extreme accumulator. Body must stay in sync with
        a2c_continuous.py:79-180; resync if upstream changes.
        """
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.amp.autocast(device_type="cuda", enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            # INSERT: ratio_extreme accumulator. action_log_probs is the
            # current-policy neg-log-prob; old_action_log_probs_batch is the
            # rollout's neg-log-prob. ratio = exp(old - new).
            with torch.no_grad():
                ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
                extreme = (ratio < 0.1) | (ratio > 10.0)
                self._mb_ratio_extreme_count += int(extreme.sum().item())
                self._mb_ratio_total += int(ratio.numel())

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip
            )

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
                )
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = (
                a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            )

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )

        self.train_result = (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )

    def trancate_gradients_and_step(self):
        """Override of CommonAgent.trancate_gradients_and_step — verbatim
        except that the return value of clip_grad_norm_ (pre-clip total norm)
        is captured into self._mb_grad_norm_max. Resync if upstream changes
        a2c_common.py:405-428.
        """
        if self.multi_gpu:
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            pre_clip = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self._record_grad_norm(pre_clip)
        else:
            # Compute pre-clip norm without applying clipping. Mirrors the
            # behavior we'd get if truncate_grads were always True for metric.
            self.scaler.unscale_(self.optimizer)
            params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
            if params_with_grad:
                norms = torch.stack([
                    p.grad.detach().norm(2) for p in params_with_grad
                ])
                self._record_grad_norm(norms.norm(2))

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _record_grad_norm(self, norm_tensor: torch.Tensor) -> None:
        v = float(norm_tensor.item())
        if not (v == v) or v == float("inf") or v == float("-inf"):
            self._mb_grad_norm_nonfinite = True
            return
        if v > self._mb_grad_norm_max:
            self._mb_grad_norm_max = v


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
