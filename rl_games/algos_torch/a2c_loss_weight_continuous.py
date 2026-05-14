"""A2CLossWeightAgent — PPO with per-cell scorer-mask weighted gradients.

See docs/superpowers/specs/2026-05-13-loss-weight-cotrain-design.md.

Per-sample minibatch weights:
    w_i = is_real_i + (1 - is_real_i) * loss_mask_i * beta
    beta = num_real / num_sim_accept   (0 if num_sim_accept == 0)

Every per-sample PPO loss term (actor, critic, entropy, bound) is reduced as
the sum(w * loss) / max(sum(w), eps). When sum(w) == 0, the optimizer step is
skipped for the minibatch.

NOTE: play_steps/prepare_dataset shadow the base bodies only to plumb is_real
and loss_mask into the dataset. calc_gradients mirrors a substantial portion
of A2CAgent.calc_gradients. Resync if upstream rl_games changes those bodies.
"""
from __future__ import annotations

import torch

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.loss_weight import compute_loss_weights
from rl_games.common import common_losses
from rl_games.common.a2c_common import _slice_batch_dict_to_train, swap_and_flatten01


class A2CLossWeightAgent(A2CAgent):
    """PPO agent that masks sim per-cell gradients by a binary scorer mask M.

    Hard incompatibilities (asserted at construction):
      - is_rnn=True (cotrain FilterConsumer doesn't reshuffle RNN states).
      - has_central_value=True (would need its own weighted update path).
      - cotrain.enabled=False (the buffer + mask plumbing depends on it).
      - cotrain.mod_method != 'loss_weight'.
      - cotrain.scoring_mode != 'binary'.
    """

    def __init__(self, base_name, params):
        A2CAgent.__init__(self, base_name, params)

        assert not self.is_rnn, (
            "A2CLossWeightAgent does not support is_rnn in v1 — cotrain "
            "FilterConsumer does not reshuffle RNN states. Use a2c_continuous."
        )
        assert not self.has_central_value, (
            "A2CLossWeightAgent does not support central_value in v1 — the "
            "central value loss path needs its own weighted update."
        )
        assert self.cotrain_enabled, (
            "A2CLossWeightAgent requires cotrain.enabled=True (the loss mask "
            "is computed by the cotrain experience buffer)."
        )
        assert self.cotrain_cfg.get("mod_method", "filter") == "loss_weight", (
            f"A2CLossWeightAgent requires cotrain.mod_method='loss_weight', "
            f"got {self.cotrain_cfg.get('mod_method', 'filter')!r}"
        )
        assert self.cotrain_cfg.get("scoring_mode", "binary") == "binary", (
            f"A2CLossWeightAgent requires cotrain.scoring_mode='binary', "
            f"got {self.cotrain_cfg.get('scoring_mode', 'binary')!r}"
        )

        # Per-minibatch diagnostics flushed by train_epoch -> diagnostics tape.
        self._mb_beta_last: float = 0.0
        self._mb_sum_w_zero_count: int = 0
        self._mb_total_count: int = 0

    def init_tensors(self):
        A2CAgent.init_tensors(self)

        # The base CotrainExperienceBuffer construction (a2c_common.py:571)
        # already passes real_env_idx from cotrain_cfg, and the buffer
        # auto-enables loss_weight allocation when mod_method='loss_weight'.
        # So we only need to (a) verify the experiment injected real_env_idx
        # and (b) declare is_real/loss_mask in tensor_list so they flow into
        # the dataset via get_transformed_list.
        cot = self.cotrain_cfg
        assert cot.get("real_env_idx") is not None, (
            "A2CLossWeightAgent requires cotrain.real_env_idx to be injected at "
            "experiment setup (see CotrainPPOExperiment.training)."
        )

        for key in ("is_real", "loss_mask"):
            if key not in self.tensor_list:
                self.tensor_list.append(key)

        td = self.experience_buffer.tensor_dict
        assert "is_real" in td and "loss_mask" in td, (
            "experience_buffer is missing is_real / loss_mask after base init — "
            "check that mod_method='loss_weight' is set in cotrain_cfg."
        )

    def play_steps(self):
        # paint is_real before rollout; the base play_steps will fire
        # apply_pre_gae and apply_post_gae which lets LossWeightConsumer
        # populate loss_mask. No further per-step overrides are needed.
        self.experience_buffer.paint_is_real()
        return super().play_steps()

    def prepare_dataset(self, batch_dict):
        # Mirror base ContinuousA2CBase.prepare_dataset verbatim, then thread
        # is_real and loss_mask into the dataset so calc_gradients sees them.
        super().prepare_dataset(batch_dict)
        is_real = batch_dict["is_real"]
        loss_mask = batch_dict["loss_mask"]
        # Append into the PPODataset's values dict. PPODataset.update_values_dict
        # merges; calling it again with the existing dict already in place
        # would clobber, so we patch the dict object directly.
        existing = self.dataset.values_dict
        existing["is_real"] = is_real
        existing["loss_mask"] = loss_mask

    def calc_gradients(self, input_dict):
        """Override of A2CAgent.calc_gradients — verbatim body except:
          - per-cell weights w from is_real + loss_mask
          - each loss term is reduced as sum(w * L) / max(sum(w), eps)
          - if sum(w) == 0, skip backward/optimizer for this minibatch
        Resync if upstream changes a2c_continuous.py:79-180.
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

        # Loss-weight tensors. Shapes from buffer flatten: (B,) bool / (B,) float.
        is_real_batch = input_dict["is_real"].bool()
        loss_mask_batch = input_dict["loss_mask"].float()
        w, beta, denom = compute_loss_weights(is_real_batch, loss_mask_batch)
        # PPODataset slices to minibatch already; w/denom now match advantage.

        self._mb_beta_last = beta
        self._mb_total_count += 1

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        with torch.amp.autocast(device_type="cuda", enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

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

            # ---- Weighted reductions ----------------------------------------
            # All per-sample losses have leading dim B. c_loss may be (B, V).
            # weight tensor is (B,). Reduce each component to a scalar via
            # sum(w * L) / max(sum(w), eps).
            eps = 1e-8
            denom_safe = denom.clamp(min=eps)

            def _weighted_mean(loss_tensor: torch.Tensor) -> torch.Tensor:
                if loss_tensor.dim() == 1:
                    return (w * loss_tensor).sum() / denom_safe
                # (B, V) — broadcast over the value dim.
                return (w.unsqueeze(-1) * loss_tensor).sum() / (denom_safe * loss_tensor.shape[-1])

            a_loss_red = _weighted_mean(a_loss)
            c_loss_red = _weighted_mean(c_loss) if c_loss.numel() > 1 else c_loss.mean()
            entropy_red = _weighted_mean(entropy)
            b_loss_red = _weighted_mean(b_loss) if b_loss.numel() > 1 else b_loss.mean()

            loss = (
                a_loss_red
                + 0.5 * c_loss_red * self.critic_coef
                - entropy_red * self.entropy_coef
                + b_loss_red * self.bounds_loss_coef
            )

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        # Skip optimizer step on zero-weight minibatch. KL still computable.
        if denom.item() == 0.0:
            self._mb_sum_w_zero_count += 1
            with torch.no_grad():
                reduce_kl = True
                kl_dist = torch_ext.policy_kl(
                    mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl
                )
            self.train_result = (
                a_loss_red.detach(),
                c_loss_red.detach() if isinstance(c_loss_red, torch.Tensor) else torch.zeros(1, device=self.ppo_device),
                entropy_red.detach(),
                kl_dist,
                self.last_lr,
                lr_mul,
                mu.detach(),
                sigma.detach(),
                b_loss_red.detach() if isinstance(b_loss_red, torch.Tensor) else torch.zeros(1, device=self.ppo_device),
            )
            return

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = True
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl
            )

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": None,
            },
            curr_e_clip,
            0,
        )

        self.train_result = (
            a_loss_red,
            c_loss_red if isinstance(c_loss_red, torch.Tensor) else torch.zeros(1, device=self.ppo_device),
            entropy_red,
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss_red if isinstance(b_loss_red, torch.Tensor) else torch.zeros(1, device=self.ppo_device),
        )
