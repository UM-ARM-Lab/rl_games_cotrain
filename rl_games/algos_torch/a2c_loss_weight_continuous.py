"""A2CLossWeightAgent — PPO with critic-only SDF rejection mask.

For each minibatch sample, the SDF scorer's binary acceptance mask
(``td['loss_mask']``, painted post-GAE by ``LossWeightConsumer``) zeroes the
critic loss on rejected sim cells (in-penetration chunks) and leaves it
untouched on accepted cells. Actor, entropy, and bound losses are reduced
identically to ``a2c_continuous`` — plain ``.mean()`` — so rejected sim
states still update the shared trunk through the policy-side terms.

Reduction (matches a2c_continuous with rnn_masks=None):

    a_loss_red  = a_loss.mean()
    c_loss_red  = (c_loss * loss_mask.unsqueeze(-1)).mean()   # c_loss: (B, V)
    entropy_red = entropy.mean()
    b_loss_red  = b_loss.mean()

Real-env samples carry loss_mask=1 by buffer construction
(CotrainExperienceBuffer paints the sim-only weight grid; real columns are
left at 1.0), so they always contribute to the critic — no separate is_real
plumbing is needed in the agent.

NOTE: prepare_dataset shadows the base body only to plumb loss_mask into the
dataset, and calc_gradients mirrors the bulk of A2CAgent.calc_gradients.
Resync if upstream rl_games changes those bodies.
"""
from __future__ import annotations

import torch

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import common_losses


class A2CLossWeightAgent(A2CAgent):
    """PPO agent that zeroes per-sample critic gradients on SDF-rejected sim cells.

    Hard incompatibilities (asserted at construction):
      - is_rnn=True (cotrain buffers don't reshuffle RNN states).
      - has_central_value=True (would need its own masked critic path).
      - cotrain.enabled=False (the loss_mask plumbing depends on it).
      - cotrain.mod_method != 'loss_weight'.
      - cotrain.scoring_mode != 'binary'.
    """

    def __init__(self, base_name, params):
        A2CAgent.__init__(self, base_name, params)

        assert not self.is_rnn, (
            "A2CLossWeightAgent does not support is_rnn — cotrain "
            "buffers do not reshuffle RNN states. Use a2c_continuous."
        )
        assert not self.has_central_value, (
            "A2CLossWeightAgent does not support central_value — the "
            "central value loss path needs its own masked critic update."
        )
        assert self.cotrain_enabled, (
            "A2CLossWeightAgent requires cotrain.enabled=True (the loss "
            "mask is computed by the cotrain experience buffer)."
        )
        assert self.cotrain_cfg.get("mod_method", "filter") == "loss_weight", (
            f"A2CLossWeightAgent requires cotrain.mod_method='loss_weight', "
            f"got {self.cotrain_cfg.get('mod_method', 'filter')!r}"
        )
        assert self.cotrain_cfg.get("scoring_mode", "binary") == "binary", (
            f"A2CLossWeightAgent requires cotrain.scoring_mode='binary', "
            f"got {self.cotrain_cfg.get('scoring_mode', 'binary')!r}"
        )

    def init_tensors(self):
        A2CAgent.init_tensors(self)

        # The base CotrainExperienceBuffer auto-allocates loss_mask whenever
        # mod_method='loss_weight'. We just need it in tensor_list so it
        # flows into the dataset via get_transformed_list.
        if "loss_mask" not in self.tensor_list:
            self.tensor_list.append("loss_mask")

        td = self.experience_buffer.tensor_dict
        assert "loss_mask" in td, (
            "experience_buffer is missing loss_mask after base init — "
            "check that mod_method='loss_weight' is set in cotrain_cfg."
        )

    def prepare_dataset(self, batch_dict):
        # Mirror base ContinuousA2CBase.prepare_dataset verbatim, then thread
        # loss_mask into the dataset so calc_gradients sees it.
        super().prepare_dataset(batch_dict)
        # Append into the PPODataset's values dict. update_values_dict would
        # clobber the existing dict; patch in place to preserve everything
        # the base just wrote.
        self.dataset.values_dict["loss_mask"] = batch_dict["loss_mask"]

    def calc_gradients(self, input_dict):
        """Override of A2CAgent.calc_gradients — verbatim body except the
        critic per-sample loss is multiplied by loss_mask before .mean().
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

        # (B,) float in {0.0, 1.0}; aligned with values/returns by the
        # same swap_and_flatten01 pass that produced this minibatch.
        loss_mask_batch = input_dict["loss_mask"].reshape(-1).float()

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

            # Plain .mean() for actor, entropy, bound — identical to
            # a2c_continuous with rnn_masks=None. Critic is masked per-sample
            # so SDF-rejected cells contribute zero gradient, then .mean()
            # over the full minibatch keeps the per-sample 1/B scale.
            a_loss_red = a_loss.mean()
            if self.has_value_loss and c_loss.numel() > 1:
                c_loss_red = (c_loss * loss_mask_batch.unsqueeze(-1)).mean()
            else:
                c_loss_red = c_loss.mean()
            entropy_red = entropy.mean()
            b_loss_red = b_loss.mean()

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
            c_loss_red,
            entropy_red,
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss_red,
        )
