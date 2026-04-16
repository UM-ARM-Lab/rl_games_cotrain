"""
Co-Training Experience Buffer with Dynamics Scoring

Wraps ExperienceBuffer for sim-real co-training. At rollout boundaries,
scores sim-env trajectory segments via a BaseScorer and resamples the
buffer to exclude low-quality segments.

Env layout: [sim_envs | real_envs]
"""

from typing import Callable, Dict, List, Optional
import torch
from rl_games.common.experience import ExperienceBuffer


class CotrainExperienceBuffer:
    """PPO experience buffer for sim-real co-training with dynamics scoring.

    Wraps a single ExperienceBuffer covering all envs (sim first, then real).
    At rollout boundaries, extracts trajectory segments from sim envs, scores
    them with the provided scorer, builds a per-timestep acceptance mask, and
    resamples accepted timesteps to fill the original batch size.

    The scorer is optional. When None (or threshold is None), the buffer
    behaves identically to a standard ExperienceBuffer -- no scoring or
    resampling is performed.
    """

    def __init__(
        self,
        env_info: dict,
        algo_info: dict,
        device: str,
        num_sim_envs: int,
        num_real_envs: int,
        scorer=None,
        threshold: Optional[float] = None,
        writer=None,
    ):
        """
        Args:
            env_info: RL Games env info dict.
            algo_info: RL Games algo info dict (num_actors, horizon_length, etc.).
            device: Torch device string.
            num_sim_envs: Number of sim envs (first columns in buffer).
            num_real_envs: Number of real envs (last columns in buffer).
            scorer: A BaseScorer instance (e.g. DynamicsScorer) or None.
                    Must support scorer.score(data, "mse") -> (B,) tensor.
                    Scorer is inference-only -- not trained online during RL.
                    # TODO: may train scorer online in the future.
            threshold: MSE threshold for binary accept/reject. Segments with
                       score > threshold are rejected. None disables scoring.
            writer: Optional tensorboard SummaryWriter for logging.
        """
        self.buffer = ExperienceBuffer(env_info, algo_info, device)
        self.device = device
        self.num_sim_envs = num_sim_envs
        self.num_real_envs = num_real_envs
        self.scorer = scorer
        self.threshold = threshold
        self.writer = writer
        self._log_step = 0

        # Cache scorer params if available
        if self.scorer is not None:
            self._horizon = self.scorer.algo.horizon
            self._future_horizon = self.scorer.algo.future_horizon

    @property
    def scoring_enabled(self) -> bool:
        return self.scorer is not None and self.threshold is not None

    # ------------------------------------------------------------------
    # Data update pass-throughs
    # ------------------------------------------------------------------

    def update_data(self, name: str, index: int, val: torch.Tensor):
        self.buffer.update_data(name, index, val)

    def update_data_rnn(self, name, indices, play_mask, val):
        self.buffer.update_data_rnn(name, indices, play_mask, val)

    def update_data_full(self, name: str, val: torch.Tensor):
        self.buffer.tensor_dict[name] = val

    # ------------------------------------------------------------------
    # Scoring + Resampling
    # ------------------------------------------------------------------

    def resample(
        self,
        rnn_states_raw: Optional[List[torch.Tensor]] = None,
        seq_length: int = 1,
    ) -> Optional[List[torch.Tensor]]:
        """Score sim-env segments and resample the buffer.

        When scoring is disabled, this is a no-op (buffer unchanged).

        Args:
            rnn_states_raw: Not used for scoring -- passed through unchanged.
                RNN resampling is not yet supported with dynamics scoring.
            seq_length: Not used -- kept for API compat with play_steps_rnn.

        Returns:
            rnn_states_raw unchanged (RNN resampling not yet implemented).
        """
        if not self.scoring_enabled:
            return rnn_states_raw

        td = self.buffer.tensor_dict
        T = td["dones"].shape[0]
        num_envs = td["dones"].shape[1]

        # Build per-timestep mask (T, num_envs): 1 = accept, 0 = reject
        mask = self._build_scoring_mask(td, T, num_envs)

        # Log scoring stats
        if self.writer is not None:
            accept_rate = mask[:, :self.num_sim_envs].float().mean().item()
            self.writer.add_scalar("cotrain/sim_accept_rate", accept_rate, self._log_step)
            self._log_step += 1

        # Resample: for each tensor in buffer, replace rejected rows
        self._resample_buffer(td, mask, T, num_envs)

        return rnn_states_raw

    def _build_scoring_mask(self, td: dict, T: int, num_envs: int) -> torch.Tensor:
        """Build (T, num_envs) binary mask. Sim envs scored, real envs always 1."""
        mask = torch.ones(T, num_envs, device=self.device)

        if self.num_sim_envs == 0:
            return mask

        # Extract sim-env data for scoring
        # collect_obses: (T, num_envs, state_dim) -- stored during play_steps
        collect_obses = td["collect_obses"][:, :self.num_sim_envs]  # (T, num_sim, D_s)
        actions = td["actions"][:, :self.num_sim_envs]              # (T, num_sim, D_a)

        H = self._horizon
        F = self._future_horizon

        # Non-overlapping chunks: chunk i starts at t = i*F
        chunk_starts = list(range(0, T - H - F + 1, F))
        if len(chunk_starts) == 0:
            # Trajectory too short for any valid segment -- accept all
            return mask

        num_sim = self.num_sim_envs
        num_chunks = len(chunk_starts)

        # Build batched scorer input: (num_chunks * num_sim, H, D_s) etc.
        all_state_0 = []
        all_actions = []
        all_state_gt = []
        for t in chunk_starts:
            all_state_0.append(collect_obses[t:t + H])          # (H, num_sim, D_s)
            all_actions.append(actions[t + H:t + H + F])         # (F, num_sim, D_a)
            all_state_gt.append(collect_obses[t + H:t + H + F]) # (F, num_sim, D_s)

        # Stack: (num_chunks, H/F, num_sim, D) -> reshape to (num_chunks*num_sim, H/F, D)
        state_0 = torch.stack(all_state_0).permute(0, 2, 1, 3).reshape(num_chunks * num_sim, H, -1)
        future_actions = torch.stack(all_actions).permute(0, 2, 1, 3).reshape(num_chunks * num_sim, F, -1)
        state_gt = torch.stack(all_state_gt).permute(0, 2, 1, 3).reshape(num_chunks * num_sim, F, -1)

        # Score all segments
        data = {"state_0": state_0, "actions": future_actions, "state_gt": state_gt}
        scores = self.scorer.score(data, mode="mse")  # (num_chunks * num_sim,)
        scores = scores.reshape(num_chunks, num_sim)   # (num_chunks, num_sim)

        # Binary mask per chunk: 1 if score <= threshold
        chunk_accept = (scores <= self.threshold).float()  # (num_chunks, num_sim)

        # Expand chunk mask to per-timestep mask for sim envs
        sim_mask = torch.ones(T, num_sim, device=self.device)
        for i, t in enumerate(chunk_starts):
            # The scored region is [t+H, t+H+F) -- the future window
            sim_mask[t + H:t + H + F] = chunk_accept[i].unsqueeze(0).expand(F, -1)

        mask[:, :self.num_sim_envs] = sim_mask
        return mask

    def _resample_buffer(self, td: dict, mask: torch.Tensor, T: int, num_envs: int):
        """Resample buffer in-place: replace rejected timesteps with accepted ones.

        After swap_and_flatten01, PPO expects (T*num_envs,) flat tensors.
        We flatten the mask, find accepted indices, and sample with replacement
        to fill the original batch size.
        """
        flat_mask = mask.reshape(-1)  # (T * num_envs,)
        accepted_idx = flat_mask.nonzero(as_tuple=True)[0]

        if accepted_idx.numel() == 0:
            # Degenerate: all rejected -- keep buffer unchanged
            return

        total = T * num_envs
        if accepted_idx.numel() == total:
            # All accepted -- no resampling needed
            return

        # Sample with replacement from accepted indices
        sample_idx = accepted_idx[torch.randint(0, accepted_idx.numel(), (total,), device=self.device)]

        # Apply resampling to all tensor fields in the buffer
        for key, val in td.items():
            if not isinstance(val, torch.Tensor):
                continue
            if val.shape[0] != T or (val.ndim >= 2 and val.shape[1] != num_envs):
                continue
            # Flatten (T, num_envs, *) -> (T*num_envs, *), resample, reshape back
            flat_shape = (total, *val.shape[2:])
            td[key] = val.reshape(flat_shape)[sample_idx].reshape(val.shape)

    # ------------------------------------------------------------------
    # Transform pass-throughs (used by play_steps after resample)
    # ------------------------------------------------------------------

    def get_transformed(self, transform_op: Callable) -> Dict[str, torch.Tensor]:
        def _apply(v):
            if v is None:
                return None
            if isinstance(v, dict):
                return {kd: transform_op(vd) for kd, vd in v.items()}
            return transform_op(v)
        return {k: _apply(v) for k, v in self.buffer.tensor_dict.items()}

    def get_transformed_list(
        self,
        transform_op: Callable,
        tensor_list: list,
    ) -> Dict[str, torch.Tensor]:
        combined = self.get_transformed(transform_op)
        return {k: combined.get(k) for k in tensor_list if k in combined}

    @property
    def tensor_dict(self) -> Dict[str, torch.Tensor]:
        return self.buffer.tensor_dict
