"""
Co-Training Experience Buffer with Dynamics Scoring

Wraps ExperienceBuffer for sim-real co-training. At rollout boundaries,
scores sim-env trajectory segments via a BaseScorer and resamples the
buffer to exclude low-quality segments.

The scorer, accept/reject threshold, and concrete sim-env indices are
passed directly to the constructor. The caller (the training experiment)
is responsible for pulling `idx_train_sim` from the env and forwarding it
here at construction time.
"""

from typing import Callable, Dict, List, Optional
import torch
from rl_games.common.experience import ExperienceBuffer, VectorizedReplayBuffer


class CotrainExperienceBuffer:
    """PPO experience buffer for sim-real co-training with dynamics scoring.

    Wraps a single ExperienceBuffer covering all envs. At rollout boundaries,
    extracts trajectory segments from the sim envs identified by
    `sim_env_idx`, scores them with the provided scorer, builds a
    per-timestep acceptance mask, and resamples accepted timesteps to fill
    the original batch size.

    Scoring is only performed when `scorer`, `threshold`, and `sim_env_idx`
    are all provided; when any is None, the buffer behaves as a plain
    pass-through ExperienceBuffer.
    """

    def __init__(
        self,
        env_info: dict,
        algo_info: dict,
        device: str,
        scorer=None,
        threshold: Optional[float] = None,
        sim_env_idx: Optional[torch.Tensor] = None,
        writer=None,
    ):
        """
        Args:
            env_info: RL Games env info dict.
            algo_info: RL Games algo info dict (num_actors, horizon_length, etc.).
            device: Torch device string.
            scorer: A BaseScorer (e.g. DynamicsScorer) exposing `.algo.horizon`,
                `.algo.future_horizon`, and `score(data, "mse") -> (B,)`.
                None disables scoring (no resampling performed).
            threshold: MSE threshold for binary accept/reject. Segments with
                score > threshold are rejected. Required whenever `scorer` is
                set.
            sim_env_idx: 1-D LongTensor of env indices (into the
                (T, num_envs, *) buffer) that should be dynamics-scored.
                Every other env is always accepted. Required whenever
                `scorer` is set.
            writer: Optional tensorboard SummaryWriter for logging.
        """
        self.buffer = ExperienceBuffer(env_info, algo_info, device)
        self.device = device
        self.writer = writer
        self._log_step = 0

        if scorer is not None:
            assert threshold is not None, "threshold is required when scorer is provided"
            assert sim_env_idx is not None, "sim_env_idx is required when scorer is provided"
            assert sim_env_idx.dim() == 1, (
                f"sim_env_idx must be 1-D, got shape {tuple(sim_env_idx.shape)}"
            )
            self.scorer = scorer
            self.threshold = float(threshold)
            self._horizon = scorer.algo.horizon
            self._future_horizon = scorer.algo.future_horizon
            self.sim_env_idx = sim_env_idx.to(self.device).long()

            # Pre-allocate the collect_obses buffer slot. ExperienceBuffer only
            # initializes standard RL Games keys (obses, actions, …); the env's
            # low-dim collect_obs stream that the scorer consumes is not one of
            # them, so `update_data("collect_obses", ...)` in play_steps would
            # otherwise hit a missing key. Shape mirrors obs_base_shape.
            T, N = self.buffer.obs_base_shape
            self.buffer.tensor_dict["collect_obses"] = torch.zeros(
                (T, N, scorer.algo.state_dim), dtype=torch.float32, device=self.device,
            )
        else:
            self.scorer = None
            self.threshold = None
            self._horizon = None
            self._future_horizon = None
            self.sim_env_idx = None

    @property
    def scoring_enabled(self) -> bool:
        return self.scorer is not None

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

        # Log scoring stats — accept rate is computed only over sim envs
        if self.writer is not None:
            accept_rate = mask.index_select(1, self.sim_env_idx).float().mean().item()
            self.writer.add_scalar("cotrain/sim_accept_rate", accept_rate, self._log_step)
            self._log_step += 1

        # Resample: for each tensor in buffer, replace rejected rows
        self._resample_buffer(td, mask, T, num_envs)

        return rnn_states_raw

    def _build_scoring_mask(self, td: dict, T: int, num_envs: int) -> torch.Tensor:
        """Build (T, num_envs) binary mask. Sim envs (identified by sim_env_idx)
        are scored; all other envs are always accepted (mask=1)."""
        mask = torch.ones(T, num_envs, device=self.device)

        num_sim = int(self.sim_env_idx.numel())
        if num_sim == 0:
            return mask

        # Gather sim-env data for scoring using the concrete env indices.
        # collect_obses: (T, num_envs, state_dim) -- stored during play_steps
        collect_obses = td["collect_obses"].index_select(1, self.sim_env_idx)  # (T, num_sim, D_s)
        actions = td["actions"].index_select(1, self.sim_env_idx)              # (T, num_sim, D_a)

        H = self._horizon
        F = self._future_horizon

        # Non-overlapping chunks: chunk i starts at t = i*F
        chunk_starts = list(range(0, T - H - F + 1, F))
        if len(chunk_starts) == 0:
            # Trajectory too short for any valid segment -- accept all
            return mask

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

        mask[:, self.sim_env_idx] = sim_mask
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


class CotrainVectorizedReplayBuffer:
    """Dual replay buffer for SAC co-training with per-episode score weighting.

    Accepts full ``(num_total_envs, ...)`` tensors on ``add()``. Rows for
    validation envs (the last ``num_total_envs - num_train_real - num_train_sim``
    rows) are dropped unconditionally; the remaining training rows are split
    contiguously into separate real and sim buffers:

        obs[0 : num_train_real]                        -> real_buf
        obs[num_train_real : num_train_real+num_train_sim] -> sim_buf
        obs[num_train_real+num_train_sim : ]           -> dropped (val envs)

    This contiguous assumption requires ``randomize_partition=False`` on the
    environment. When a scorer is provided, per-env episode scores are updated
    via ``traj_resampler.score_episode`` when a done fires, and sim transitions
    are sampled proportionally to their env's current score.
    """

    def __init__(
        self,
        obs_shape: tuple,
        action_shape: tuple,
        capacity: int,
        device,
        num_train_real: int,
        num_train_sim: int,
        num_total_envs: int,
        real_data_ratio: float = 0.3,
        traj_resampler=None,
        train_scorer: bool = False,
        writer=None,
    ):
        assert num_total_envs >= num_train_real + num_train_sim, (
            f"num_total_envs ({num_total_envs}) must be >= num_train_real+num_train_sim "
            f"({num_train_real + num_train_sim})"
        )

        self.device = device
        self.num_train_real = num_train_real
        self.num_train_sim = num_train_sim
        self.num_train = num_train_real + num_train_sim
        self.num_total_envs = num_total_envs
        self.real_data_ratio = real_data_ratio
        self.traj_resampler = traj_resampler
        self.train_scorer = train_scorer
        self.writer = writer
        self._log_step = 0

        # Proportional capacity split. When one side has no envs, its capacity is 0.
        if num_train_real > 0 and num_train_sim > 0:
            real_cap = max(1, int(capacity * real_data_ratio))
            sim_cap = capacity - real_cap
        elif num_train_sim == 0:
            real_cap, sim_cap = capacity, 0
        else:
            real_cap, sim_cap = 0, capacity

        self._real_buf = (
            VectorizedReplayBuffer(obs_shape, action_shape, real_cap, device)
            if real_cap > 0 and num_train_real > 0 else None
        )
        self._sim_buf = (
            VectorizedReplayBuffer(obs_shape, action_shape, sim_cap, device)
            if sim_cap > 0 and num_train_sim > 0 else None
        )

        # Per-slot env_id (position within [0, num_train_sim)) for sim weighted sampling.
        # Layout mirrors VectorizedReplayBuffer's write order so a slot's sim env id
        # matches the obs that was written there.
        self._sim_env_ids = (
            torch.zeros(sim_cap, dtype=torch.long, device=device)
            if sim_cap > 0 else None
        )
        # Per-env episode score; defaults to 1.0 until first episode completes.
        self._sim_scores = (
            torch.ones(num_train_sim, dtype=torch.float32, device=device)
            if num_train_sim > 0 else None
        )
        # Per-sim-env episode accumulators cleared on done.
        self._acc_obs: list = [[] for _ in range(num_train_sim)]
        self._acc_act: list = [[] for _ in range(num_train_sim)]

        print(
            f"[CotrainVectorizedReplayBuffer] train_real={num_train_real} "
            f"train_sim={num_train_sim} num_total_envs={num_total_envs} "
            f"real_cap={real_cap} sim_cap={sim_cap} "
            f"scorer={type(traj_resampler).__name__ if traj_resampler else 'None'}"
        )

    # SACAgent polls these for logging/compat with VectorizedReplayBuffer.
    @property
    def idx(self):
        if self._sim_buf is not None:
            return self._sim_buf.idx
        return self._real_buf.idx if self._real_buf is not None else 0

    @property
    def full(self):
        if self._sim_buf is not None:
            return self._sim_buf.full
        return self._real_buf.full if self._real_buf is not None else False

    def add(self, obs, action, reward, next_obs, done):
        """Add one step from all envs.

        Args:
            obs:      (num_total_envs, *obs_shape)
            action:   (num_total_envs, *action_shape)
            reward:   (num_total_envs, 1)
            next_obs: (num_total_envs, *obs_shape)
            done:     (num_total_envs, 1)
        """
        assert obs.shape[0] == self.num_total_envs, (
            f"expected obs.shape[0]={self.num_total_envs}, got {obs.shape[0]}"
        )

        real_end = self.num_train_real
        sim_end = self.num_train_real + self.num_train_sim

        if self.num_train_real > 0 and self._real_buf is not None:
            self._real_buf.add(
                obs[:real_end], action[:real_end], reward[:real_end],
                next_obs[:real_end], done[:real_end],
            )
        if self.num_train_sim > 0 and self._sim_buf is not None:
            self._add_sim(
                obs[real_end:sim_end], action[real_end:sim_end], reward[real_end:sim_end],
                next_obs[real_end:sim_end], done[real_end:sim_end],
            )
        # obs[sim_end:] are val envs -- intentionally dropped.

    def _add_sim(self, obs, action, reward, next_obs, done):
        num_sim = obs.shape[0]
        cap = self._sim_buf.capacity
        old_idx = self._sim_buf.idx
        remaining = min(cap - old_idx, num_sim)
        overflow = num_sim - remaining

        # Mirror VectorizedReplayBuffer's write layout to tag each slot with its sim env id.
        if remaining > 0:
            self._sim_env_ids[old_idx : old_idx + remaining] = torch.arange(
                remaining, device=self.device
            )
        if overflow > 0:
            self._sim_env_ids[0:overflow] = torch.arange(
                remaining, num_sim, device=self.device
            )

        self._sim_buf.add(obs, action, reward, next_obs, done)

        if self.traj_resampler is None:
            return

        done_flat = done.squeeze(-1)
        for i in range(num_sim):
            self._acc_obs[i].append(obs[i])
            self._acc_act[i].append(action[i])
            if done_flat[i].item():
                self._score_episode(i)

    def _score_episode(self, env_i: int):
        """Score completed sim episode and update per-env weight."""
        acc_obs = self._acc_obs[env_i]
        acc_act = self._acc_act[env_i]
        self._acc_obs[env_i] = []
        self._acc_act[env_i] = []

        if len(acc_obs) == 0:
            return

        states = torch.stack(acc_obs)
        actions = torch.stack(acc_act)
        with torch.no_grad():
            score = float(self.traj_resampler.score_episode(states, actions))

        self._sim_scores[env_i] = score
        if self.writer is not None:
            self.writer.add_scalar("cotrain/sim_score", score, self._log_step)
            self._log_step += 1

        if self.train_scorer:
            self.traj_resampler.scorer_train_step(
                {"obses": states.unsqueeze(1), "actions": actions.unsqueeze(1)}
            )

    def sample(self, batch_size: int):
        """Sample a combined batch: score-weighted sim + uniform real."""
        sim_size = (
            self._sim_buf.capacity if self._sim_buf.full else self._sim_buf.idx
        ) if self._sim_buf is not None else 0
        real_size = (
            self._real_buf.capacity if self._real_buf.full else self._real_buf.idx
        ) if self._real_buf is not None else 0

        if real_size > 0 and sim_size > 0:
            num_real = int(batch_size * self.real_data_ratio)
            num_sim = batch_size - num_real
        elif sim_size == 0:
            num_real, num_sim = batch_size, 0
        else:
            num_real, num_sim = 0, batch_size

        parts = []
        if num_sim > 0:
            if self.traj_resampler is not None:
                weights = self._sim_scores[self._sim_env_ids[:sim_size]].clamp(min=1e-6)
                sim_idx = torch.multinomial(weights, num_sim, replacement=True)
            else:
                sim_idx = torch.randint(0, sim_size, (num_sim,), device=self.device)
            parts.append((
                self._sim_buf.obses[sim_idx],
                self._sim_buf.actions[sim_idx],
                self._sim_buf.rewards[sim_idx],
                self._sim_buf.next_obses[sim_idx],
                self._sim_buf.dones[sim_idx],
            ))

        if num_real > 0:
            real_idx = torch.randint(0, real_size, (num_real,), device=self.device)
            parts.append((
                self._real_buf.obses[real_idx],
                self._real_buf.actions[real_idx],
                self._real_buf.rewards[real_idx],
                self._real_buf.next_obses[real_idx],
                self._real_buf.dones[real_idx],
            ))

        obses      = torch.cat([p[0] for p in parts])
        actions    = torch.cat([p[1] for p in parts])
        rewards    = torch.cat([p[2] for p in parts])
        next_obses = torch.cat([p[3] for p in parts])
        dones      = torch.cat([p[4] for p in parts])

        perm = torch.randperm(obses.shape[0], device=self.device)
        return obses[perm], actions[perm], rewards[perm], next_obses[perm], dones[perm]

    def get_stats(self) -> dict:
        stats = {"cotrain/real_data_ratio": self.real_data_ratio}
        if self._sim_buf is not None:
            stats["cotrain/sim_buffer_size"] = (
                self._sim_buf.capacity if self._sim_buf.full else self._sim_buf.idx
            )
        if self._real_buf is not None:
            stats["cotrain/real_buffer_size"] = (
                self._real_buf.capacity if self._real_buf.full else self._real_buf.idx
            )
        if self._sim_scores is not None:
            stats["cotrain/sim_score_mean"] = self._sim_scores.mean().item()
            stats["cotrain/sim_score_min"] = self._sim_scores.min().item()
            stats["cotrain/sim_score_max"] = self._sim_scores.max().item()
        return stats
