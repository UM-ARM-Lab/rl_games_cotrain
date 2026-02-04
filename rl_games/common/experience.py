import numpy as np
import random
import gym
import torch
from rl_games.common.segment_tree import SumSegmentTree, MinSegmentTree
import torch

from rl_games.algos_torch.torch_ext import numpy_to_torch_dtype_dict

class ReplayBuffer(object):
    def __init__(self, size, ob_space):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._obses = np.zeros((size,) + ob_space.shape, dtype=ob_space.dtype)
        self._next_obses = np.zeros((size,) + ob_space.shape, dtype=ob_space.dtype)
        self._rewards = np.zeros(size)
        self._actions = np.zeros(size, dtype=np.int32)
        self._dones = np.zeros(size, dtype=np.bool)

        self._maxsize = size
        self._next_idx = 0
        self._curr_size = 0

    def __len__(self):
        return self._curr_size

    def add(self, obs_t, action, reward, obs_tp1, done):

        self._curr_size = min(self._curr_size + 1, self._maxsize )

        self._obses[self._next_idx] = obs_t
        self._next_obses[self._next_idx] = obs_tp1
        self._rewards[self._next_idx] = reward
        self._actions[self._next_idx] = action
        self._dones[self._next_idx] = done

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _get(self, idx):
        return self._obses[idx], self._actions[idx], self._rewards[idx], self._next_obses[idx], self._dones[idx]

    def _encode_sample(self, idxes):
        batch_size = len(idxes)
        obses_t, actions, rewards, obses_tp1, dones = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size
        it = 0
        for i in idxes:
            data = self._get(i)
            obs_t, action, reward, obs_tp1, done = data
            obses_t[it] = np.array(obs_t, copy=False)
            actions[it] = np.array(action, copy=False)
            rewards[it] = reward
            obses_tp1[it] = np.array(obs_tp1, copy=False)
            dones[it] = done
            it = it + 1
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, self._curr_size - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, ob_space):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, ob_space)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self._curr_size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self._curr_size) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self._curr_size) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self._curr_size
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class VectorizedReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, device):
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device

        self.obses = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.next_obses = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=self.device)

        self.capacity = capacity
        self.idx = 0
        self.full = False


    def add(self, obs, action, reward, next_obs, done):

        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.obses[0: overflow] = obs[-overflow:]
            self.actions[0: overflow] = action[-overflow:]
            self.rewards[0: overflow] = reward[-overflow:]
            self.next_obses[0: overflow] = next_obs[-overflow:]
            self.dones[0: overflow] = done[-overflow:]
            self.full = True
        self.obses[self.idx: self.idx + remaining_capacity] = obs[:remaining_capacity]
        self.actions[self.idx: self.idx + remaining_capacity] = action[:remaining_capacity]
        self.rewards[self.idx: self.idx + remaining_capacity] = reward[:remaining_capacity]
        self.next_obses[self.idx: self.idx + remaining_capacity] = next_obs[:remaining_capacity]
        self.dones[self.idx: self.idx + remaining_capacity] = done[:remaining_capacity]

        self.idx = (self.idx + num_observations) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obses: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        not_dones_no_max: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not, specifically exlcuding maximum episode steps
        """

        idxs = torch.randint(0,
                            self.capacity if self.full else self.idx,
                            (batch_size,), device=self.device)
        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        dones = self.dones[idxs]

        return obses, actions, rewards, next_obses, dones


class CotrainVectorizedReplayBuffer:
    """
    Dual replay buffer for SAC co-training with trajectory-based reweighting.

    Manages two separate VectorizedReplayBuffer instances (sim and real) and
    provides trajectory-level scoring for weighted sampling from the sim buffer.

    This mirrors the PPO CotrainExperienceBuffer design while respecting SAC's
    off-policy requirements (circular buffer, batch scoring, weighted sampling).

    Environment layout:
        [0, num_sim_envs)           -> sim envs (scaled hole, easier)
        [num_sim_envs, num_total_envs) -> real envs (tight clearance)
    """

    def __init__(
        self,
        obs_shape: tuple,
        action_shape: tuple,
        capacity: int,
        device: torch.device,
        num_sim_envs: int,
        num_total_envs: int,
        real_data_ratio: float = 0.3,
        traj_resampler=None,
        score_batch_size: int = 16,
        train_scorer: bool = False,
        writer=None,
    ):
        """
        Args:
            obs_shape: Shape of observations
            action_shape: Shape of actions
            capacity: Total replay buffer capacity (split between sim/real)
            device: Torch device
            num_sim_envs: Number of sim environments (indices [0, num_sim_envs))
            num_total_envs: Total environments (num_sim_envs + num_real_envs)
            real_data_ratio: Fraction of batch from real buffer (0.3 = 30% real)
            traj_resampler: TrajResamplerInterface for scoring trajectories
            score_batch_size: Number of trajectories to accumulate before scoring
            train_scorer: If True, call scorer_train_step(); if False, inference only
            writer: SummaryWriter for logging
        """
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity

        # Env counts
        self.num_sim_envs = num_sim_envs
        self.num_real_envs = num_total_envs - num_sim_envs
        self.num_total_envs = num_total_envs
        self.real_data_ratio = real_data_ratio

        # Scorer
        self.traj_resampler = traj_resampler
        self.score_batch_size = score_batch_size
        self.train_scorer = train_scorer
        self.writer = writer
        self._log_step = 0

        # Allocate capacity proportionally
        if self.num_real_envs > 0 and self.num_sim_envs > 0:
            sim_capacity = int(capacity * (1 - real_data_ratio))
            real_capacity = capacity - sim_capacity
        elif self.num_real_envs == 0:
            sim_capacity = capacity
            real_capacity = 0
        else:
            sim_capacity = 0
            real_capacity = capacity

        # Create internal buffers
        self.sim_buffer = VectorizedReplayBuffer(
            obs_shape, action_shape, max(sim_capacity, 1), device
        ) if sim_capacity > 0 else None

        self.real_buffer = VectorizedReplayBuffer(
            obs_shape, action_shape, max(real_capacity, 1), device
        ) if real_capacity > 0 else None

        # Trajectory tracking for sim envs
        # Maps buffer index -> trajectory ID
        self.sim_traj_ids = torch.zeros(sim_capacity, dtype=torch.long, device=device) if sim_capacity > 0 else None
        self.sim_traj_scores = {}  # traj_id -> score
        self._next_traj_id = 0

        # Per-env current trajectory ID
        self._env_current_traj_id = torch.zeros(num_sim_envs, dtype=torch.long, device=device) if num_sim_envs > 0 else None

        # Trajectory accumulation for scoring
        self._pending_trajectories = []  # List of {obs, action, traj_id}
        self._env_traj_buffers = {i: {'obses': [], 'actions': []} for i in range(num_sim_envs)}

        # Initialize trajectory IDs for each env
        if num_sim_envs > 0:
            for i in range(num_sim_envs):
                self._env_current_traj_id[i] = self._next_traj_id
                self._next_traj_id += 1

        print(f"[CotrainVectorizedReplayBuffer] Initialized:")
        print(f"  Sim envs: {num_sim_envs}, Real envs: {self.num_real_envs}")
        print(f"  Sim capacity: {sim_capacity}, Real capacity: {real_capacity}")
        print(f"  Real data ratio: {100*real_data_ratio:.1f}%")
        print(f"  Scorer: {type(traj_resampler).__name__ if traj_resampler else 'None (uniform)'}")
        print(f"  Train scorer: {train_scorer}")

    @property
    def idx(self):
        """Current index for compatibility."""
        if self.sim_buffer is not None:
            return self.sim_buffer.idx
        elif self.real_buffer is not None:
            return self.real_buffer.idx
        return 0

    @property
    def full(self):
        """Whether buffer is full for compatibility."""
        if self.sim_buffer is not None:
            return self.sim_buffer.full
        elif self.real_buffer is not None:
            return self.real_buffer.full
        return False

    def add(self, obs, action, reward, next_obs, done):
        """
        Add transitions split by environment index.

        Args:
            obs: (num_total_envs, *obs_shape)
            action: (num_total_envs, *action_shape)
            reward: (num_total_envs, 1)
            next_obs: (num_total_envs, *obs_shape)
            done: (num_total_envs, 1)
        """
        # Split by env index: [sim][real]
        if self.num_sim_envs > 0 and self.sim_buffer is not None:
            sim_obs = obs[:self.num_sim_envs]
            sim_action = action[:self.num_sim_envs]
            sim_reward = reward[:self.num_sim_envs]
            sim_next_obs = next_obs[:self.num_sim_envs]
            sim_done = done[:self.num_sim_envs]

            # Track which buffer indices get which trajectory IDs
            self._add_sim_with_tracking(
                sim_obs, sim_action, sim_reward, sim_next_obs, sim_done
            )

        if self.num_real_envs > 0 and self.real_buffer is not None:
            real_obs = obs[self.num_sim_envs:]
            real_action = action[self.num_sim_envs:]
            real_reward = reward[self.num_sim_envs:]
            real_next_obs = next_obs[self.num_sim_envs:]
            real_done = done[self.num_sim_envs:]

            self.real_buffer.add(
                real_obs, real_action, real_reward, real_next_obs, real_done
            )

    def _add_sim_with_tracking(self, obs, action, reward, next_obs, done):
        """Add sim transitions with trajectory tracking and episode detection."""
        num_envs = obs.shape[0]
        sim_capacity = self.sim_buffer.capacity

        # Record starting index
        start_idx = self.sim_buffer.idx

        # Add to buffer
        self.sim_buffer.add(obs, action, reward, next_obs, done)

        # Update trajectory IDs for new entries
        for env_idx in range(num_envs):
            traj_id = self._env_current_traj_id[env_idx].item()
            buf_idx = (start_idx + env_idx) % sim_capacity
            self.sim_traj_ids[buf_idx] = traj_id

            # Accumulate trajectory data for scoring
            self._env_traj_buffers[env_idx]['obses'].append(obs[env_idx].unsqueeze(0))
            self._env_traj_buffers[env_idx]['actions'].append(action[env_idx].unsqueeze(0))

        # Check for episode completions
        done_envs = done.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
        for env_idx in done_envs.tolist() if done_envs.dim() > 0 else ([done_envs.item()] if done_envs.numel() == 1 else []):
            self._on_episode_complete(env_idx)

    def _on_episode_complete(self, env_idx: int):
        """Handle episode completion: queue trajectory for scoring."""
        traj_id = self._env_current_traj_id[env_idx].item()
        traj_data = self._env_traj_buffers[env_idx]

        if len(traj_data['obses']) > 0:
            # Stack trajectory
            traj_obses = torch.cat(traj_data['obses'], dim=0)  # (T, *obs_shape)
            traj_actions = torch.cat(traj_data['actions'], dim=0)  # (T, *action_shape)

            self._pending_trajectories.append({
                'obses': traj_obses,
                'actions': traj_actions,
                'traj_id': traj_id,
            })

        # Reset trajectory buffer for this env
        self._env_traj_buffers[env_idx] = {'obses': [], 'actions': []}

        # Assign new trajectory ID
        self._env_current_traj_id[env_idx] = self._next_traj_id
        self._next_traj_id += 1

        # Check if we should score
        if len(self._pending_trajectories) >= self.score_batch_size:
            self._score_pending_trajectories()

    def _score_pending_trajectories(self):
        """Score pending trajectories and update weights."""
        if not self._pending_trajectories or self.traj_resampler is None:
            # Clear pending and assign default scores
            for traj in self._pending_trajectories:
                self.sim_traj_scores[traj['traj_id']] = 1.0
            self._pending_trajectories = []
            return

        # Find max trajectory length for padding
        max_len = max(t['obses'].shape[0] for t in self._pending_trajectories)
        batch_size = len(self._pending_trajectories)

        # Pad and stack trajectories: (T, batch, ...)
        padded_obses = []
        padded_actions = []
        traj_lengths = []

        for traj in self._pending_trajectories:
            T = traj['obses'].shape[0]
            traj_lengths.append(T)

            # Pad to max_len
            if T < max_len:
                obs_pad = torch.zeros(
                    (max_len - T, *self.obs_shape),
                    device=self.device, dtype=traj['obses'].dtype
                )
                act_pad = torch.zeros(
                    (max_len - T, *self.action_shape),
                    device=self.device, dtype=traj['actions'].dtype
                )
                padded_obses.append(torch.cat([traj['obses'], obs_pad], dim=0))
                padded_actions.append(torch.cat([traj['actions'], act_pad], dim=0))
            else:
                padded_obses.append(traj['obses'])
                padded_actions.append(traj['actions'])

        # Stack: (batch, T, ...) -> (T, batch, ...)
        obses_batch = torch.stack(padded_obses, dim=0).transpose(0, 1)  # (T, batch, *obs_shape)
        actions_batch = torch.stack(padded_actions, dim=0).transpose(0, 1)  # (T, batch, *action_shape)

        # Create tensor_dict for scorer
        tensor_dict = {
            'obses': obses_batch,
            'actions': actions_batch,
        }

        # Optional: train scorer
        if self.train_scorer:
            scorer_train_step = getattr(self.traj_resampler, 'scorer_train_step', None)
            if callable(scorer_train_step):
                result = scorer_train_step(tensor_dict)
                if result is not None and self.writer is not None:
                    loss, metrics = result
                    for key, value in metrics.items():
                        self.writer.add_scalar(f'scorer/{key}', value, self._log_step)

        # Get resampling mask
        with torch.no_grad():
            mask = self.traj_resampler.resample(tensor_dict)

        # mask shape: (num_envs, T) or (T, num_envs)
        if mask.shape[0] == batch_size and mask.shape[1] == max_len:
            # (batch, T) -> keep as is
            pass
        elif mask.shape[0] == max_len and mask.shape[1] == batch_size:
            # (T, batch) -> transpose
            mask = mask.transpose(0, 1)

        # Convert mask to per-trajectory score (fraction of steps kept)
        for i, traj in enumerate(self._pending_trajectories):
            traj_id = traj['traj_id']
            T = traj_lengths[i]

            # Score = fraction of valid steps kept (accounting for padding)
            valid_mask = mask[i, :T]
            score = valid_mask.float().mean().item()
            score = max(score, 0.01)  # Minimum score to avoid zero weights

            self.sim_traj_scores[traj_id] = score

        # Log stats
        if self.writer is not None:
            scores = [self.sim_traj_scores[t['traj_id']] for t in self._pending_trajectories]
            self.writer.add_scalar('cotrain/traj_score_mean', np.mean(scores), self._log_step)
            self.writer.add_scalar('cotrain/traj_score_std', np.std(scores), self._log_step)
            self.writer.add_scalar('cotrain/traj_score_min', np.min(scores), self._log_step)
            self.writer.add_scalar('cotrain/traj_score_max', np.max(scores), self._log_step)
            self.writer.add_scalar('cotrain/num_scored_trajs', len(self._pending_trajectories), self._log_step)
            self._log_step += 1

        # Clear pending
        self._pending_trajectories = []

        # Cleanup stale trajectory scores (older than buffer capacity)
        self._cleanup_stale_scores()

    def _cleanup_stale_scores(self):
        """Remove trajectory scores for IDs no longer in the buffer."""
        if self.sim_buffer is None or not self.sim_buffer.full:
            return

        # Get unique trajectory IDs currently in buffer
        current_traj_ids = set(self.sim_traj_ids.unique().tolist())

        # Remove scores for trajectories no longer in buffer
        stale_ids = [tid for tid in self.sim_traj_scores.keys() if tid not in current_traj_ids]
        for tid in stale_ids:
            del self.sim_traj_scores[tid]

    def _compute_transition_weights(self) -> torch.Tensor:
        """Compute per-transition weights from trajectory scores."""
        if self.sim_buffer is None:
            return torch.tensor([], device=self.device)

        sim_size = self.sim_buffer.capacity if self.sim_buffer.full else self.sim_buffer.idx
        if sim_size == 0:
            return torch.tensor([], device=self.device)

        weights = torch.ones(sim_size, device=self.device)

        for idx in range(sim_size):
            traj_id = self.sim_traj_ids[idx].item()
            if traj_id in self.sim_traj_scores:
                weights[idx] = self.sim_traj_scores[traj_id]
            # else: default weight = 1.0 (unscored trajectory)

        # Normalize
        weights = weights / weights.sum()
        return weights

    def sample(self, batch_size):
        """
        Sample a combined batch with alpha-weighted real/sim sampling.

        Returns:
            obses, actions, rewards, next_obses, dones - same as VectorizedReplayBuffer
        """
        # Determine batch split
        num_real_samples = int(batch_size * self.real_data_ratio)
        num_sim_samples = batch_size - num_real_samples

        # Get buffer sizes
        sim_size = 0
        real_size = 0
        if self.sim_buffer is not None:
            sim_size = self.sim_buffer.capacity if self.sim_buffer.full else self.sim_buffer.idx
        if self.real_buffer is not None:
            real_size = self.real_buffer.capacity if self.real_buffer.full else self.real_buffer.idx

        # Handle edge cases
        if real_size == 0:
            num_sim_samples = batch_size
            num_real_samples = 0
        if sim_size == 0:
            num_real_samples = batch_size
            num_sim_samples = 0

        samples = []

        # Sample from SIM: weighted by trajectory scores
        if num_sim_samples > 0 and sim_size > 0:
            if self.traj_resampler is not None and self.sim_traj_scores:
                # Weighted sampling
                weights = self._compute_transition_weights()
                sim_indices = torch.multinomial(weights, num_sim_samples, replacement=True)
            else:
                # Uniform sampling
                sim_indices = torch.randint(0, sim_size, (num_sim_samples,), device=self.device)

            sim_obses = self.sim_buffer.obses[sim_indices]
            sim_actions = self.sim_buffer.actions[sim_indices]
            sim_rewards = self.sim_buffer.rewards[sim_indices]
            sim_next_obses = self.sim_buffer.next_obses[sim_indices]
            sim_dones = self.sim_buffer.dones[sim_indices]

            samples.append((sim_obses, sim_actions, sim_rewards, sim_next_obses, sim_dones))

        # Sample from REAL: uniform
        if num_real_samples > 0 and real_size > 0:
            real_indices = torch.randint(0, real_size, (num_real_samples,), device=self.device)

            real_obses = self.real_buffer.obses[real_indices]
            real_actions = self.real_buffer.actions[real_indices]
            real_rewards = self.real_buffer.rewards[real_indices]
            real_next_obses = self.real_buffer.next_obses[real_indices]
            real_dones = self.real_buffer.dones[real_indices]

            samples.append((real_obses, real_actions, real_rewards, real_next_obses, real_dones))

        # Combine samples
        if len(samples) == 0:
            # Empty buffer case
            return (
                torch.empty((0, *self.obs_shape), device=self.device),
                torch.empty((0, *self.action_shape), device=self.device),
                torch.empty((0, 1), device=self.device),
                torch.empty((0, *self.obs_shape), device=self.device),
                torch.empty((0, 1), dtype=torch.bool, device=self.device),
            )

        obses = torch.cat([s[0] for s in samples], dim=0)
        actions = torch.cat([s[1] for s in samples], dim=0)
        rewards = torch.cat([s[2] for s in samples], dim=0)
        next_obses = torch.cat([s[3] for s in samples], dim=0)
        dones = torch.cat([s[4] for s in samples], dim=0)

        # Shuffle combined batch
        perm = torch.randperm(obses.shape[0], device=self.device)
        obses = obses[perm]
        actions = actions[perm]
        rewards = rewards[perm]
        next_obses = next_obses[perm]
        dones = dones[perm]

        return obses, actions, rewards, next_obses, dones

    def get_stats(self) -> dict:
        """Get co-training statistics for logging."""
        stats = {
            'cotrain/real_data_ratio': self.real_data_ratio,
            'cotrain/num_scored_trajs': len(self.sim_traj_scores),
        }

        if self.sim_buffer is not None:
            sim_size = self.sim_buffer.capacity if self.sim_buffer.full else self.sim_buffer.idx
            stats['cotrain/sim_buffer_size'] = sim_size

        if self.real_buffer is not None:
            real_size = self.real_buffer.capacity if self.real_buffer.full else self.real_buffer.idx
            stats['cotrain/real_buffer_size'] = real_size

        if self.sim_traj_scores:
            scores = list(self.sim_traj_scores.values())
            stats['cotrain/score_mean'] = np.mean(scores)
            stats['cotrain/score_std'] = np.std(scores)
            stats['cotrain/score_min'] = np.min(scores)
            stats['cotrain/score_max'] = np.max(scores)

        return stats


class ExperienceBuffer:
    '''
    More generalized than replay buffers.
    Implemented for on-policy algos
    '''
    def __init__(self, env_info, algo_info, device, aux_tensor_dict=None):
        self.env_info = env_info
        self.algo_info = algo_info
        self.device = device

        self.num_agents = env_info.get('agents', 1)
        self.action_space = env_info['action_space']

        self.num_actors = algo_info['num_actors']
        self.horizon_length = algo_info['horizon_length']
        self.has_central_value = algo_info['has_central_value']
        self.use_action_masks = algo_info.get('use_action_masks', False)
        batch_size = self.num_actors * self.num_agents
        self.is_discrete = False
        self.is_multi_discrete = False
        self.is_continuous = False
        self.obs_base_shape = (self.horizon_length, self.num_agents * self.num_actors)
        self.state_base_shape = (self.horizon_length, self.num_actors)
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_shape = ()
            self.actions_num = self.action_space.n
            self.is_discrete = True
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_shape = (len(self.action_space),)
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        if type(self.action_space) is gym.spaces.Box:
            self.actions_shape = (self.action_space.shape[0],)
            self.actions_num = self.action_space.shape[0]
            self.is_continuous = True
        self.tensor_dict = {}
        self._init_from_env_info(self.env_info)

        self.aux_tensor_dict = aux_tensor_dict
        if self.aux_tensor_dict is not None:
            self._init_from_aux_dict(self.aux_tensor_dict)

    def _init_from_env_info(self, env_info):
        obs_base_shape = self.obs_base_shape
        state_base_shape = self.state_base_shape

        self.tensor_dict['obses'] = self._create_tensor_from_space(env_info['observation_space'], obs_base_shape)
        if self.has_central_value:
            self.tensor_dict['states'] = self._create_tensor_from_space(env_info['state_space'], state_base_shape)

        val_space = gym.spaces.Box(low=0, high=1,shape=(env_info.get('value_size',1),))
        self.tensor_dict['rewards'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['values'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['neglogpacs'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.float32), obs_base_shape)
        self.tensor_dict['dones'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.uint8), obs_base_shape)

        if self.is_discrete or self.is_multi_discrete:
            self.tensor_dict['actions'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=int), obs_base_shape)
        if self.use_action_masks:
            self.tensor_dict['action_masks'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape + (np.sum(self.actions_num),), dtype=np.bool), obs_base_shape)
        if self.is_continuous:
            self.tensor_dict['actions'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)
            self.tensor_dict['mus'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)
            self.tensor_dict['sigmas'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)

    def _init_from_aux_dict(self, tensor_dict):
        obs_base_shape = self.obs_base_shape
        for k,v in tensor_dict.items():
            self.tensor_dict[k] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(v), dtype=np.float32), obs_base_shape)

    def _create_tensor_from_space(self, space, base_shape):
        if type(space) is gym.spaces.Box:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape + space.shape, dtype= dtype, device = self.device)
        if type(space) is gym.spaces.Discrete:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape, dtype= dtype, device = self.device)
        if type(space) is gym.spaces.Tuple:
            '''
            assuming that tuple is only Discrete tuple
            '''
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            tuple_len = len(space)
            return torch.zeros(base_shape +(tuple_len,), dtype= dtype, device = self.device)
        if type(space) is gym.spaces.Dict:
            t_dict = {}
            for k,v in space.spaces.items():
                t_dict[k] = self._create_tensor_from_space(v, base_shape)
            return t_dict

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k,v in val.items():
                self.tensor_dict[name][k][index,:] = v
        else:
            self.tensor_dict[name][index,:] = val


    def update_data_rnn(self, name, indices,play_mask, val):
        if type(val) is dict:
            for k,v in val:
                self.tensor_dict[name][k][indices,play_mask] = v
        else:
            self.tensor_dict[name][indices,play_mask] = val

    def get_transformed(self, transform_op):
        res_dict = {}
        for k, v in self.tensor_dict.items():
            if type(v) is dict:
                transformed_dict = {}
                for kd,vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)

        return res_dict

    def get_transformed_list(self, transform_op, tensor_list):
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            if type(v) is dict:
                transformed_dict = {}
                for kd,vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)

        return res_dict
