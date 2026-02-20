"""
Dual Experience Buffer for Sim-Real Co-Training

This module provides a dual-buffer architecture for PPO that manages
real and sim data separately, with weighted sampling for training.

Key features:
- Simultaneous collection from real and sim environments
- Reliability scoring of sim trajectories using DynamicsScorerInterface
- Alpha-weighted sampling: (alpha)% real + (1-alpha)% sim (weighted by score)
- Transparent interface - downstream PPO sees standard batch format
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from omegaconf import DictConfig, OmegaConf
from rl_games.common.experience import ExperienceBuffer


class RLGPPOTrajResampler(ABC):
    """
    Abstract interface for dynamics models that score trajectory reliability.

    Users implement this interface with their specific scoring method.
    The scorer evaluates trajectory segments and produces reliability scores
    used to weight simulated data during PPO training.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the scorer with configuration.

        Args:
            cfg: Configuration dict for the scorer, can include model paths,
                 hyperparameters, etc.
        """
        self.cfg = cfg

    @abstractmethod
    def resample(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        rnn_states_raw: Optional[List[torch.Tensor]],
        seq_length: int,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Resample the full combined rollout at the sequence-block level.

        Args:
            tensor_dict: Combined sim+real rollout data. Tensor values have shape
                (T, num_envs, *) where T = num_seqs * seq_length.
            rnn_states_raw: Per-sequence initial RNN states. Each element has shape
                (num_seqs, num_layers, num_envs, hidden_size). None for non-RNN callers.
            seq_length: Length of each indivisible sequence block. T must be divisible
                by seq_length. seq_length == T gives env-level resampling (num_seqs=1).

        Returns:
            Tuple of:
              - Resampled tensor dict. Tensor values have shape
                (num_seqs, seq_length, num_envs, *). Temporal order within each
                seq_length block is preserved. Non-tensor values passed through unchanged.
              - Resampled rnn_states_raw with the same per-element shape as input.
                None if rnn_states_raw was None.
        """
        raise NotImplementedError("resample() must be implemented by subclass.")

    @abstractmethod
    def scorer_train_step(self, tensor_dict) -> Optional[Tuple[float, dict]]:
        """
        Optional training/update step for the scorer.

        Called after each rollout collection in the RL loop.
        - For trainable scorers (DynamicsScorer): performs gradient update
        - For non-trainable (DTWScorer): may update prototypes or statistics

        Args:
            tensor_dict: Dictionary containing trajectory data with shape (T, num_envs, ...):
                - 'obses': observations (dict or tensor)
                - 'actions': (T, num_envs, action_dim)
                - 'rewards': (T, num_envs, value_size)
                - 'dones': (T, num_envs)

        Returns:
            None, or tuple of (loss, metrics_dict) for logging
        """
        raise NotImplementedError("scorer_train_step is optional and can be left unimplemented if not needed.")


class TrajResamplerRegistry:
    """Minimal registry for config-driven resampler spawning."""

    registry: Dict[str, type] = {}

    # Hard-coded algo -> interface mapping.
    ALGO_INTERFACE_MAP: Dict[str, Optional[type]] = {
        "ppo": RLGPPOTrajResampler,
        "sac": None,
    }

    @classmethod
    def register(cls, name: str, impl_class: type):
        """Register a PPO trajectory resampler implementation by name."""
        if not issubclass(impl_class, RLGPPOTrajResampler):
            raise TypeError(f"Registered class {impl_class.__name__!r} must inherit RLGPPOTrajResampler.")
        cls.registry[name] = impl_class

    @classmethod
    def factory(cls, cfg: DictConfig, algo_name: str):
        """Instantiate a registered resampler from config for a target algorithm."""
        if cfg is None:
            return None

        interface_cls = cls.ALGO_INTERFACE_MAP.get(algo_name)
        if algo_name not in cls.ALGO_INTERFACE_MAP:
            raise ValueError(f"Unknown algo_name={algo_name!r}. Available: {list(cls.ALGO_INTERFACE_MAP.keys())}")
        if interface_cls is None:
            raise ValueError(f"No TrajResampler interface configured for algo_name={algo_name!r}.")

        class_name = cfg.class_name
        if class_name is None:
            raise ValueError("traj_resampler/scorer_cfg must include 'class_name'.")

        impl_class = cls.registry.get(class_name)
        if impl_class is None:
            raise ValueError(f"Unknown class_name={class_name!r}. Available: {list(cls.registry.keys())}")
        if not issubclass(impl_class, interface_cls):
            raise TypeError(
                f"Class {impl_class.__name__!r} is not compatible with algo={algo_name!r} "
                f"(required interface: {interface_cls.__name__!r})."
            )

        return impl_class(cfg)


class CotrainExperienceBuffer:
    """
    Thin PPO experience buffer for sim-real co-training.

    Wraps a single ExperienceBuffer covering all envs (sim first, then real).
    At rollout boundaries, trains the scorer and delegates all resampling to it.
    The resampler owns the sim/real partition logic; this buffer is agnostic to it.
    """

    def __init__(
        self,
        env_info: dict,
        algo_info: dict,
        device: str,
        num_real_envs: int,
        num_sim_envs: int,
        scorer_cfg: Union[dict, DictConfig],
        writer=None,
    ):
        self.traj_resampler = TrajResamplerRegistry.factory(OmegaConf.create(scorer_cfg), algo_name="ppo")
        assert self.traj_resampler is not None
        self.buffer = ExperienceBuffer(env_info, algo_info, device)
        self.writer = writer
        self._scorer_trained = False
        self._log_step = 0

    def update_data(self, name: str, index: int, val: torch.Tensor):
        self._scorer_trained = False
        self.buffer.update_data(name, index, val)

    def update_data_rnn(self, name, indices, play_mask, val):
        self._scorer_trained = False
        self.buffer.update_data_rnn(name, indices, play_mask, val)

    def update_data_full(self, name: str, val: torch.Tensor):
        """set the data tensor for re-sampling"""
        self._scorer_trained = False
        self.buffer.tensor_dict[name] = val

    def resample(
        self,
        rnn_states_raw: Optional[List[torch.Tensor]] = None,
        seq_length: int = 1,
    ) -> Optional[List[torch.Tensor]]:
        """Train scorer and resample the rollout buffer.

        Args:
            rnn_states_raw: Per-sequence initial RNN states, each element shaped
                (num_seqs, num_layers, num_envs, hidden_size). None for non-RNN.
            seq_length: Sequence block length passed through to the resampler.

        Returns:
            Resampled rnn_states_raw with the same per-element shape, or None.
        """
        self._scorer_trained = True
        train_result = self.traj_resampler.scorer_train_step(self.buffer.tensor_dict)
        if train_result is not None:
            _, metrics = train_result
            if self.writer is not None and metrics:
                for key, value in metrics.items():
                    self.writer.add_scalar(key, value, self._log_step)
            self._log_step += 1

        resampled_td, resampled_rnn = self.traj_resampler.resample(
            self.buffer.tensor_dict, rnn_states_raw, seq_length
        )
        assert isinstance(resampled_td, dict)

        # Flatten (num_seqs, seq_length, num_envs, *) → (T, num_envs, *) before storing.
        T = self.buffer.tensor_dict["dones"].shape[0]
        num_seqs = T // seq_length
        for key, v in resampled_td.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 3 and v.shape[0] == num_seqs and v.shape[1] == seq_length:
                self.buffer.tensor_dict[key] = v.reshape(T, *v.shape[2:])
            else:
                self.buffer.tensor_dict[key] = v

        return resampled_rnn

    def get_transformed(self, transform_op: Callable) -> Dict[str, torch.Tensor]:
        """Train scorer if needed, resample, and apply transform_op."""
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
        """Raw combined rollout data, used by PPO for value/advantage computation."""
        return self.buffer.tensor_dict


# class CotrainVectorizedReplayBuffer:
#     """
#     Dual replay buffer for SAC co-training with trajectory-based reweighting.

#     Manages two separate VectorizedReplayBuffer instances (sim and real) and
#     provides trajectory-level scoring for weighted sampling from the sim buffer.

#     This mirrors the PPO CotrainExperienceBuffer design while respecting SAC's
#     off-policy requirements (circular buffer, batch scoring, weighted sampling).

#     Environment layout:
#         [0, num_sim_envs)           -> sim envs (scaled hole, easier)
#         [num_sim_envs, num_total_envs) -> real envs (tight clearance)
#     """

#     def __init__(
#         self,
#         obs_shape: tuple,
#         action_shape: tuple,
#         capacity: int,
#         device: torch.device,
#         num_sim_envs: int,
#         num_total_envs: int,
#         real_data_ratio: float = 0.3,
#         traj_resampler=None,
#         score_batch_size: int = 16,
#         train_scorer: bool = False,
#         writer=None,
#     ):
#         """
#         Args:
#             obs_shape: Shape of observations
#             action_shape: Shape of actions
#             capacity: Total replay buffer capacity (split between sim/real)
#             device: Torch device
#             num_sim_envs: Number of sim environments (indices [0, num_sim_envs))
#             num_total_envs: Total environments (num_sim_envs + num_real_envs)
#             real_data_ratio: Fraction of batch from real buffer (0.3 = 30% real)
#             traj_resampler: RLGPPOTrajResampler for scoring trajectories
#             score_batch_size: Number of trajectories to accumulate before scoring
#             train_scorer: If True, call scorer_train_step(); if False, inference only
#             writer: SummaryWriter for logging
#         """
#         self.device = device
#         self.obs_shape = obs_shape
#         self.action_shape = action_shape
#         self.capacity = capacity

#         # Env counts
#         self.num_sim_envs = num_sim_envs
#         self.num_real_envs = num_total_envs - num_sim_envs
#         self.num_total_envs = num_total_envs
#         self.real_data_ratio = real_data_ratio

#         # Scorer
#         self.traj_resampler = traj_resampler
#         self.score_batch_size = score_batch_size
#         self.train_scorer = train_scorer
#         self.writer = writer
#         self._log_step = 0

#         # Allocate capacity proportionally
#         if self.num_real_envs > 0 and self.num_sim_envs > 0:
#             sim_capacity = int(capacity * (1 - real_data_ratio))
#             real_capacity = capacity - sim_capacity
#         elif self.num_real_envs == 0:
#             sim_capacity = capacity
#             real_capacity = 0
#         else:
#             sim_capacity = 0
#             real_capacity = capacity

#         # Create internal buffers
#         self.sim_buffer = (
#             VectorizedReplayBuffer(obs_shape, action_shape, max(sim_capacity, 1), device) if sim_capacity > 0 else None
#         )

#         self.real_buffer = (
#             VectorizedReplayBuffer(obs_shape, action_shape, max(real_capacity, 1), device)
#             if real_capacity > 0
#             else None
#         )

#         # Trajectory tracking for sim envs
#         # Maps buffer index -> trajectory ID
#         self.sim_traj_ids = torch.zeros(sim_capacity, dtype=torch.long, device=device) if sim_capacity > 0 else None
#         self.sim_traj_scores = {}  # traj_id -> score
#         self._next_traj_id = 0

#         # Per-env current trajectory ID
#         self._env_current_traj_id = (
#             torch.zeros(num_sim_envs, dtype=torch.long, device=device) if num_sim_envs > 0 else None
#         )

#         # Trajectory accumulation for scoring
#         self._pending_trajectories = []  # List of {obs, action, traj_id}
#         self._env_traj_buffers = {i: {"obses": [], "actions": []} for i in range(num_sim_envs)}

#         # Initialize trajectory IDs for each env
#         if num_sim_envs > 0:
#             for i in range(num_sim_envs):
#                 self._env_current_traj_id[i] = self._next_traj_id
#                 self._next_traj_id += 1

#         print("[CotrainVectorizedReplayBuffer] Initialized:")
#         print(f"  Sim envs: {num_sim_envs}, Real envs: {self.num_real_envs}")
#         print(f"  Sim capacity: {sim_capacity}, Real capacity: {real_capacity}")
#         print(f"  Real data ratio: {100 * real_data_ratio:.1f}%")
#         print(f"  Scorer: {type(traj_resampler).__name__ if traj_resampler else 'None (uniform)'}")
#         print(f"  Train scorer: {train_scorer}")

#     @property
#     def idx(self):
#         """Current index for compatibility."""
#         if self.sim_buffer is not None:
#             return self.sim_buffer.idx
#         elif self.real_buffer is not None:
#             return self.real_buffer.idx
#         return 0

#     @property
#     def full(self):
#         """Whether buffer is full for compatibility."""
#         if self.sim_buffer is not None:
#             return self.sim_buffer.full
#         elif self.real_buffer is not None:
#             return self.real_buffer.full
#         return False

#     def add(self, obs, action, reward, next_obs, done):
#         """
#         Add transitions split by environment index.

#         Args:
#             obs: (num_total_envs, *obs_shape)
#             action: (num_total_envs, *action_shape)
#             reward: (num_total_envs, 1)
#             next_obs: (num_total_envs, *obs_shape)
#             done: (num_total_envs, 1)
#         """
#         # Split by env index: [sim][real]
#         if self.num_sim_envs > 0 and self.sim_buffer is not None:
#             sim_obs = obs[: self.num_sim_envs]
#             sim_action = action[: self.num_sim_envs]
#             sim_reward = reward[: self.num_sim_envs]
#             sim_next_obs = next_obs[: self.num_sim_envs]
#             sim_done = done[: self.num_sim_envs]

#             # Track which buffer indices get which trajectory IDs
#             self._add_sim_with_tracking(sim_obs, sim_action, sim_reward, sim_next_obs, sim_done)

#         if self.num_real_envs > 0 and self.real_buffer is not None:
#             real_obs = obs[self.num_sim_envs :]
#             real_action = action[self.num_sim_envs :]
#             real_reward = reward[self.num_sim_envs :]
#             real_next_obs = next_obs[self.num_sim_envs :]
#             real_done = done[self.num_sim_envs :]

#             self.real_buffer.add(real_obs, real_action, real_reward, real_next_obs, real_done)

#     def _add_sim_with_tracking(self, obs, action, reward, next_obs, done):
#         """Add sim transitions with trajectory tracking and episode detection.
#         Called from `add` for the sim slice of each environment step."""
#         num_envs = obs.shape[0]
#         sim_capacity = self.sim_buffer.capacity

#         # Record starting index
#         start_idx = self.sim_buffer.idx

#         # Add to buffer
#         self.sim_buffer.add(obs, action, reward, next_obs, done)

#         # Update trajectory IDs for new entries
#         for env_idx in range(num_envs):
#             traj_id = self._env_current_traj_id[env_idx].item()
#             buf_idx = (start_idx + env_idx) % sim_capacity
#             self.sim_traj_ids[buf_idx] = traj_id

#             # Accumulate trajectory data for scoring
#             self._env_traj_buffers[env_idx]["obses"].append(obs[env_idx].unsqueeze(0))
#             self._env_traj_buffers[env_idx]["actions"].append(action[env_idx].unsqueeze(0))

#         # Check for episode completions
#         done_envs = done.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
#         for env_idx in (
#             done_envs.tolist() if done_envs.dim() > 0 else ([done_envs.item()] if done_envs.numel() == 1 else [])
#         ):
#             self._on_episode_complete(env_idx)

#     def _on_episode_complete(self, env_idx: int):
#         """Handle episode completion: queue trajectory for scoring.
#         Triggered by `_add_sim_with_tracking` when a sim env emits done."""
#         traj_id = self._env_current_traj_id[env_idx].item()
#         traj_data = self._env_traj_buffers[env_idx]

#         if len(traj_data["obses"]) > 0:
#             # Stack trajectory
#             traj_obses = torch.cat(traj_data["obses"], dim=0)  # (T, *obs_shape)
#             traj_actions = torch.cat(traj_data["actions"], dim=0)  # (T, *action_shape)

#             self._pending_trajectories.append(
#                 {
#                     "obses": traj_obses,
#                     "actions": traj_actions,
#                     "traj_id": traj_id,
#                 }
#             )

#         # Reset trajectory buffer for this env
#         self._env_traj_buffers[env_idx] = {"obses": [], "actions": []}

#         # Assign new trajectory ID
#         self._env_current_traj_id[env_idx] = self._next_traj_id
#         self._next_traj_id += 1

#         # Check if we should score
#         if len(self._pending_trajectories) >= self.score_batch_size:
#             self._score_pending_trajectories()

#     def _score_pending_trajectories(self):
#         """Score pending trajectories and update weights.
#         Called from `_on_episode_complete` when pending count reaches threshold."""
#         if not self._pending_trajectories or self.traj_resampler is None:
#             # Clear pending and assign default scores
#             for traj in self._pending_trajectories:
#                 self.sim_traj_scores[traj["traj_id"]] = 1.0
#             self._pending_trajectories = []
#             return

#         # Find max trajectory length for padding
#         max_len = max(t["obses"].shape[0] for t in self._pending_trajectories)
#         batch_size = len(self._pending_trajectories)

#         # Pad and stack trajectories: (T, batch, ...)
#         padded_obses = []
#         padded_actions = []
#         traj_lengths = []

#         for traj in self._pending_trajectories:
#             T = traj["obses"].shape[0]
#             traj_lengths.append(T)

#             # Pad to max_len
#             if T < max_len:
#                 obs_pad = torch.zeros((max_len - T, *self.obs_shape), device=self.device, dtype=traj["obses"].dtype)
#                 act_pad = torch.zeros(
#                     (max_len - T, *self.action_shape), device=self.device, dtype=traj["actions"].dtype
#                 )
#                 padded_obses.append(torch.cat([traj["obses"], obs_pad], dim=0))
#                 padded_actions.append(torch.cat([traj["actions"], act_pad], dim=0))
#             else:
#                 padded_obses.append(traj["obses"])
#                 padded_actions.append(traj["actions"])

#         # Stack: (batch, T, ...) -> (T, batch, ...)
#         obses_batch = torch.stack(padded_obses, dim=0).transpose(0, 1)  # (T, batch, *obs_shape)
#         actions_batch = torch.stack(padded_actions, dim=0).transpose(0, 1)  # (T, batch, *action_shape)

#         # Create tensor_dict for scorer
#         tensor_dict = {
#             "obses": obses_batch,
#             "actions": actions_batch,
#         }

#         # Optional: train scorer
#         if self.train_scorer:
#             scorer_train_step = getattr(self.traj_resampler, "scorer_train_step", None)
#             if callable(scorer_train_step):
#                 result = scorer_train_step(tensor_dict)
#                 if result is not None and self.writer is not None:
#                     loss, metrics = result
#                     for key, value in metrics.items():
#                         self.writer.add_scalar(f"scorer/{key}", value, self._log_step)

#         # Get resampling mask
#         with torch.no_grad():
#             mask = self.traj_resampler.resample(tensor_dict)

#         # mask shape: (num_envs, T) or (T, num_envs)
#         if mask.shape[0] == batch_size and mask.shape[1] == max_len:
#             # (batch, T) -> keep as is
#             pass
#         elif mask.shape[0] == max_len and mask.shape[1] == batch_size:
#             # (T, batch) -> transpose
#             mask = mask.transpose(0, 1)

#         # Convert mask to per-trajectory score (fraction of steps kept)
#         for i, traj in enumerate(self._pending_trajectories):
#             traj_id = traj["traj_id"]
#             T = traj_lengths[i]

#             # Score = fraction of valid steps kept (accounting for padding)
#             valid_mask = mask[i, :T]
#             score = valid_mask.float().mean().item()
#             score = max(score, 0.01)  # Minimum score to avoid zero weights

#             self.sim_traj_scores[traj_id] = score

#         # Log stats
#         if self.writer is not None:
#             scores = [self.sim_traj_scores[t["traj_id"]] for t in self._pending_trajectories]
#             self.writer.add_scalar("cotrain/traj_score_mean", np.mean(scores), self._log_step)
#             self.writer.add_scalar("cotrain/traj_score_std", np.std(scores), self._log_step)
#             self.writer.add_scalar("cotrain/traj_score_min", np.min(scores), self._log_step)
#             self.writer.add_scalar("cotrain/traj_score_max", np.max(scores), self._log_step)
#             self.writer.add_scalar("cotrain/num_scored_trajs", len(self._pending_trajectories), self._log_step)
#             self._log_step += 1

#         # Clear pending
#         self._pending_trajectories = []

#         # Cleanup stale trajectory scores (older than buffer capacity)
#         self._cleanup_stale_scores()

#     def _cleanup_stale_scores(self):
#         """Remove trajectory scores for IDs no longer in the buffer.
#         Invoked by `_score_pending_trajectories` after score updates."""
#         if self.sim_buffer is None or not self.sim_buffer.full:
#             return

#         # Get unique trajectory IDs currently in buffer
#         current_traj_ids = set(self.sim_traj_ids.unique().tolist())

#         # Remove scores for trajectories no longer in buffer
#         stale_ids = [tid for tid in self.sim_traj_scores.keys() if tid not in current_traj_ids]
#         for tid in stale_ids:
#             del self.sim_traj_scores[tid]

#     def _compute_transition_weights(self) -> torch.Tensor:
#         """Compute per-transition weights from trajectory scores.
#         Used by `sample` when weighted sim replay sampling is enabled."""
#         if self.sim_buffer is None:
#             return torch.tensor([], device=self.device)

#         sim_size = self.sim_buffer.capacity if self.sim_buffer.full else self.sim_buffer.idx
#         if sim_size == 0:
#             return torch.tensor([], device=self.device)

#         weights = torch.ones(sim_size, device=self.device)

#         for idx in range(sim_size):
#             traj_id = self.sim_traj_ids[idx].item()
#             if traj_id in self.sim_traj_scores:
#                 weights[idx] = self.sim_traj_scores[traj_id]
#             # else: default weight = 1.0 (unscored trajectory)

#         # Normalize
#         weights = weights / weights.sum()
#         return weights

#     def sample(self, batch_size):
#         """
#         Sample a combined batch with alpha-weighted real/sim sampling.

#         Returns:
#             obses, actions, rewards, next_obses, dones - same as VectorizedReplayBuffer
#         """
#         # Determine batch split
#         num_real_samples = int(batch_size * self.real_data_ratio)
#         num_sim_samples = batch_size - num_real_samples

#         # Get buffer sizes
#         sim_size = 0
#         real_size = 0
#         if self.sim_buffer is not None:
#             sim_size = self.sim_buffer.capacity if self.sim_buffer.full else self.sim_buffer.idx
#         if self.real_buffer is not None:
#             real_size = self.real_buffer.capacity if self.real_buffer.full else self.real_buffer.idx

#         # Handle edge cases
#         if real_size == 0:
#             num_sim_samples = batch_size
#             num_real_samples = 0
#         if sim_size == 0:
#             num_real_samples = batch_size
#             num_sim_samples = 0

#         samples = []

#         # Sample from SIM: weighted by trajectory scores
#         if num_sim_samples > 0 and sim_size > 0:
#             if self.traj_resampler is not None and self.sim_traj_scores:
#                 # Weighted sampling
#                 weights = self._compute_transition_weights()
#                 sim_indices = torch.multinomial(weights, num_sim_samples, replacement=True)
#             else:
#                 # Uniform sampling
#                 sim_indices = torch.randint(0, sim_size, (num_sim_samples,), device=self.device)

#             sim_obses = self.sim_buffer.obses[sim_indices]
#             sim_actions = self.sim_buffer.actions[sim_indices]
#             sim_rewards = self.sim_buffer.rewards[sim_indices]
#             sim_next_obses = self.sim_buffer.next_obses[sim_indices]
#             sim_dones = self.sim_buffer.dones[sim_indices]

#             samples.append((sim_obses, sim_actions, sim_rewards, sim_next_obses, sim_dones))

#         # Sample from REAL: uniform
#         if num_real_samples > 0 and real_size > 0:
#             real_indices = torch.randint(0, real_size, (num_real_samples,), device=self.device)

#             real_obses = self.real_buffer.obses[real_indices]
#             real_actions = self.real_buffer.actions[real_indices]
#             real_rewards = self.real_buffer.rewards[real_indices]
#             real_next_obses = self.real_buffer.next_obses[real_indices]
#             real_dones = self.real_buffer.dones[real_indices]

#             samples.append((real_obses, real_actions, real_rewards, real_next_obses, real_dones))

#         # Combine samples
#         if len(samples) == 0:
#             # Empty buffer case
#             return (
#                 torch.empty((0, *self.obs_shape), device=self.device),
#                 torch.empty((0, *self.action_shape), device=self.device),
#                 torch.empty((0, 1), device=self.device),
#                 torch.empty((0, *self.obs_shape), device=self.device),
#                 torch.empty((0, 1), dtype=torch.bool, device=self.device),
#             )

#         obses = torch.cat([s[0] for s in samples], dim=0)
#         actions = torch.cat([s[1] for s in samples], dim=0)
#         rewards = torch.cat([s[2] for s in samples], dim=0)
#         next_obses = torch.cat([s[3] for s in samples], dim=0)
#         dones = torch.cat([s[4] for s in samples], dim=0)

#         # Shuffle combined batch
#         perm = torch.randperm(obses.shape[0], device=self.device)
#         obses = obses[perm]
#         actions = actions[perm]
#         rewards = rewards[perm]
#         next_obses = next_obses[perm]
#         dones = dones[perm]

#         return obses, actions, rewards, next_obses, dones

#     def get_stats(self) -> dict:
#         """Get co-training statistics for logging."""
#         stats = {
#             "cotrain/real_data_ratio": self.real_data_ratio,
#             "cotrain/num_scored_trajs": len(self.sim_traj_scores),
#         }

#         if self.sim_buffer is not None:
#             sim_size = self.sim_buffer.capacity if self.sim_buffer.full else self.sim_buffer.idx
#             stats["cotrain/sim_buffer_size"] = sim_size

#         if self.real_buffer is not None:
#             real_size = self.real_buffer.capacity if self.real_buffer.full else self.real_buffer.idx
#             stats["cotrain/real_buffer_size"] = real_size

#         if self.sim_traj_scores:
#             scores = list(self.sim_traj_scores.values())
#             stats["cotrain/score_mean"] = np.mean(scores)
#             stats["cotrain/score_std"] = np.std(scores)
#             stats["cotrain/score_min"] = np.min(scores)
#             stats["cotrain/score_max"] = np.max(scores)

#         return stats
