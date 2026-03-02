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
    def score_episode(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """
        Score a single completed episode for SAC replay buffer weighting.

        Called once per episode per sim environment when the done flag fires.
        The score weights this episode's transitions during replay sampling.

        Args:
            states:  (T, state_dim) float tensor of observed states.
            actions: (T, action_dim) float tensor of actions taken.

        Returns:
            score in [0, 1]. 1.0 = fully accept, 0.0 = reject.
            Unfit scorers must return 1.0.
        """
        raise NotImplementedError

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
        "sac": RLGPPOTrajResampler,
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


class CotrainVectorizedReplayBuffer:
    """
    Dual replay buffer for SAC co-training with per-episode score weighting.

    Wraps two VectorizedReplayBuffer instances (sim and real).
    Per-env episode scores are updated when done flags fire via scorer.score_episode().
    Sim transitions are sampled proportionally to their env's current score.

    Environment layout:
        [0, num_sim_envs)               -> sim envs
        [num_sim_envs, num_total_envs)  -> real envs
    """

    def __init__(
        self,
        obs_shape: tuple,
        action_shape: tuple,
        capacity: int,
        device,
        num_sim_envs: int,
        num_total_envs: int,
        real_data_ratio: float = 0.3,
        traj_resampler=None,
        score_batch_size: int = 16,  # kept for API compat, unused
        train_scorer: bool = False,
        writer=None,
    ):
        from rl_games.common.experience import VectorizedReplayBuffer

        self.device = device
        self.num_sim_envs = num_sim_envs
        self.num_real_envs = num_total_envs - num_sim_envs
        self.real_data_ratio = real_data_ratio
        self.traj_resampler = traj_resampler
        self.train_scorer = train_scorer
        self.writer = writer
        self._log_step = 0

        # Proportional capacity split
        ns, nr = num_sim_envs, self.num_real_envs
        sim_cap = max(1, int(capacity * (1 - real_data_ratio))) if ns > 0 else 0
        real_cap = (capacity - sim_cap) if nr > 0 else 0

        self._sim_buf = VectorizedReplayBuffer(obs_shape, action_shape, sim_cap, device) if sim_cap > 0 else None
        self._real_buf = VectorizedReplayBuffer(obs_shape, action_shape, real_cap, device) if real_cap > 0 else None

        # Per-slot env_id for weighted sim sampling (mirrors VectorizedReplayBuffer write layout)
        self._sim_env_ids = torch.zeros(sim_cap, dtype=torch.long, device=device) if sim_cap > 0 else None
        # Per-env score; default 1.0 until first episode completes
        self._sim_scores = torch.ones(ns, dtype=torch.float32, device=device) if ns > 0 else None

        # Per-env episode accumulators (cleared on done)
        self._acc_obs: list = [[] for _ in range(ns)]
        self._acc_act: list = [[] for _ in range(ns)]

        print(
            f"[CotrainVectorizedReplayBuffer] sim_envs={ns} real_envs={nr} "
            f"sim_cap={sim_cap} real_cap={real_cap} "
            f"scorer={type(traj_resampler).__name__ if traj_resampler else 'None'}"
        )

    # compat properties used by SACAgent
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
        """
        Add one step from all envs.

        Args:
            obs:      (num_total_envs, *obs_shape)
            action:   (num_total_envs, *action_shape)
            reward:   (num_total_envs, 1)
            next_obs: (num_total_envs, *obs_shape)
            done:     (num_total_envs, 1)
        """
        ns = self.num_sim_envs
        if ns > 0 and self._sim_buf is not None:
            self._add_sim(obs[:ns], action[:ns], reward[:ns], next_obs[:ns], done[:ns])
        if self.num_real_envs > 0 and self._real_buf is not None:
            self._real_buf.add(obs[ns:], action[ns:], reward[ns:], next_obs[ns:], done[ns:])

    def _add_sim(self, obs, action, reward, next_obs, done):
        ns = obs.shape[0]
        cap = self._sim_buf.capacity
        old_idx = self._sim_buf.idx
        remaining = min(cap - old_idx, ns)
        overflow = ns - remaining

        # Mirror VectorizedReplayBuffer's write layout to track env_id per slot
        if remaining > 0:
            self._sim_env_ids[old_idx : old_idx + remaining] = torch.arange(remaining, device=self.device)
        if overflow > 0:
            self._sim_env_ids[0:overflow] = torch.arange(remaining, ns, device=self.device)

        self._sim_buf.add(obs, action, reward, next_obs, done)

        # Accumulate per-env and score on episode end
        done_flat = done.squeeze(-1)
        for i in range(ns):
            self._acc_obs[i].append(obs[i])
            self._acc_act[i].append(action[i])
            if done_flat[i].item():
                self._score_episode(i)

    def _score_episode(self, env_i: int):
        """Score completed episode, update sim_scores[env_i], reset accumulators."""
        acc_obs = self._acc_obs[env_i]
        acc_act = self._acc_act[env_i]
        self._acc_obs[env_i] = []
        self._acc_act[env_i] = []

        if self.traj_resampler is None or len(acc_obs) == 0:
            return

        states = torch.stack(acc_obs)   # (T, obs_dim)
        actions = torch.stack(acc_act)  # (T, action_dim)
        with torch.no_grad():
            score = float(self.traj_resampler.score_episode(states, actions))

        self._sim_scores[env_i] = score
        if self.writer is not None:
            self.writer.add_scalar("cotrain/sim_score", score, self._log_step)
            self._log_step += 1

        if self.train_scorer:
            self.traj_resampler.scorer_train_step({"obses": states.unsqueeze(1), "actions": actions.unsqueeze(1)})

    def sample(self, batch_size: int):
        """
        Sample a combined batch: score-weighted sim + uniform real.

        Returns:
            obses, actions, rewards, next_obses, dones
        """
        sim_size = (self._sim_buf.capacity if self._sim_buf.full else self._sim_buf.idx) if self._sim_buf else 0
        real_size = (self._real_buf.capacity if self._real_buf.full else self._real_buf.idx) if self._real_buf else 0

        nr = int(batch_size * self.real_data_ratio) if real_size > 0 else 0
        ns_batch = (batch_size - nr) if sim_size > 0 else 0
        nr = batch_size - ns_batch  # recalculate in case sim is empty

        parts = []
        if ns_batch > 0:
            weights = self._sim_scores[self._sim_env_ids[:sim_size]].clamp(min=1e-6)
            sim_idx = torch.multinomial(weights, ns_batch, replacement=True)
            parts.append((
                self._sim_buf.obses[sim_idx],
                self._sim_buf.actions[sim_idx],
                self._sim_buf.rewards[sim_idx],
                self._sim_buf.next_obses[sim_idx],
                self._sim_buf.dones[sim_idx],
            ))

        if nr > 0:
            real_idx = torch.randint(0, real_size, (nr,), device=self.device)
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
