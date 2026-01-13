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

import torch
import numpy as np
from typing import Optional, Dict, Callable
from rl_games.common.experience import ExperienceBuffer


def swap_and_flatten01(arr):
    """Swap and flatten axes 0 and 1."""
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class DualExperienceBuffer:
    """
    Manages two experience buffers (real and sim) for co-training.

    The environment is expected to have a mix of real and sim envs,
    with sim envs in indices [0, num_sim_envs) and real envs in
    indices [num_sim_envs, num_total_envs).

    This follows IsaacLab's FlexHole pattern where large_env_fraction
    controls the fraction of sim (large hole) environments.
    """

    def __init__(
        self,
        env_info: dict,
        algo_info: dict,
        device: str,
        num_real_envs: int,
        num_total_envs: int,
        real_data_ratio: float = 0.3,
        dynamics_scorer=None,
        temperature: float = 1.0,
        aux_tensor_dict: dict = None,
    ):
        """
        Args:
            env_info: Environment info dict
            algo_info: Algorithm info dict containing horizon_length, etc.
            device: Torch device
            num_real_envs: Number of "real" (standard) environments
            num_total_envs: Total number of environments (num_actors)
            real_data_ratio: Desired fraction of real data in batch (alpha)
                            0.3 means 30% real, 70% sim
            dynamics_scorer: DynamicsScorerInterface for scoring sim trajectories
            temperature: Softmax temperature for sim weighting
                        Lower = sharper (favor high-scoring)
                        Higher = smoother (more uniform)
            aux_tensor_dict: Additional tensors to track
        """
        self.device = device
        self.num_real_envs = num_real_envs
        self.num_total_envs = num_total_envs
        self.num_sim_envs = num_total_envs - num_real_envs
        self.real_data_ratio = real_data_ratio
        self.sim_data_ratio = 1.0 - real_data_ratio
        self.dynamics_scorer = dynamics_scorer
        self.temperature = temperature
        self.horizon_length = algo_info['horizon_length']

        # Validate
        assert self.num_sim_envs + self.num_real_envs == self.num_total_envs
        assert 0 <= real_data_ratio <= 1

        # Create separate algo_info for each buffer
        real_algo_info = algo_info.copy()
        real_algo_info['num_actors'] = num_real_envs

        sim_algo_info = algo_info.copy()
        sim_algo_info['num_actors'] = self.num_sim_envs

        # Create two internal buffers
        self.real_buffer = ExperienceBuffer(env_info, real_algo_info, device, aux_tensor_dict)
        self.sim_buffer = ExperienceBuffer(env_info, sim_algo_info, device, aux_tensor_dict)

        # Reliability scores for sim data
        self.sim_reliability_scores = None
        self.sim_sampling_weights = None

        print(f"[DualExperienceBuffer] Initialized:")
        print(f"  Real envs: {num_real_envs} ({100*num_real_envs/num_total_envs:.1f}%)")
        print(f"  Sim envs: {self.num_sim_envs} ({100*self.num_sim_envs/num_total_envs:.1f}%)")
        print(f"  Target real ratio in batch: {100*real_data_ratio:.1f}%")
        print(f"  Scorer: {type(dynamics_scorer).__name__ if dynamics_scorer else 'None (uniform)'}")

    def update_data(self, name: str, index: int, val: torch.Tensor):
        """
        Update both buffers with data split by env index.

        Data is expected to be concatenated: [sim_envs, real_envs].
        This matches IsaacLab's FlexHole layout.

        Args:
            name: Key name (e.g., 'obses', 'actions', 'rewards')
            index: Time step index
            val: Values for all envs, shape (num_total_envs, ...)
        """
        if isinstance(val, dict):
            # Handle dict observations
            sim_val = {k: v[:self.num_sim_envs] for k, v in val.items()}
            real_val = {k: v[self.num_sim_envs:] for k, v in val.items()}
        else:
            # Sim envs come first (large hole), then real envs (standard)
            sim_val = val[:self.num_sim_envs]
            real_val = val[self.num_sim_envs:]

        self.sim_buffer.update_data(name, index, sim_val)
        self.real_buffer.update_data(name, index, real_val)

    def compute_sim_reliability(self) -> torch.Tensor:
        """
        Compute reliability scores for sim trajectories.

        Uses the dynamics_scorer to evaluate each sim trajectory.
        Scores are used for weighted sampling of sim data.

        Returns:
            (num_sim_envs,) tensor of reliability scores in [0, 1]
        """
        if self.dynamics_scorer is None:
            # No scorer - uniform weights
            self.sim_reliability_scores = torch.ones(
                self.num_sim_envs, device=self.device
            )
            return self.sim_reliability_scores

        # Extract trajectories from sim buffer
        # tensor_dict['obses']: (horizon, num_sim_envs, obs_dim) or dict
        obses = self.sim_buffer.tensor_dict['obses']
        actions = self.sim_buffer.tensor_dict['actions']

        # Preprocess for scoring
        states, actions_processed = self.dynamics_scorer.preprocess_for_scoring(
            obses, actions
        )

        # Transpose to (num_envs, horizon, dim) for scoring
        if isinstance(states, dict):
            states = {k: v.transpose(0, 1) for k, v in states.items()}
            # Use the state_key to get the actual state tensor
            state_key = self.dynamics_scorer.state_key
            if state_key in states:
                states_tensor = states[state_key]
            else:
                # Fall back to first key
                states_tensor = list(states.values())[0]
        else:
            states_tensor = states.transpose(0, 1)

        actions_transposed = actions_processed.transpose(0, 1)

        # Compute next_states (shifted by 1)
        next_states = torch.roll(states_tensor, -1, dims=1)

        # Score (exclude last step since next_state is invalid)
        with torch.no_grad():
            self.sim_reliability_scores = self.dynamics_scorer.score_trajectories(
                states_tensor[:, :-1],
                actions_transposed[:, :-1],
                next_states[:, :-1]
            )

        return self.sim_reliability_scores

    def compute_sampling_weights(self) -> torch.Tensor:
        """
        Compute per-trajectory sampling weights for sim data.

        Uses softmax with temperature scaling.

        Returns:
            (num_sim_envs,) normalized weights summing to 1
        """
        if self.sim_reliability_scores is None:
            self.compute_sim_reliability()

        scores = self.sim_reliability_scores

        # Apply temperature scaling
        if self.temperature != 1.0:
            scores = scores / self.temperature

        # Handle case where all scores are 0
        if scores.sum() < 1e-8:
            self.sim_sampling_weights = torch.ones_like(scores) / len(scores)
        else:
            self.sim_sampling_weights = torch.softmax(scores, dim=0)

        return self.sim_sampling_weights

    def get_combined_batch(
        self,
        transform_op: Callable,
        target_batch_size: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a combined batch with alpha-weighted real/sim sampling.

        Sampling distribution:
        - (alpha)% samples uniformly from real data
        - (1-alpha)% samples from sim data weighted by reliability scores

        This is the key method for co-training. The returned batch has
        the same format as standard PPO, so downstream training is unchanged.

        Args:
            transform_op: Transformation to apply (e.g., swap_and_flatten01)
            target_batch_size: Total samples to draw. If None, uses full buffer size.

        Returns:
            combined_dict: Dictionary with all PPO keys, transparently combined
        """
        # Ensure reliability scores are computed
        if self.sim_reliability_scores is None:
            self.compute_sim_reliability()
        if self.sim_sampling_weights is None:
            self.compute_sampling_weights()

        # Transform both buffers to flat format
        real_dict = self.real_buffer.get_transformed(transform_op)
        sim_dict = self.sim_buffer.get_transformed(transform_op)

        # Get sizes after flattening
        def get_batch_size(d):
            for k, v in d.items():
                if v is not None:
                    if isinstance(v, dict):
                        return list(v.values())[0].shape[0]
                    return v.shape[0]
            return 0

        real_size = get_batch_size(real_dict)
        sim_size = get_batch_size(sim_dict)

        # Determine batch size
        if target_batch_size is None:
            target_batch_size = real_size + sim_size

        # ============ ALPHA-WEIGHTED SAMPLING ============
        # Split batch according to real_data_ratio
        num_real_samples = int(target_batch_size * self.real_data_ratio)
        num_sim_samples = target_batch_size - num_real_samples

        # Sample from REAL: uniform
        if num_real_samples > 0 and real_size > 0:
            real_indices = torch.randint(0, real_size, (num_real_samples,), device=self.device)
        else:
            real_indices = torch.tensor([], dtype=torch.long, device=self.device)
            num_real_samples = 0
            num_sim_samples = target_batch_size

        # Sample from SIM: weighted by reliability scores
        if num_sim_samples > 0 and sim_size > 0:
            # Expand trajectory-level weights to sample-level
            samples_per_traj = sim_size // self.num_sim_envs
            sim_sample_probs = self.sim_sampling_weights.repeat_interleave(samples_per_traj)

            # Ensure it sums to 1 for multinomial
            sim_sample_probs = sim_sample_probs / sim_sample_probs.sum()

            sim_indices = torch.multinomial(sim_sample_probs, num_sim_samples, replacement=True)
        else:
            sim_indices = torch.tensor([], dtype=torch.long, device=self.device)

        # ============ BUILD COMBINED BATCH ============
        combined_dict = {}
        for key in real_dict.keys():
            if real_dict[key] is None:
                combined_dict[key] = None
                continue

            if isinstance(real_dict[key], dict):
                combined_dict[key] = {}
                for k in real_dict[key].keys():
                    real_samples = real_dict[key][k][real_indices] if len(real_indices) > 0 else torch.empty(0, *real_dict[key][k].shape[1:], device=self.device)
                    sim_samples = sim_dict[key][k][sim_indices] if len(sim_indices) > 0 else torch.empty(0, *sim_dict[key][k].shape[1:], device=self.device)
                    combined_dict[key][k] = torch.cat([real_samples, sim_samples], dim=0)
            else:
                real_samples = real_dict[key][real_indices] if len(real_indices) > 0 else torch.empty(0, *real_dict[key].shape[1:], device=self.device)
                sim_samples = sim_dict[key][sim_indices] if len(sim_indices) > 0 else torch.empty(0, *sim_dict[key].shape[1:], device=self.device)
                combined_dict[key] = torch.cat([real_samples, sim_samples], dim=0)

        # ============ SHUFFLE ============
        total_size = num_real_samples + num_sim_samples
        if total_size > 0:
            perm = torch.randperm(total_size, device=self.device)
            for key, val in combined_dict.items():
                if val is None:
                    continue
                if isinstance(val, dict):
                    combined_dict[key] = {k: v[perm] for k, v in val.items()}
                else:
                    combined_dict[key] = val[perm]

        return combined_dict

    def get_transformed(self, transform_op: Callable) -> Dict[str, torch.Tensor]:
        """
        Get transformed data using alpha-weighted sampling.

        This is the main interface called by play_steps().
        Replaces the standard ExperienceBuffer.get_transformed().
        """
        return self.get_combined_batch(transform_op)

    def get_transformed_list(
        self,
        transform_op: Callable,
        tensor_list: list,
    ) -> Dict[str, torch.Tensor]:
        """
        Get transformed data for specific keys with alpha-weighted sampling.

        This is the main interface called by play_steps().
        Replaces the standard ExperienceBuffer.get_transformed_list().
        """
        # Get full combined batch
        combined = self.get_combined_batch(transform_op)

        # Filter to requested keys
        return {k: combined.get(k) for k in tensor_list if k in combined}

    def get_reliability_stats(self) -> Dict[str, float]:
        """
        Get statistics about reliability scores for logging.

        Returns:
            dict with reliability statistics
        """
        if self.sim_reliability_scores is None:
            return {}

        scores = self.sim_reliability_scores
        stats = {
            'cotrain/reliability_mean': scores.mean().item(),
            'cotrain/reliability_std': scores.std().item(),
            'cotrain/reliability_min': scores.min().item(),
            'cotrain/reliability_max': scores.max().item(),
            'cotrain/real_data_ratio': self.real_data_ratio,
        }

        if self.sim_sampling_weights is not None:
            # Entropy of sampling distribution (higher = more uniform)
            weights = self.sim_sampling_weights
            entropy = -(weights * torch.log(weights + 1e-8)).sum().item()
            max_entropy = np.log(len(weights))
            stats['cotrain/sampling_entropy'] = entropy
            stats['cotrain/sampling_uniformity'] = entropy / max_entropy if max_entropy > 0 else 1.0

        return stats

    @property
    def tensor_dict(self) -> Dict[str, torch.Tensor]:
        """
        For compatibility: return combined tensor dict.

        Note: This returns concatenated data without weighted resampling.
        Used for computing advantages before get_combined_batch().
        """
        combined = {}
        for key in self.real_buffer.tensor_dict.keys():
            real_val = self.real_buffer.tensor_dict[key]
            sim_val = self.sim_buffer.tensor_dict[key]

            if real_val is None:
                combined[key] = None
            elif isinstance(real_val, dict):
                combined[key] = {
                    k: torch.cat([sim_val[k], real_val[k]], dim=1)
                    for k in real_val.keys()
                }
            else:
                # Concatenate along env dimension (dim=1)
                # Layout: [sim_envs, real_envs] to match update_data
                combined[key] = torch.cat([sim_val, real_val], dim=1)

        return combined

    def reset_scores(self):
        """Reset reliability scores for next epoch."""
        self.sim_reliability_scores = None
        self.sim_sampling_weights = None
