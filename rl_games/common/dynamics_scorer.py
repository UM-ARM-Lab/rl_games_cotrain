"""
Dynamics Scorer Interface for Sim-Real Co-Training

This module provides an abstract interface for scoring trajectory reliability,
used to weight simulated data during PPO training.

The interface is designed to be model-agnostic - users can implement it with:
- Dynamics model (prediction error based)
- DTW scorer (trajectory similarity based)
- Any other trajectory quality metric
"""

from abc import ABC, abstractmethod
import torch
from typing import Union, Tuple


class DynamicsScorerInterface(ABC):
    """
    Abstract interface for dynamics models that score trajectory reliability.

    Users implement this interface with their specific scoring method.
    The scorer evaluates trajectory segments and produces reliability scores
    used to weight simulated data during PPO training.
    """

    @abstractmethod
    def score_trajectories(
        self,
        states: torch.Tensor,      # (B, T, state_dim)
        actions: torch.Tensor,     # (B, T, action_dim)
        next_states: torch.Tensor, # (B, T, state_dim)
    ) -> torch.Tensor:
        """
        Compute reliability scores for trajectories.

        Args:
            states: Batch of state trajectories, shape (B, T, state_dim)
            actions: Batch of action trajectories, shape (B, T, action_dim)
            next_states: Batch of next state trajectories, shape (B, T, state_dim)

        Returns:
            scores: (B,) tensor of reliability scores in [0, 1]
                   Higher = more reliable/realistic trajectory
        """
        pass

    @property
    @abstractmethod
    def state_key(self) -> str:
        """
        Key to extract state from obs_dict for scoring.
        Example: 'low_dim_state', 'proprio', 'robot_state'
        """
        pass

    @property
    def action_key(self) -> str:
        """Key to extract action. Usually 'actions'."""
        return 'actions'

    def preprocess_for_scoring(
        self,
        obs_dict: Union[dict, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess observations and actions before scoring.

        Default implementation extracts state using self.state_key.
        Override if you need custom preprocessing.

        Args:
            obs_dict: Observation dictionary or tensor from buffer
            actions: Actions tensor

        Returns:
            states: Preprocessed states for scoring
            actions: Preprocessed actions for scoring
        """
        if isinstance(obs_dict, dict):
            states = obs_dict[self.state_key]
        else:
            states = obs_dict
        return states, actions


class DTWScorerAdapter(DynamicsScorerInterface):
    """
    Adapter to use DTWScorerGPU as a DynamicsScorerInterface.

    This adapter wraps the DTW-based trajectory scorer to provide
    reliability scores for the co-training framework.

    The DTW scorer evaluates trajectory segments by comparing them
    to learned prototypes from in-distribution data. Segments that
    are similar to prototypes get high scores; OOD segments get low scores.
    """

    def __init__(
        self,
        dtw_scorer,  # DTWScorerGPU instance
        state_key: str = 'low_dim_state',
        state_dims: list = None,
        score_method: str = 'threshold',  # 'threshold' or 'distance'
    ):
        """
        Args:
            dtw_scorer: DTWScorerGPU instance (from algorithms.common.dtw_scorer_gpu)
            state_key: Key to extract state from obs_dict
            state_dims: Which dimensions to use for DTW (default: [0,1,2] for xyz)
            score_method: How to convert DTW distance to score:
                - 'threshold': 1.0 if distance <= threshold, else 0.0
                - 'distance': exp(-distance / threshold) for smooth scoring
        """
        self.dtw_scorer = dtw_scorer
        self._state_key = state_key
        self.state_dims = state_dims if state_dims is not None else [0, 1, 2]
        self.score_method = score_method

    @property
    def state_key(self) -> str:
        return self._state_key

    def score_trajectories(
        self,
        states: torch.Tensor,      # (B, T, state_dim)
        actions: torch.Tensor,     # (B, T, action_dim)
        next_states: torch.Tensor, # (B, T, state_dim)
    ) -> torch.Tensor:
        """
        Score trajectories using DTW distance to prototypes.

        Args:
            states: (B, T, state_dim) trajectory states
            actions: (B, T, action_dim) - not used by DTW but kept for interface
            next_states: (B, T, state_dim) - not used, DTW uses states directly

        Returns:
            scores: (B,) reliability scores in [0, 1]
        """
        B, T, D = states.shape
        device = states.device

        # Extract relevant dimensions for DTW
        if D > len(self.state_dims):
            states_xyz = states[:, :, self.state_dims]
        else:
            states_xyz = states

        # Check if trajectory length matches DTW chunk length
        chunk_length = self.dtw_scorer.chunk_length

        if T == chunk_length:
            # Direct scoring - trajectory is exactly one chunk
            distances = self.dtw_scorer.compute_distances_batch(states_xyz)
        elif T > chunk_length:
            # Score multiple chunks and aggregate
            # Use mean distance across all chunks in the trajectory
            step_size = self.dtw_scorer.step_size
            all_distances = []

            for start in range(0, T - chunk_length + 1, step_size):
                chunk = states_xyz[:, start:start + chunk_length, :]
                chunk_distances = self.dtw_scorer.compute_distances_batch(chunk)
                all_distances.append(chunk_distances)

            if len(all_distances) > 0:
                distances = torch.stack(all_distances).mean(dim=0)
            else:
                # Trajectory too short for any chunk
                distances = torch.zeros(B, device=device)
        else:
            # Trajectory shorter than chunk length - can't score properly
            # Return neutral score
            return torch.ones(B, device=device) * 0.5

        # Convert distances to scores
        threshold = self.dtw_scorer.threshold

        if self.score_method == 'threshold':
            # Binary: 1.0 if accepted, 0.0 if rejected
            scores = (distances <= threshold).float()
        elif self.score_method == 'distance':
            # Smooth: exp(-distance / threshold)
            # This gives ~0.37 at threshold, ~1.0 for distance=0
            scores = torch.exp(-distances / threshold)
        else:
            raise ValueError(f"Unknown score_method: {self.score_method}")

        return scores.clamp(0, 1)

    def score_single_trajectory(
        self,
        trajectory: torch.Tensor,  # (T, state_dim)
    ) -> float:
        """
        Score a single trajectory.

        Convenience method for scoring one trajectory at a time.

        Args:
            trajectory: (T, state_dim) single trajectory

        Returns:
            score: Reliability score in [0, 1]
        """
        # Add batch dimension
        states = trajectory.unsqueeze(0)
        # Dummy actions and next_states (not used by DTW)
        actions = torch.zeros_like(states)
        next_states = states

        scores = self.score_trajectories(states, actions, next_states)
        return scores[0].item()


class UniformScorer(DynamicsScorerInterface):
    """
    Uniform scorer that gives all trajectories score = 1.0.

    Use this as a baseline or when you don't have a scoring model.
    All sim trajectories will be weighted equally.
    """

    def __init__(self, state_key: str = 'low_dim_state'):
        self._state_key = state_key

    @property
    def state_key(self) -> str:
        return self._state_key

    def score_trajectories(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return uniform scores of 1.0 for all trajectories."""
        B = states.shape[0]
        return torch.ones(B, device=states.device)


def create_dtw_scorer_adapter(
    checkpoint_path: str,
    device: str = 'cuda',
    state_key: str = 'low_dim_state',
    state_dims: list = None,
    score_method: str = 'distance',
    distance_mode: str = 'euclidean',
) -> DTWScorerAdapter:
    """
    Factory function to create DTWScorerAdapter from checkpoint.

    Args:
        checkpoint_path: Path to DTW evaluator pickle file
        device: Torch device
        state_key: Key in obs_dict for state
        state_dims: Which dimensions to use (default: [0,1,2])
        score_method: 'threshold' or 'distance'
        distance_mode: 'euclidean' or 'soft_dtw'

    Returns:
        DTWScorerAdapter instance
    """
    # Import here to avoid circular dependency
    import sys
    sys.path.insert(0, '/home/houhd/code/force_tool')
    from algorithms.common.dtw_scorer_gpu import DTWScorerGPU

    dtw_scorer = DTWScorerGPU.from_checkpoint(
        path=checkpoint_path,
        device=device,
        distance_mode=distance_mode,
    )

    return DTWScorerAdapter(
        dtw_scorer=dtw_scorer,
        state_key=state_key,
        state_dims=state_dims,
        score_method=score_method,
    )
