# Sim-Real Co-Training Architecture for RL Games PPO

## Overview

This document describes an architectural extension to RL Games PPO that enables co-training with both simulated (sim) and real-world data, where simulated data is weighted by a dynamics model's reliability score.

## Current Architecture Analysis

### 1. PPO Replay Buffer (`ExperienceBuffer`)

**Location:** `rl_games/common/experience.py:285-418`

**Interface:**
```python
class ExperienceBuffer:
    def __init__(self, env_info, algo_info, device, aux_tensor_dict=None):
        # Creates tensor_dict with: obses, rewards, values, neglogpacs, dones, actions, mus, sigmas

    def update_data(self, name, index, val):
        # Updates tensor_dict[name][index, :] = val

    def get_transformed(self, transform_op):
        # Returns transformed copy of tensor_dict
```

**Key Dimensions:**
- `obs_base_shape = (horizon_length, num_agents * num_actors)`
- Stores rolling horizon of experience for on-policy PPO

### 2. Data Flow in PPO

1. **Collection:** `play_steps()` in `a2c_common.py:733-795`
   - Collects `horizon_length` steps from vectorized env
   - Stores in `ExperienceBuffer`
   - Computes GAE advantages and returns

2. **Dataset Preparation:** `prepare_dataset()` in `a2c_common.py:1242-1300`
   - Flattens experience buffer: `(horizon, num_envs) → (horizon * num_envs,)`
   - Normalizes advantages
   - Creates `dataset_dict` for `PPODataset`

3. **Training:** `train_epoch()` loops over `PPODataset` minibatches

### 3. IsaacLab Environment Pattern

The `factory_env_flexhole.py` returns observations with multiple keys for different env subsets:

```python
def _get_observations(self):
    obs = super()._get_observations()
    policy_obs = obs["policy"]
    obs["policy_large"] = policy_obs[:self.num_large_envs]      # Subset 1
    obs["policy_reg"] = policy_obs[self.num_large_envs:]        # Subset 2
    return obs
```

This pattern allows simultaneous collection from different env types without switching.

---

## Proposed Architecture: Dual-Buffer Co-Training

### Design Goals

1. **Simultaneous collection** from two observation keys in the same env step
2. Accept data from two sources: **real** (high-trust) and **sim** (variable-trust)
3. Score simulated trajectories using a user-provided dynamics model
4. Weight samples during PPO training based on reliability
5. Policy inference uses single observation format (unchanged from standard PPO)

### Key Components

#### 1. Dynamics Model Interface (Abstract)

**File:** `rl_games/common/dynamics_scorer.py`

```python
from abc import ABC, abstractmethod
import torch


class DynamicsScorerInterface(ABC):
    """
    Abstract interface for dynamics models that score trajectory reliability.

    Users MUST implement this interface with their specific dynamics model.
    The scorer evaluates how well observed transitions match the learned dynamics,
    producing reliability scores used to weight simulated data.
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

        The implementation should evaluate how well the observed transitions
        (state, action) -> next_state match the learned dynamics model.

        Args:
            states: Batch of state trajectories, shape (B, T, state_dim)
            actions: Batch of action trajectories, shape (B, T, action_dim)
            next_states: Batch of next state trajectories, shape (B, T, state_dim)

        Returns:
            scores: (B,) tensor of reliability scores in [0, 1]
                   Higher = more reliable/realistic trajectory

        Example scoring strategies:
            - Prediction error: exp(-MSE(predicted, actual))
            - Ensemble disagreement: exp(-variance_across_ensemble)
            - NLL under probabilistic model: exp(-NLL)
            - Combined: exp(-MSE - variance)
        """
        pass

    @property
    @abstractmethod
    def state_key(self) -> str:
        """
        Key to extract state from obs_dict for dynamics scoring.

        This should match the observation key that contains the state
        information your dynamics model was trained on.

        Example: 'low_dim_state', 'proprio', 'robot_state'
        """
        pass

    @property
    @abstractmethod
    def action_key(self) -> str:
        """
        Key to extract action from the experience buffer.

        Usually 'actions' for standard RL Games setup.
        """
        pass

    def preprocess_for_scoring(
        self,
        obs_dict: dict,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optional: Preprocess observations and actions before scoring.

        Default implementation extracts state using self.state_key.
        Override if you need custom preprocessing (e.g., normalization,
        feature extraction, combining multiple obs keys).

        Args:
            obs_dict: Observation dictionary from buffer
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
```

#### 2. Dual Experience Buffer (Simultaneous Collection)

**File:** `rl_games/common/experience.py` (extend existing)

```python
class DualExperienceBuffer:
    """
    Manages two experience buffers (real and sim) collected simultaneously
    from different observation keys in the same environment step.

    The environment is expected to return observations with separate keys
    for real and sim data (e.g., from IsaacLab's split observation pattern).
    """

    def __init__(
        self,
        env_info,
        algo_info,
        device,
        real_obs_key: str = "policy_real",
        sim_obs_key: str = "policy_sim",
        dynamics_scorer: DynamicsScorerInterface = None,
        real_data_ratio: float = 0.3,
        temperature: float = 1.0,
        aux_tensor_dict=None,
    ):
        """
        Args:
            env_info: Environment info dict
            algo_info: Algorithm info dict containing:
                - num_actors: Total number of envs
                - num_actors_real: Number of real/high-fidelity envs
                - num_actors_sim: Number of sim envs
                - horizon_length: Rollout length
            device: Torch device
            real_obs_key: Key in obs_dict for real env observations
            sim_obs_key: Key in obs_dict for sim env observations
            dynamics_scorer: User-provided model to score sim trajectory reliability
            real_data_ratio: Probability of sampling from real data (default 0.3 = 30%)
                            Sim data ratio is (1 - real_data_ratio) = 0.7 = 70%
            temperature: Softmax temperature for sim trajectory weighting
                        Lower = sharper (favor high-scoring trajectories)
                        Higher = smoother (more uniform sampling)
            aux_tensor_dict: Additional tensors to track
        """
        self.device = device
        self.real_obs_key = real_obs_key
        self.sim_obs_key = sim_obs_key
        self.dynamics_scorer = dynamics_scorer
        self.real_data_ratio = real_data_ratio
        self.sim_data_ratio = 1.0 - real_data_ratio  # Derived: 70% if real is 30%
        self.temperature = temperature

        # Parse actor counts
        self.num_actors_real = algo_info['num_actors_real']
        self.num_actors_sim = algo_info['num_actors_sim']
        self.num_actors_total = algo_info['num_actors']
        self.horizon_length = algo_info['horizon_length']

        assert self.num_actors_real + self.num_actors_sim == self.num_actors_total, \
            f"num_actors_real ({self.num_actors_real}) + num_actors_sim ({self.num_actors_sim}) " \
            f"must equal num_actors ({self.num_actors_total})"

        # Create separate env_info for each buffer
        real_env_info = self._create_subset_env_info(env_info, self.num_actors_real)
        sim_env_info = self._create_subset_env_info(env_info, self.num_actors_sim)

        real_algo_info = {**algo_info, 'num_actors': self.num_actors_real}
        sim_algo_info = {**algo_info, 'num_actors': self.num_actors_sim}

        # Create two separate internal buffers
        self.real_buffer = ExperienceBuffer(real_env_info, real_algo_info, device, aux_tensor_dict)
        self.sim_buffer = ExperienceBuffer(sim_env_info, sim_algo_info, device, aux_tensor_dict)

        # Reliability scores for sim data (computed after collection)
        self.sim_reliability_scores = None
        self.sim_sampling_weights = None

        # Async scoring support
        self._scoring_future = None
        self._scoring_stream = None
        self._use_async_scoring = True  # Can be disabled for debugging

    def _create_subset_env_info(self, env_info, num_actors):
        """Create env_info for a subset of environments."""
        # Most env_info stays the same, just for documentation
        subset_info = env_info.copy()
        return subset_info

    def update_data(self, name, index, val_real, val_sim):
        """
        Update both buffers simultaneously with split data.

        Args:
            name: Key name (e.g., 'obses', 'actions', 'rewards')
            index: Time step index
            val_real: Values for real envs
            val_sim: Values for sim envs
        """
        self.real_buffer.update_data(name, index, val_real)
        self.sim_buffer.update_data(name, index, val_sim)

    def update_data_from_split_obs(self, name, index, full_obs_dict):
        """
        Update observation data by extracting from split obs dict.

        Args:
            name: Key name (should be 'obses')
            index: Time step index
            full_obs_dict: Dict containing real_obs_key and sim_obs_key
        """
        val_real = full_obs_dict[self.real_obs_key]
        val_sim = full_obs_dict[self.sim_obs_key]
        self.update_data(name, index, val_real, val_sim)

    def update_data_from_concat(self, name, index, val_concat):
        """
        Update data that comes concatenated (real first, then sim).

        Used for rewards, dones, actions, values, etc. that are not
        split in the obs dict but concatenated across all envs.

        Args:
            name: Key name
            index: Time step index
            val_concat: Concatenated values, shape (num_actors_total, ...)
        """
        val_real = val_concat[:self.num_actors_real]
        val_sim = val_concat[self.num_actors_real:]
        self.update_data(name, index, val_real, val_sim)

    def compute_sim_reliability_async(self):
        """
        Launch asynchronous computation of reliability scores.

        This method returns immediately, allowing the main thread to continue
        with simulation while scoring runs on a separate CUDA stream.

        Call wait_for_scoring() or get_combined_batch() before using the scores.
        """
        if self.dynamics_scorer is None:
            # No scorer: uniform weights (instant, no async needed)
            self.sim_reliability_scores = torch.ones(
                self.num_actors_sim, device=self.device
            )
            return

        if not self._use_async_scoring:
            # Fallback to synchronous
            self._compute_sim_reliability_sync()
            return

        # Create a separate CUDA stream for scoring
        if self._scoring_stream is None:
            self._scoring_stream = torch.cuda.Stream(device=self.device)

        # Extract and prepare data on main stream (fast, just references/transposes)
        obses = self.sim_buffer.tensor_dict['obses']
        actions = self.sim_buffer.tensor_dict['actions']

        states, actions_processed = self.dynamics_scorer.preprocess_for_scoring(
            obses, actions
        )

        # Transpose to (num_envs, horizon, dim)
        if isinstance(states, dict):
            states = {k: v.transpose(0, 1).contiguous() for k, v in states.items()}
        else:
            states = states.transpose(0, 1).contiguous()
        actions_processed = actions_processed.transpose(0, 1).contiguous()

        # Compute next states
        if isinstance(states, dict):
            next_states = {k: torch.roll(v, -1, dims=1) for k, v in states.items()}
        else:
            next_states = torch.roll(states, -1, dims=1)

        # Prepare slices (exclude last step)
        if isinstance(states, dict):
            states_slice = {k: v[:, :-1].contiguous() for k, v in states.items()}
            next_states_slice = {k: v[:, :-1].contiguous() for k, v in next_states.items()}
        else:
            states_slice = states[:, :-1].contiguous()
            next_states_slice = next_states[:, :-1].contiguous()
        actions_slice = actions_processed[:, :-1].contiguous()

        # Record event on main stream to ensure data is ready
        ready_event = torch.cuda.Event()
        ready_event.record()

        # Launch scoring on separate stream
        def _score_async():
            with torch.cuda.stream(self._scoring_stream):
                # Wait for data preparation to complete
                self._scoring_stream.wait_event(ready_event)

                with torch.no_grad():
                    scores = self.dynamics_scorer.score_trajectories(
                        states_slice,
                        actions_slice,
                        next_states_slice
                    )

                self.sim_reliability_scores = scores

                # Record completion event
                self._scoring_complete_event = torch.cuda.Event()
                self._scoring_complete_event.record()

        # Execute async
        _score_async()

    def wait_for_scoring(self):
        """
        Block until async scoring is complete.

        Safe to call multiple times - will return immediately if already done.
        """
        if hasattr(self, '_scoring_complete_event') and self._scoring_complete_event is not None:
            self._scoring_complete_event.synchronize()
            self._scoring_complete_event = None

    def compute_sim_reliability(self):
        """
        Synchronous wrapper for backward compatibility.

        Launches async scoring and immediately waits for completion.
        """
        self.compute_sim_reliability_async()
        self.wait_for_scoring()
        return self.sim_reliability_scores

    def _compute_sim_reliability_sync(self):
        """
        Synchronous implementation (used when async is disabled).
        """
        obses = self.sim_buffer.tensor_dict['obses']
        actions = self.sim_buffer.tensor_dict['actions']

        states, actions_processed = self.dynamics_scorer.preprocess_for_scoring(
            obses, actions
        )

        if isinstance(states, dict):
            states = {k: v.transpose(0, 1) for k, v in states.items()}
        else:
            states = states.transpose(0, 1)
        actions_processed = actions_processed.transpose(0, 1)

        if isinstance(states, dict):
            next_states = {k: torch.roll(v, -1, dims=1) for k, v in states.items()}
            states_slice = {k: v[:, :-1] for k, v in states.items()}
            next_states_slice = {k: v[:, :-1] for k, v in next_states.items()}
        else:
            next_states = torch.roll(states, -1, dims=1)
            states_slice = states[:, :-1]
            next_states_slice = next_states[:, :-1]

        with torch.no_grad():
            self.sim_reliability_scores = self.dynamics_scorer.score_trajectories(
                states_slice,
                actions_processed[:, :-1],
                next_states_slice
            )

    def compute_sampling_weights(self):
        """
        Compute per-trajectory sampling weights for sim data using softmax.

        Returns:
            weights: (num_sim_envs,) normalized weights summing to 1
        """
        if self.sim_reliability_scores is None:
            self.compute_sim_reliability()

        # Apply temperature scaling and softmax
        scores = self.sim_reliability_scores
        if self.temperature != 1.0:
            scores = scores / self.temperature

        # Handle case where all scores are 0 (all below threshold)
        if scores.sum() < 1e-8:
            self.sim_sampling_weights = torch.ones_like(scores) / len(scores)
        else:
            self.sim_sampling_weights = torch.softmax(scores, dim=0)

        return self.sim_sampling_weights

    def get_transformed_separate(self, transform_op):
        """
        Get transformed data from both buffers separately.

        Returns:
            real_dict: Transformed real buffer data
            sim_dict: Transformed sim buffer data
        """
        real_dict = self.real_buffer.get_transformed(transform_op)
        sim_dict = self.sim_buffer.get_transformed(transform_op)
        return real_dict, sim_dict

    def get_combined_batch(self, transform_op, target_batch_size: int = None):
        """
        Sample a combined batch from both buffers using the simple weighting scheme:

        Sampling distribution:
        - 30% (real_data_ratio) probability: sample uniformly from real data
        - 70% (sim_data_ratio) probability: sample from sim data weighted by softmax(scores/T)

        Real trajectories have weight=1 (uniform within real).
        Sim trajectories are weighted by dynamics model likelihood scores.

        IMPORTANT: This method automatically waits for async scoring to complete.

        Args:
            transform_op: Transformation to apply (e.g., swap_and_flatten01)
            target_batch_size: Number of samples to draw. If None, uses real_size + sim_size.

        Returns:
            combined_dict: Dictionary with:
                - All standard PPO keys (obses, actions, rewards, etc.)
                - 'sample_weights': Per-sample importance weights (for loss weighting)
                - 'source_labels': 0 for real, 1 for sim
        """
        # Wait for any pending async scoring to complete
        self.wait_for_scoring()

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
            if isinstance(d['obses'], dict):
                return list(d['obses'].values())[0].shape[0]
            return d['obses'].shape[0]

        real_size = get_batch_size(real_dict)
        sim_size = get_batch_size(sim_dict)

        # Determine batch size
        if target_batch_size is None:
            target_batch_size = real_size + sim_size

        # ============ SIMPLE SAMPLING SCHEME ============
        # Split batch: 30% real, 70% sim (configurable via real_data_ratio)
        num_real_samples = int(target_batch_size * self.real_data_ratio)
        num_sim_samples = target_batch_size - num_real_samples

        # Sample from REAL: uniform (all weight = 1)
        real_indices = torch.randint(0, real_size, (num_real_samples,), device=self.device)

        # Sample from SIM: weighted by softmax(scores / T)
        # Expand trajectory-level weights to sample-level
        samples_per_traj = sim_size // self.num_actors_sim
        sim_sample_probs = self.sim_sampling_weights.repeat_interleave(samples_per_traj)
        # Ensure it sums to 1 for multinomial
        sim_sample_probs = sim_sample_probs / sim_sample_probs.sum()

        sim_indices = torch.multinomial(sim_sample_probs, num_sim_samples, replacement=True)

        # ============ BUILD COMBINED BATCH ============
        combined_dict = {}
        for key in real_dict.keys():
            if real_dict[key] is None:
                combined_dict[key] = None
                continue

            if isinstance(real_dict[key], dict):
                combined_dict[key] = {}
                for k in real_dict[key].keys():
                    real_samples = real_dict[key][k][real_indices]
                    sim_samples = sim_dict[key][k][sim_indices]
                    combined_dict[key][k] = torch.cat([real_samples, sim_samples], dim=0)
            else:
                real_samples = real_dict[key][real_indices]
                sim_samples = sim_dict[key][sim_indices]
                combined_dict[key] = torch.cat([real_samples, sim_samples], dim=0)

        # ============ SAMPLE WEIGHTS FOR LOSS ============
        # Real samples: weight = 1.0
        real_weights = torch.ones(num_real_samples, device=self.device)

        # Sim samples: weight = 1.0 (importance sampling correction already done via sampling)
        # Or optionally keep the reliability score as weight for further loss weighting
        sim_weights = torch.ones(num_sim_samples, device=self.device)

        combined_dict['sample_weights'] = torch.cat([real_weights, sim_weights], dim=0)

        # Source labels for debugging/logging
        combined_dict['source_labels'] = torch.cat([
            torch.zeros(num_real_samples, device=self.device),
            torch.ones(num_sim_samples, device=self.device)
        ], dim=0)

        # ============ SHUFFLE ============
        total_size = num_real_samples + num_sim_samples
        perm = torch.randperm(total_size, device=self.device)
        for key, val in combined_dict.items():
            if val is None:
                continue
            if isinstance(val, dict):
                combined_dict[key] = {k: v[perm] for k, v in val.items()}
            else:
                combined_dict[key] = val[perm]

        return combined_dict

    def get_reliability_stats(self):
        """
        Get statistics about reliability scores for logging.

        Returns:
            dict with reliability statistics
        """
        if self.sim_reliability_scores is None:
            return {}

        scores = self.sim_reliability_scores
        stats = {
            'reliability/mean': scores.mean().item(),
            'reliability/std': scores.std().item(),
            'reliability/min': scores.min().item(),
            'reliability/max': scores.max().item(),
        }

        if self.sim_sampling_weights is not None:
            # Entropy of sampling distribution (higher = more uniform)
            entropy = -(self.sim_sampling_weights * torch.log(self.sim_sampling_weights + 1e-8)).sum().item()
            max_entropy = torch.log(torch.tensor(float(len(self.sim_sampling_weights)))).item()
            stats['reliability/sampling_entropy'] = entropy
            stats['reliability/sampling_uniformity'] = entropy / max_entropy  # 1.0 = perfectly uniform

        return stats

    @property
    def tensor_dict(self):
        """
        For compatibility: return combined tensor dict.
        Note: This returns concatenated data without weighting.
        Use get_combined_batch() for weighted training data.
        """
        # This is mainly for compatibility with existing code that
        # accesses tensor_dict directly (e.g., for computing advantages)
        combined = {}
        for key in self.real_buffer.tensor_dict.keys():
            real_val = self.real_buffer.tensor_dict[key]
            sim_val = self.sim_buffer.tensor_dict[key]

            if real_val is None:
                combined[key] = None
            elif isinstance(real_val, dict):
                combined[key] = {
                    k: torch.cat([real_val[k], sim_val[k]], dim=1)
                    for k in real_val.keys()
                }
            else:
                combined[key] = torch.cat([real_val, sim_val], dim=1)

        return combined
```

#### 3. Extended PPO Dataset

**File:** `rl_games/common/datasets.py` (extend existing)

```python
class WeightedPPODataset(PPODataset):
    """
    PPO Dataset that supports per-sample importance weights for co-training.

    Weights can be used in two ways:
    1. Weighted sampling: Sample minibatches with probability proportional to weight
    2. Weighted loss: Apply weights to per-sample losses during gradient computation

    This implementation supports both approaches.
    """

    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_length,
                 use_weighted_sampling: bool = False):
        super().__init__(batch_size, minibatch_size, is_discrete, is_rnn, device, seq_length)
        self.use_weighted_sampling = use_weighted_sampling
        self.has_weights = False
        self.sample_weights = None

    def update_values_dict(self, values_dict):
        super().update_values_dict(values_dict)
        if values_dict is not None:
            self.has_weights = 'sample_weights' in values_dict
            if self.has_weights:
                self.sample_weights = values_dict['sample_weights']

    def _get_item(self, idx):
        """Get minibatch with optional weighted sampling."""
        if self.use_weighted_sampling and self.has_weights:
            # Weighted sampling: sample indices proportional to weights
            indices = torch.multinomial(
                self.sample_weights,
                self.minibatch_size,
                replacement=True
            )
        else:
            # Standard sequential minibatching
            start = idx * self.minibatch_size
            end = (idx + 1) * self.minibatch_size
            indices = torch.arange(start, end, device=self.device)

        self.last_range = (indices[0].item(), indices[-1].item() + 1)

        input_dict = {}
        for k, v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                if isinstance(v, dict):
                    input_dict[k] = {kd: vd[indices] for kd, vd in v.items()}
                else:
                    input_dict[k] = v[indices]

        # Always include weights for potential use in loss weighting
        if self.has_weights:
            input_dict['sample_weights'] = self.sample_weights[indices]

        return input_dict
```

---

## Configuration

### YAML Config Example

```yaml
# rl_games config with co-training
params:
  config:
    # ... standard config ...

    # Co-training configuration
    cotrain:
      enabled: true

      # Observation keys for real and sim data
      # These must match keys returned by your IsaacLab env's _get_observations()
      real_obs_key: "policy_real"    # Key for real/high-fidelity env observations
      sim_obs_key: "policy_sim"      # Key for simulated env observations

      # Environment split
      num_actors_real: 128           # Number of real/high-fidelity envs
      num_actors_sim: 384            # Number of sim envs (num_actors - num_actors_real)

      # Sampling distribution (simple scheme)
      real_data_ratio: 0.3           # 30% samples from real (uniform)
                                     # 70% samples from sim (weighted by softmax(scores/T))
      temperature: 1.0               # Softmax temperature for sim weighting
                                     # Lower = sharper (favor high-scoring)
                                     # Higher = smoother (more uniform)

      # Dynamics model (user must provide programmatically)
      dynamics_scorer: null          # Set in Python, not YAML
```

#### 4. Modified play_steps() for Simultaneous Collection

**File:** `rl_games/common/a2c_common.py` (modifications to `ContinuousA2CBase`)

```python
def play_steps(self):
    """
    Collect experience from environment with simultaneous dual-buffer updates.

    Key features:
    - Observations come with split keys (e.g., 'policy_real', 'policy_sim')
    - Both buffers are updated in the SAME loop iteration
    - No buffer switching - all data flows through simultaneously
    - Async scoring overlaps with advantage computation
    """
    update_list = self.update_list
    step_time = 0.0

    for n in range(self.horizon_length):
        res_dict = self.get_action_values(self.obs)

        # Update observations - extract from split keys simultaneously
        # self.obs['obs'] contains both 'policy_real' and 'policy_sim' keys
        self.experience_buffer.update_data_from_split_obs('obses', n, self.obs['obs'])

        # Update dones - comes concatenated [real_envs, sim_envs]
        self.experience_buffer.update_data_from_concat('dones', n, self.dones)

        # Update policy outputs (actions, values, etc.) - all concatenated
        for k in update_list:
            self.experience_buffer.update_data_from_concat(k, n, res_dict[k])

        # Step environment - actions go to ALL envs, returns come back concatenated
        step_time_start = time.time()
        self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
        step_time += time.time() - step_time_start

        # Update rewards - concatenated
        shaped_rewards = self.rewards_shaper(rewards)
        self.experience_buffer.update_data_from_concat('rewards', n, shaped_rewards)

        # ... rest of bookkeeping (game_rewards, etc.) ...

    # =========== ASYNC SCORING ===========
    # Launch scoring on separate CUDA stream - returns immediately
    self.experience_buffer.compute_sim_reliability_async()

    # While scoring runs async, compute advantages/returns on main stream
    # This overlaps dynamics model inference with GAE computation
    last_values = self.get_values(self.obs)
    fdones = self.dones.float()
    mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
    mb_values = self.experience_buffer.tensor_dict['values']
    mb_rewards = self.experience_buffer.tensor_dict['rewards']
    mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
    mb_returns = mb_advs + mb_values

    # get_combined_batch() automatically waits for scoring to complete
    # before using the reliability scores
    batch_dict = self.experience_buffer.get_combined_batch(swap_and_flatten01)
    batch_dict['returns'] = swap_and_flatten01(mb_returns)
    batch_dict['played_frames'] = self.batch_size
    batch_dict['step_time'] = step_time

    return batch_dict
```

**Key points:**
1. **Single loop** - no separate collection phases for real vs sim
2. **`update_data_from_split_obs()`** - extracts real/sim from obs dict keys
3. **`update_data_from_concat()`** - splits concatenated tensors by env index
4. **`compute_sim_reliability_async()`** - launches scoring on separate CUDA stream, returns immediately
5. **Overlapped execution** - GAE/advantage computation runs on main stream while scoring runs async
6. **`get_combined_batch()`** - automatically waits for scoring to complete before using scores

---

## User Implementation Guide

### Step 1: Implement Your Dynamics Scorer

Create a class that extends `DynamicsScorerInterface`:

```python
# my_project/dynamics_scorer.py

from rl_games.common.dynamics_scorer import DynamicsScorerInterface
import torch


class MyDynamicsScorer(DynamicsScorerInterface):
    """
    Example implementation using your trained dynamics model.
    """

    def __init__(self, dynamics_model, score_method='prediction_error'):
        """
        Args:
            dynamics_model: Your pre-trained dynamics model
            score_method: How to compute reliability score
        """
        self.model = dynamics_model
        self.score_method = score_method
        self._state_key = 'low_dim_state'  # Adjust to your obs structure
        self._action_key = 'actions'

    @property
    def state_key(self) -> str:
        return self._state_key

    @property
    def action_key(self) -> str:
        return self._action_key

    def score_trajectories(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score trajectories based on dynamics model predictions.
        """
        B, T, D = states.shape

        with torch.no_grad():
            # Your model's forward pass
            # Adjust this to match your dynamics model's interface
            state_0 = states[:, 0:1, :]
            predictions = self.model.forward(state_0, actions)

            if self.score_method == 'prediction_error':
                # For ensemble model: predictions shape (E, B, T, D)
                if predictions.dim() == 4:
                    pred_mean = predictions.mean(dim=0)
                else:
                    pred_mean = predictions

                mse = ((pred_mean - next_states) ** 2).mean(dim=(1, 2))
                scores = torch.exp(-mse)

            elif self.score_method == 'ensemble_disagreement':
                # Requires ensemble model
                pred_var = predictions.var(dim=0).mean(dim=(1, 2))
                scores = torch.exp(-pred_var)

            elif self.score_method == 'nll':
                # For probabilistic model returning (mean, log_var)
                pred_mean, pred_log_var = predictions
                pred_var = pred_log_var.exp()
                diff = next_states - pred_mean
                nll = 0.5 * (pred_log_var + diff**2 / (pred_var + 1e-8))
                scores = torch.exp(-nll.mean(dim=(1, 2)))

            else:
                raise ValueError(f"Unknown score_method: {self.score_method}")

        return scores.clamp(0, 1)
```

### Step 2: Modify Your IsaacLab Environment

Your environment must return split observations. Example pattern:

```python
# In your IsaacLab env

class MyCotrainEnv(DirectRLEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Configure which envs are "real" vs "sim"
        self.num_real_envs = cfg.num_real_envs
        self.num_sim_envs = self.num_envs - self.num_real_envs

    def _get_observations(self):
        obs = super()._get_observations()

        # Split policy observations by env type
        policy_obs = obs["policy"]
        obs["policy_real"] = policy_obs[:self.num_real_envs]
        obs["policy_sim"] = policy_obs[self.num_real_envs:]

        return obs
```

### Step 3: Configure and Run Training

```python
# training_script.py

from my_project.dynamics_scorer import MyDynamicsScorer
from my_project.dynamics_model import DynamicsEnsemble
from rl_games.torch_runner import Runner

# Load your pre-trained dynamics model
dynamics_model = DynamicsEnsemble.load("checkpoints/dynamics.pt")
dynamics_scorer = MyDynamicsScorer(dynamics_model, score_method='prediction_error')

# Configure RL Games
config = {
    'params': {
        'config': {
            # ... standard rl_games config ...
            'num_actors': 512,  # Total envs

            'cotrain': {
                'enabled': True,
                'real_obs_key': 'policy_real',
                'sim_obs_key': 'policy_sim',
                'num_actors_real': 128,
                'num_actors_sim': 384,
                'dynamics_scorer': dynamics_scorer,  # Pass your scorer
                'sim_data_ratio': 0.5,
                'temperature': 0.5,
                'weight_by_reward': True,
                'min_reliability': 0.1,
                'use_weighted_loss': True,
            }
        }
    }
}

# Run training
runner = Runner()
runner.load(config)
runner.run()
```

---

## Scoring/Weighting Strategies

### 1. Prediction Error Based
```
score = exp(-MSE(predicted_next_state, actual_next_state))
```
- High score when dynamics model accurately predicts transitions
- Favors trajectories in well-learned regions of state space

### 2. Ensemble Disagreement Based
```
score = exp(-variance_across_ensemble)
```
- High score when ensemble members agree (low epistemic uncertainty)
- Favors trajectories where model is confident

### 3. NLL-Based (Probabilistic Model)
```
score = exp(-NLL(actual_next_state | predicted_distribution))
```
- Uses full probabilistic prediction
- Accounts for both mean accuracy and predicted uncertainty

### 4. Combined
```
score = exp(-alpha * MSE - beta * variance)
```
- Balances accuracy and confidence
- Hyperparameters control relative importance

### 5. Reward-Weighted
```
final_score = reliability_score * normalized_trajectory_reward
```
- Prioritizes reliable AND high-reward trajectories
- Helps policy learn from successful demonstrations

### 6. Temperature Scaling
```
sampling_weight = softmax(scores / temperature)
```
- `T < 1`: Sharper selection (mostly highest-scoring trajectories)
- `T = 1`: Standard softmax
- `T > 1`: Smoother selection (more uniform)

---

## Metrics to Log

The `DualExperienceBuffer.get_reliability_stats()` method provides:

| Metric | Description |
|--------|-------------|
| `reliability/mean` | Mean reliability score across sim trajectories |
| `reliability/std` | Standard deviation of scores |
| `reliability/min` | Minimum score (worst trajectory) |
| `reliability/max` | Maximum score (best trajectory) |
| `reliability/above_threshold` | Fraction of trajectories above min_reliability |
| `reliability/effective_sim_ratio` | Actual weighted contribution of sim data |

Additional metrics to log in training loop:
- Separate loss curves for real vs sim samples (using `source_labels`)
- Value function error by source
- Policy KL divergence by source

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    IsaacLab Environment                         │
│  ┌─────────────────┐              ┌─────────────────┐          │
│  │  Real Envs      │              │   Sim Envs      │          │
│  │  (0..N_real)    │              │  (N_real..N)    │          │
│  └────────┬────────┘              └────────┬────────┘          │
│           │ policy_real                    │ policy_sim        │
└───────────┼────────────────────────────────┼────────────────────┘
            │                                │
            ▼                                ▼
┌───────────────────────────────────────────────────────────────┐
│                  DualExperienceBuffer                         │
│  ┌─────────────────┐              ┌─────────────────┐        │
│  │  Real Buffer    │              │   Sim Buffer    │        │
│  │  (trusted)      │              │  (weighted)     │        │
│  └─────────────────┘              └────────┬────────┘        │
│                                            │                  │
│                                            ▼                  │
│                              ┌─────────────────────────┐     │
│                              │   DynamicsScorerInterface│     │
│                              │   (user-implemented)    │     │
│                              └────────────┬────────────┘     │
│                                           │                  │
│                                           ▼                  │
│                              ┌─────────────────────────┐     │
│                              │  Reliability Scores     │     │
│                              │  + Softmax Weighting    │     │
│                              └─────────────────────────┘     │
│                                                              │
│  get_combined_batch() ──────────────────────────────────────►│
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│                   WeightedPPODataset                          │
│   - sample_weights for loss weighting                        │
│   - source_labels for diagnostics                            │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│                      PPO Training                             │
│   - Weighted actor/critic losses                             │
│   - Standard policy inference (single obs format)            │
└───────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

1. **Simultaneous collection, not switching**: Both buffers are filled in the same env step, avoiding the complexity of buffer switching logic.

2. **Abstract scorer interface**: Users implement their own `DynamicsScorerInterface` with their specific dynamics model, keeping RL Games agnostic to model details.

3. **Two internal buffers**: Keeps real and sim data cleanly separated, allows independent advantage/return computation, and simplifies the reliability scoring which only applies to sim data.

4. **Observation key configuration**: Users specify which keys in the obs dict correspond to real vs sim data, matching the IsaacLab split observation pattern.

5. **Flexible weighting**: Temperature scaling, reward weighting, and minimum thresholds give users control over how aggressively to filter sim data.

6. **Backward compatible**: When `cotrain.enabled=False`, falls back to standard single-buffer PPO with no code changes required.
