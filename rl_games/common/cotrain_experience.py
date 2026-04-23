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


class ChunkScorer:
    """Chunked-MSE scoring helper shared by CotrainExperienceBuffer (PPO) and
    CotrainVectorizedReplayBuffer (SAC).

    Stateless wrt rollout data: `score_chunks` consumes (T, N, *) sim-only
    tensors and returns ((K, N) MSE, K chunk-start timesteps); `paint_weights`
    expands per-chunk weights onto a (T, N) per-timestep grid via per-mode
    dispatch (binary or softmax); `sample_idx` returns a multinomial / uniform
    sample over a flat weight vector; `log_stats` writes per-mode tensorboard
    scalars.

    Owned config: scorer (BaseScorer-like), threshold (float), scoring_mode
    ("binary" | "softmax"), temperature (float, softmax-only), device.
    """

    SCORING_MODES = ("binary", "softmax")

    def __init__(self, scorer, threshold, scoring_mode, temperature, device):
        assert scorer is not None
        assert threshold is not None, "threshold required when scoring is enabled"
        assert scoring_mode in self.SCORING_MODES, (
            f"scoring_mode must be one of {self.SCORING_MODES}, got {scoring_mode!r}"
        )
        if scoring_mode == "softmax":
            assert temperature is not None and float(temperature) > 0, (
                "temperature > 0 is required when scoring_mode='softmax'"
            )
        self.scorer = scorer
        self.threshold = float(threshold)
        self.scoring_mode = scoring_mode
        self.temperature = float(temperature) if temperature is not None else None
        self.device = device

    @property
    def horizon(self) -> int:
        return self.scorer.algo.horizon

    @property
    def future_horizon(self) -> int:
        return self.scorer.algo.future_horizon

    def score_chunks(self, sim_obses, sim_actions):
        """Extract sim chunks and run the scorer.

        Args:
            sim_obses:   (T, N, D_s)
            sim_actions: (T, N, D_a)
        Returns:
            (scores: (K, N) | None, chunk_starts: list[int])
            (None, []) when T < H + F (rollout too short for any valid chunk).
        """
        H = self.horizon
        F = self.future_horizon
        T = sim_obses.shape[0]
        N = sim_obses.shape[1]
        chunk_starts = list(range(0, T - H - F + 1, F))
        if len(chunk_starts) == 0:
            return None, []

        all_state_0, all_actions, all_state_gt = [], [], []
        for t in chunk_starts:
            all_state_0.append(sim_obses[t:t + H])
            all_actions.append(sim_actions[t + H - 1:t + H - 1 + F])
            all_state_gt.append(sim_obses[t + H:t + H + F])

        K = len(chunk_starts)
        state_0 = torch.stack(all_state_0).permute(0, 2, 1, 3).reshape(K * N, H, -1)
        future_actions = torch.stack(all_actions).permute(0, 2, 1, 3).reshape(K * N, F, -1)
        state_gt = torch.stack(all_state_gt).permute(0, 2, 1, 3).reshape(K * N, F, -1)

        data = {"state_0": state_0, "actions": future_actions, "state_gt": state_gt}
        scores = self.scorer.score(data, mode="mse").reshape(K, N)
        return scores, chunk_starts

    def paint_weights(self, scores, chunk_starts, T, N):
        """Build (T, N) per-timestep weights from per-chunk scores.

        Timesteps not covered by any chunk's future window stay at 1.0
        (caller is responsible for filling the real-env complement).
        """
        sim_w = torch.ones(T, N, device=self.device)
        if scores is None:
            return sim_w
        if self.scoring_mode == "binary":
            chunk_weight = self._chunk_weight_binary(scores)
        elif self.scoring_mode == "softmax":
            chunk_weight = self._chunk_weight_softmax(scores)
        else:
            raise NotImplementedError(f"Unknown scoring_mode={self.scoring_mode!r}")
        F = self.future_horizon
        H = self.horizon
        for i, t in enumerate(chunk_starts):
            sim_w[t + H:t + H + F] = chunk_weight[i].unsqueeze(0).expand(F, -1)
        return sim_w

    def _chunk_weight_binary(self, scores):
        return (scores <= self.threshold).float()

    def _chunk_weight_softmax(self, scores):
        return torch.exp(-scores / self.temperature)

    def sample_idx(self, flat_weights, total):
        """Return a (total,) LongTensor index into flat_weights, or None for
        no-op (all weights == 1.0 in binary mode, or zero total weight)."""
        if self.scoring_mode == "binary":
            return self._sample_idx_binary(flat_weights, total)
        if self.scoring_mode == "softmax":
            return self._sample_idx_softmax(flat_weights, total)
        raise NotImplementedError(f"Unknown scoring_mode={self.scoring_mode!r}")

    def _sample_idx_binary(self, flat_w, total):
        accepted_idx = flat_w.nonzero(as_tuple=True)[0]
        if accepted_idx.numel() == 0:
            return None
        if accepted_idx.numel() == total:
            return None
        return accepted_idx[torch.randint(0, accepted_idx.numel(), (total,), device=self.device)]

    def _sample_idx_softmax(self, flat_w, total):
        if flat_w.sum() <= 0:
            return None
        return torch.multinomial(flat_w, total, replacement=True)

    def log_stats(self, weights, writer, step):
        """Write per-mode scalars to tensorboard. weights is (T, N) sim-only."""
        if writer is None:
            return
        if self.scoring_mode == "binary":
            writer.add_scalar(
                "cotrain/sim_accept_rate", weights.float().mean().item(), step,
            )
        elif self.scoring_mode == "softmax":
            w_flat = weights.reshape(-1)
            total = w_flat.numel()
            ess = (w_flat.sum() ** 2) / (w_flat.pow(2).sum().clamp_min(1e-12))
            writer.add_scalar("cotrain/sim_weight_mean", w_flat.mean().item(), step)
            writer.add_scalar("cotrain/sim_weight_ess_norm", (ess / total).item(), step)
        else:
            raise NotImplementedError(f"Unknown scoring_mode={self.scoring_mode!r}")


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
        scoring_mode: str = "binary",
        temperature: Optional[float] = None,
        plot_dir: Optional[str] = None,
        plot_every: int = 1,
        plot_num_trajectories: int = 10,
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
            scoring_mode: "binary" (default, hard accept/reject at threshold)
                or "softmax" (sampling weights ∝ exp(-score / temperature) on
                scored sim cells; real / pre-history cells get weight 1.0).
            temperature: Softmax temperature τ for mode="softmax" — sets the
                sharpness of the weighting. τ ≈ threshold gives a transition
                at the threshold a weight of exp(-1) ≈ 0.37 relative to a
                score-0 transition. Required when scoring_mode="softmax".
            plot_dir: If set, save a per-resample diagnostic PNG of randomly
                sampled sim-env trajectories colored by per-chunk MSE into
                this directory (created if missing). Intended for monitoring
                the dynamics-scorer distribution shift over RL training.
            plot_every: Save a plot every N resample() calls. plot_dir must
                be set for this to fire.
            plot_num_trajectories: How many random sim envs to render per
                plot (capped at num_sim_envs).
            writer: Optional tensorboard SummaryWriter for logging.
        """
        self.buffer = ExperienceBuffer(env_info, algo_info, device)
        self.device = device
        self.writer = writer
        self._log_step = 0

        self.scoring_mode = scoring_mode

        self.plot_dir = plot_dir
        self.plot_every = max(1, int(plot_every))
        self.plot_num_trajectories = max(1, int(plot_num_trajectories))

        if scorer is not None:
            assert sim_env_idx is not None, "sim_env_idx is required when scorer is provided"
            assert sim_env_idx.dim() == 1, (
                f"sim_env_idx must be 1-D, got shape {tuple(sim_env_idx.shape)}"
            )
            self._chunk_scorer = ChunkScorer(
                scorer=scorer, threshold=threshold,
                scoring_mode=scoring_mode, temperature=temperature, device=device,
            )
            # Back-compat aliases used by plotting code paths below.
            self.scorer = scorer
            self.threshold = self._chunk_scorer.threshold
            self.temperature = self._chunk_scorer.temperature
            self._horizon = self._chunk_scorer.horizon
            self._future_horizon = self._chunk_scorer.future_horizon
            self.sim_env_idx = sim_env_idx.to(self.device).long()
        else:
            self._chunk_scorer = None
            self.scorer = None
            self.threshold = None
            self.temperature = None
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

        # Score once; reuse for weight construction AND diagnostic plotting.
        scores, chunk_starts = self._score_sim_chunks(td, T)
        weights = self._build_scoring_weights_from_scores(
            scores, chunk_starts, T, num_envs,
        )

        self._log_scoring_stats(weights)
        # self._maybe_save_resample_plot(td, scores, chunk_starts)
        if scores is not None and self.threshold is not None:
            accept_ratio = (scores < self.threshold).float().mean().item()
            print(f"[Cotrain] sim acceptance ratio (score<thr): {accept_ratio:.4f}")
        self._apply_resample(td, weights, T, num_envs)

        return rnn_states_raw

    # ------------------------------------------------------------------
    # Weight construction (one method per mode, dispatched by scoring_mode)
    # ------------------------------------------------------------------

    def _build_scoring_weights_from_scores(
        self,
        scores: Optional[torch.Tensor],
        chunk_starts: list,
        T: int,
        num_envs: int,
    ) -> torch.Tensor:
        """Build (T, num_envs) non-negative sampling weights from precomputed
        per-chunk MSE scores. Real envs (complement of sim_env_idx) and
        sim-env timesteps outside any scored chunk window always receive
        weight 1.0. Scored sim chunks get their per-mode weight from
        the shared ChunkScorer.paint_weights helper."""
        weights = torch.ones(T, num_envs, device=self.device)
        num_sim = int(self.sim_env_idx.numel())
        if num_sim == 0 or scores is None:
            return weights
        sim_w = self._chunk_scorer.paint_weights(scores, chunk_starts, T, num_sim)
        weights[:, self.sim_env_idx] = sim_w
        return weights

    def _score_sim_chunks(self, td: dict, T: int):
        """Extract sim-env chunks, run the scorer, return (scores, chunk_starts)."""
        sim_obses = td["obses"].index_select(1, self.sim_env_idx)
        actions   = td["actions"].index_select(1, self.sim_env_idx)
        return self._chunk_scorer.score_chunks(sim_obses, actions)

    # Delegation shims — keep these on CotrainExperienceBuffer so existing
    # tests and call-sites that access them directly continue to work.
    def _chunk_weight_binary(self, scores: torch.Tensor) -> torch.Tensor:
        return self._chunk_scorer._chunk_weight_binary(scores)

    def _chunk_weight_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        return self._chunk_scorer._chunk_weight_softmax(scores)

    def _sample_idx_binary(self, flat_w: torch.Tensor, total: int) -> Optional[torch.Tensor]:
        return self._chunk_scorer._sample_idx_binary(flat_w, total)

    def _sample_idx_softmax(self, flat_w: torch.Tensor, total: int) -> Optional[torch.Tensor]:
        return self._chunk_scorer._sample_idx_softmax(flat_w, total)

    # ------------------------------------------------------------------
    # Sampling (one method per mode, dispatched by scoring_mode)
    # ------------------------------------------------------------------

    def _apply_resample(self, td: dict, weights: torch.Tensor, T: int, num_envs: int):
        """Draw sample_idx via the per-mode sampler and shuffle every field in
        `td` shaped (T, num_envs, *) by the SAME indices. Sharing sample_idx
        across fields preserves row-wise relationships (e.g. returns - values
        -> advantages) post-resample."""
        flat_w = weights.reshape(-1)
        total = T * num_envs
        sample_idx = self._chunk_scorer.sample_idx(flat_w, total)
        if sample_idx is None:
            return
        for key, val in td.items():
            if not isinstance(val, torch.Tensor):
                continue
            if val.shape[0] != T or (val.ndim >= 2 and val.shape[1] != num_envs):
                continue
            flat_shape = (total, *val.shape[2:])
            td[key] = val.reshape(flat_shape)[sample_idx].reshape(val.shape)

    # ------------------------------------------------------------------
    # Logging (per-mode stats)
    # ------------------------------------------------------------------

    def _log_scoring_stats(self, weights: torch.Tensor) -> None:
        # _log_step advances on every resample so it can name plot files
        # consistently even when the tensorboard writer is absent.
        self._log_step += 1
        if self.writer is None:
            return
        sim_w = weights.index_select(1, self.sim_env_idx)
        self._chunk_scorer.log_stats(sim_w, self.writer, self._log_step)

    # ------------------------------------------------------------------
    # Diagnostic plotting (per-resample distribution-shift monitor)
    # ------------------------------------------------------------------

    def _maybe_save_resample_plot(
        self,
        td: dict,
        scores: Optional[torch.Tensor],
        chunk_starts: list,
    ) -> None:
        """Save diagnostic PNGs of sim-env data being scored by the dynamics
        model. Renders (a) per-chunk MSE coloring on sampled trajectories and
        (b) per-chunk scorer prediction vs actual-rollout 3D comparisons for a
        handful of random chunks — intended for monitoring dynamics-scorer
        distribution shift over the course of RL training."""
        if self.plot_dir is None or scores is None or len(chunk_starts) == 0:
            return
        # _log_step was just incremented in _log_scoring_stats, so the first
        # call writes step_1 rather than step_0.
        if self._log_step % self.plot_every != 0:
            return

        import logging, os
        # Silence third-party DEBUG spam that floods logs when the upstream
        # logger is configured at DEBUG level (Hydra, etc.):
        # - matplotlib.font_manager: per-font scoring on every plot call
        # - PIL.PngImagePlugin / PIL.Image: chunk parsing on every PNG save
        for _name in ("matplotlib", "PIL"):
            logging.getLogger(_name).setLevel(logging.WARNING)

        os.makedirs(self.plot_dir, exist_ok=True)
        step_tag = f"step_{self._log_step:05d}"

        self._plot_scoring_overview(td, scores, chunk_starts, step_tag)
        self._plot_chunk_pred_vs_actual(td, scores, chunk_starts, step_tag)

    def _plot_scoring_overview(
        self,
        td: dict,
        scores: torch.Tensor,
        chunk_starts: list,
        step_tag: str,
    ) -> None:
        """Two trajectory-level views: per-chunk MSE heat map + binary accept/reject."""
        sim_obses = td["obses"].index_select(1, self.sim_env_idx)  # (T, N_sim, D_s)
        N_sim = sim_obses.shape[1]
        k = min(int(self.plot_num_trajectories), N_sim)
        rand_idx = torch.randperm(N_sim, device=self.device)[:k]

        trajs = sim_obses.index_select(1, rand_idx).permute(1, 0, 2).detach().cpu().numpy()  # (k, T, D_s)
        seg_scores = scores.index_select(1, rand_idx).permute(1, 0).detach().cpu().numpy()    # (k, K)

        # Local imports keep matplotlib dependency out of hot paths when plotting is off.
        from dynamics_cotrain.visualization.scorer_plot import (
            plot_stepwise_evaluation, plot_stepwise_accept_reject,
        )
        import matplotlib.pyplot as plt
        import os

        metric_name = (
            f"Chunked MSE (softmax τ={self.temperature:.5f})"
            if self.scoring_mode == "softmax" else
            f"Chunked MSE (binary, thr={self.threshold:.5f})"
        )

        fig_eval = plot_stepwise_evaluation(
            trajectories=trajs,
            segment_scores=seg_scores,
            chunk_starts=list(chunk_starts),
            horizon=self._horizon,
            future_horizon=self._future_horizon,
            metric_name=metric_name,
            title=f"Cotrain resample {step_tag}",
            save_path=os.path.join(self.plot_dir, f"{step_tag}_stepwise.png"),
        )
        plt.close(fig_eval)

        fig_ar = plot_stepwise_accept_reject(
            trajectories=trajs,
            segment_scores=seg_scores,
            chunk_starts=list(chunk_starts),
            horizon=self._horizon,
            future_horizon=self._future_horizon,
            threshold=self.threshold,
            metric_name=metric_name,
            title=f"Cotrain resample {step_tag} (accept/reject @ threshold)",
            save_path=os.path.join(self.plot_dir, f"{step_tag}_accept_reject.png"),
        )
        plt.close(fig_ar)

    def _plot_chunk_pred_vs_actual(
        self,
        td: dict,
        scores: torch.Tensor,
        chunk_starts: list,
        step_tag: str,
    ) -> None:
        """For ``plot_num_trajectories`` random (env, chunk) pairs, render the
        scorer's predicted rollout against the actual state_gt rollout in 3D
        (dims 0:3). Mirrors the `_run_validation` helper in exp_dynamics."""
        K = scores.shape[0]
        N_sim = scores.shape[1]
        total_chunks = K * N_sim
        if total_chunks == 0:
            return
        n_plots = min(int(self.plot_num_trajectories), total_chunks)

        flat_pick = torch.randperm(total_chunks, device=self.device)[:n_plots]
        chunk_idx = (flat_pick // N_sim).cpu().tolist()   # index into chunk_starts
        env_idx = (flat_pick % N_sim).cpu().tolist()       # sim env index

        sim_obses = td["obses"].index_select(1, self.sim_env_idx)  # (T, N_sim, D_s)
        sim_actions = td["actions"].index_select(1, self.sim_env_idx)      # (T, N_sim, D_a)

        H = self._horizon
        F = self._future_horizon
        state_0_list, actions_list, state_gt_list = [], [], []
        for ci, ei in zip(chunk_idx, env_idx):
            t0 = chunk_starts[ci]
            state_0_list.append(sim_obses[t0:t0 + H, ei])
            actions_list.append(sim_actions[t0 + H - 1:t0 + H - 1 + F, ei])
            state_gt_list.append(sim_obses[t0 + H:t0 + H + F, ei])
        state_0 = torch.stack(state_0_list)     # (n_plots, H, D_s)
        actions = torch.stack(actions_list)     # (n_plots, F, D_a)
        state_gt = torch.stack(state_gt_list)   # (n_plots, F, D_s)

        # predict() handles normalize → algo → unnormalize, so mu comes back
        # in raw state space — directly comparable to state_gt.
        mu, log_var, _ = self.scorer.predict({"state_0": state_0, "actions": actions})

        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from dynamics_cotrain.visualization.scorer_plot import plot_3d_trajectory_with_uncertainty

        mu_np = mu.detach().cpu().numpy()
        log_var_np = log_var.detach().cpu().numpy()
        state_gt_np = state_gt.detach().cpu().numpy()
        state_0_np = state_0.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        for i in range(n_plots):
            init = state_0_np[i, -1:]                                    # (1, D_s)
            gt = np.concatenate([init, state_gt_np[i]], axis=0)          # (F+1, D_s)
            pred = np.concatenate([init, mu_np[i]], axis=0)              # (F+1, D_s)
            zero = np.zeros_like(init)
            var = np.concatenate([zero, log_var_np[i]], axis=0)
            ci, ei = chunk_idx[i], env_idx[i]
            score_val = float(scores_np[ci, ei])
            fig = plot_3d_trajectory_with_uncertainty(
                ground_truth=gt,
                predicted=pred,
                dim_indices=[0, 1, 2],
                dim_names=["X", "Y", "Z"],
                variance=var,
                title=(f"{step_tag} | env={ei} chunk={ci} "
                       f"(t0={chunk_starts[ci]}) MSE={score_val:.5f}"),
                figsize=(9, 7),
                save_path=os.path.join(
                    self.plot_dir, f"{step_tag}_pred_vs_actual_{i:02d}.png",
                ),
            )
            plt.close(fig)

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
