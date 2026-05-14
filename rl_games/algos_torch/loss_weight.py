"""Per-cell loss-weighting helper for A2CLossWeightAgent.

Implements the master per-sample weight rule:

    w_i = I_i + (1 - I_i) * M_i * β

where:
    I_i      ∈ {0,1}: 1 if sample i comes from a real env, else 0
    M_i      ∈ {0,1}: binary scorer acceptance for sample i (real cells are 1
                     by construction in the buffer's loss_mask; the helper
                     excludes real cells from the β denominator using I_i)
    β        = n_real / n_sim_accept,  0 when n_sim_accept = 0

See docs/superpowers/specs/2026-05-13-loss-weight-cotrain-design.md.
"""
from __future__ import annotations

from typing import Tuple

import torch


def compute_loss_weights(
    is_real: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """Return (w, β, denom) for a flattened PPO minibatch.

    Args:
        is_real:   (B,) or (B, 1) bool — True for real-env samples.
        loss_mask: (B,) or (B, 1) float in {0.0, 1.0} — scorer acceptance.
                   Real cells are 1.0 by buffer construction; this is fine
                   because real cells use I_i, not (1-I_i)*M_i*β.

    Returns:
        w:     (B,) detached float weights, same flat layout as inputs.
        β:     python float, the detached balancing multiplier.
        denom: detached scalar tensor = w.sum(), for the weighted-mean
               reduction. Caller is responsible for skipping the optimizer
               step when denom == 0.
    """
    assert is_real.dtype == torch.bool, f"is_real must be bool, got {is_real.dtype}"
    is_real = is_real.reshape(-1).detach()
    loss_mask = loss_mask.reshape(-1).detach().to(torch.float32)
    assert is_real.shape == loss_mask.shape, (
        f"is_real {tuple(is_real.shape)} and loss_mask {tuple(loss_mask.shape)} "
        f"must have matching flat shape"
    )
    # Spec contract: loss_mask is binary {0, 1}. Non-binary mask would mean
    # the scorer ran in softmax mode, which is not supported by loss_weight.
    unique = torch.unique(loss_mask)
    assert torch.all((unique == 0.0) | (unique == 1.0)), (
        f"loss_mask must be binary {{0,1}}; got unique values {unique.tolist()}. "
        f"loss_weight mode requires scoring_mode='binary'."
    )

    I = is_real.to(torch.float32)
    not_real = 1.0 - I

    n_real = I.sum()
    n_sim_accept = (not_real * loss_mask).sum()

    if n_sim_accept.item() > 0.0:
        beta_t = n_real / n_sim_accept
    else:
        beta_t = torch.zeros((), dtype=torch.float32, device=loss_mask.device)

    w = I + not_real * loss_mask * beta_t
    denom = w.sum().detach()
    return w.detach(), float(beta_t.item()), denom
