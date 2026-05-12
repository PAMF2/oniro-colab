"""JEPA loss (Joint-Embedding Predictive Architecture).

Predicts next-step latent representation, stop-grad on target. VICReg on current
slots prevents collapse. The action-conditioned predictor lives in dynamics_mamba.py;
this file only defines the loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from oniro.losses.vicreg import vicreg


def jepa_loss(
    predicted_next: torch.Tensor,
    target_next: torch.Tensor,
    slots_now: torch.Tensor,
    vicreg_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    predicted_next: (B, K, d) D(slots_t, a_t)
    target_next:    (B, K, d) encoder(x_{t+1}), stop-grad applied internally
    slots_now:      (B, K, d) used only for VICReg term

    returns dict: {pred, var, cov, total}
    """
    pred_loss = F.mse_loss(predicted_next, target_next.detach())

    z_flat = slots_now.flatten(0, 1)
    var_l, cov_l = vicreg(z_flat)

    total = pred_loss + vicreg_weight * (var_l + cov_l)
    return {"pred": pred_loss, "var": var_l, "cov": cov_l, "total": total}
