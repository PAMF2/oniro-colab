"""VICReg variance-covariance regularization (arXiv:2105.04906).

Prevents JEPA latent collapse. Variance term hinges per-dim std to a target;
covariance term penalises off-diagonal correlations between dimensions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vicreg(
    z: torch.Tensor,
    std_target: float = 1.0,
    eps: float = 1e-4,
    cov_weight: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    z: (B, D) batch of latent vectors
    returns: (var_loss, cov_loss)
    """
    assert z.ndim == 2, "vicreg expects (B, D)"
    B, D = z.shape

    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(std_target - std).mean()

    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(B - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = cov_weight * (off_diag ** 2).sum() / D

    return var_loss, cov_loss
