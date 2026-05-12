"""TopK SAE training loss: reconstruction + L1 sparsity on the sparse code."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    sparse_features: torch.Tensor,
    l1_alpha: float = 5e-4,
) -> dict[str, torch.Tensor]:
    """
    x:               (..., d_in) ground-truth slot activations (EMA-frozen)
    recon:           (..., d_in) SAE reconstruction
    sparse_features: (..., dict_size) top-K activations

    returns dict: {recon, l1, total}
    """
    rec = F.mse_loss(recon, x)
    l1 = sparse_features.abs().sum(dim=-1).mean()
    total = rec + l1_alpha * l1
    return {"recon": rec, "l1": l1, "total": total}
