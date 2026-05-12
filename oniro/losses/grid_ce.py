"""Grid cross-entropy loss with class-weighting for ARC palette.

Cor 0 (background) dominates ARC grids ~80%. Naive CE collapses to all-zeros
prediction. Down-weight class 0 (default 0.2x) and up-weight classes 1-9.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grid_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bg_weight: float = 0.2,
    n_colors: int = 10,
) -> torch.Tensor:
    """
    logits: (B, n_colors, H, W)
    target: (B, H, W) int64 with values in [0, n_colors-1]
    bg_weight: weight applied to class 0 (background). Classes 1..n-1 keep weight 1.
    """
    w = torch.ones(n_colors, device=logits.device)
    w[0] = bg_weight
    return F.cross_entropy(logits, target, weight=w)
