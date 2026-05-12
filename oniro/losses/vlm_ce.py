"""VLM cross-entropy loss for the auxiliary language-grounding head."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vlm_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  (B, T, V)
    targets: (B, T)
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )
