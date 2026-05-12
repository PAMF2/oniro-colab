"""InfoNCE contrastive loss for slot embeddings.

Forces slot embeddings from different tasks to diverge while pulling matched
input/output pairs together. Operates on flattened slot vectors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    anchors:   (B, D)
    positives: (B, D)  positive[i] is the positive for anchor[i].
    Other anchors in the batch serve as negatives.
    """
    a = F.normalize(anchors, dim=-1)
    p = F.normalize(positives, dim=-1)
    logits = a @ p.T / max(temperature, 1e-3)
    targets = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(logits, targets)


def slot_infonce(
    slots_a: torch.Tensor,
    slots_p: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pool slots (mean across K), then InfoNCE."""
    a = slots_a.mean(dim=1)
    p = slots_p.mean(dim=1)
    return info_nce(a, p, temperature=temperature)
