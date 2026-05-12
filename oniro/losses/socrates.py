"""Socrates Loss (arxiv:2604.12245).

Adds an extra "UNKNOWN" class to the output, with a dynamic uncertainty
penalty that encourages the model to either be confident OR predict unknown,
rather than be wrong-confident.

Adapted from per-token classification to per-pixel grid classification.

Output logits expected shape: (B, n_colors + 1, H, W), where the last class
is UNKNOWN.

Loss:
    L = CE(logits, targets) + alpha * uncertainty_penalty
        - if target is real color c, CE chooses logit[c] (one of 10 colors)
        - UNKNOWN class is never the ground-truth target during supervised
          training - it's a slack class used only at inference for low-conf
          predictions
    uncertainty_penalty = -beta * mean( max_softmax_excluding_unknown )
        - encourages high confidence among real colors when not unknown
    OR
    can fold UNKNOWN as a learned slack via:
        L = -log( p(target) + p(unknown) * gamma )
        gamma=0 -> standard CE; gamma>0 -> reward predicting unknown
        as partial credit when uncertain (Socrates style).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def socrates_grid_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_colors: int = 10,
    unknown_class: int = 10,
    gamma: float = 0.05,
    bg_weight: float = 0.15,
    safe_softmax: bool = False,
) -> torch.Tensor:
    """Socrates-style CE for grid classification.

    Args:
        logits: (B, n_colors + 1, H, W) - last channel is UNKNOWN
        target: (B, H, W) ints in [0, n_colors-1]
        gamma:  partial credit for predicting UNKNOWN when target is real.
                gamma=0 collapses to standard CE; gamma>0 rewards calibrated
                uncertainty.
        bg_weight: scale factor for the background class (target==0)
        safe_softmax: when True, subtract per-pixel max from logits before
                       softmax for extra numerical stability. F.log_softmax
                       does this internally already; this flag is mainly a
                       v40.2 toggle to make the behaviour explicit.

    Returns: scalar loss.
    """
    B, C, H, W = logits.shape
    assert C == n_colors + 1, f"expected C={n_colors+1}, got {C}"
    if safe_softmax:
        logits = logits - logits.max(dim=1, keepdim=True).values
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    # gather p(target) and p(unknown)
    tgt_one_hot = F.one_hot(target.clamp(0, n_colors - 1), num_classes=C).permute(0, 3, 1, 2).float()
    p_tgt = (probs * tgt_one_hot).sum(dim=1)  # (B, H, W)
    p_unk = probs[:, unknown_class]            # (B, H, W)
    # Socrates: accept full credit on target OR partial credit (gamma) on unknown
    combined = p_tgt + gamma * p_unk
    # log of combined, clamped for stability
    loss_pix = -torch.log(combined.clamp_min(1e-8))
    # weight: bg cells down-weighted
    bg_mask = (target == 0).float()
    w = bg_mask * bg_weight + (1.0 - bg_mask) * 1.0
    return (loss_pix * w).sum() / w.sum().clamp_min(1.0)


def socrates_argmax(logits: torch.Tensor, n_colors: int = 10,
                    unknown_class: int = 10,
                    unknown_threshold: float = 0.5) -> torch.Tensor:
    """At inference, if argmax falls on UNKNOWN, fall back to argmax over real colors.

    logits: (B, n_colors+1, H, W). Returns (B, H, W) int prediction in [0, n_colors-1].
    """
    probs = F.softmax(logits, dim=1)
    # argmax over real colors only
    real_pred = probs[:, :n_colors].argmax(dim=1)
    # Optional: if want to mark unknown cells explicitly, threshold on p(unknown)
    # but for final grid we must commit to a real color, so always fall back.
    return real_pred
