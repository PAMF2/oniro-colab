"""AIRV — Augment-Inference-Reverse-Vote ensemble for ARC eval.

MindsAI trick (ARC Prize 2024). For each test input:
    1. Apply 8 dihedral augmentations (rotations + flips).
    2. Run model on each augmented input → 8 candidate outputs.
    3. Reverse-augment each output back to original orientation.
    4. Per-pixel majority vote across 8 reversed outputs.

Deterministic. Adds 8x inference cost but no training. Documented lift on ARC.
"""

from __future__ import annotations

import torch


# 8 dihedral group operations as (rotate_k, flip_h)
DIHEDRAL_OPS = [
    (0, False), (1, False), (2, False), (3, False),
    (0, True),  (1, True),  (2, True),  (3, True),
]


def _apply_op(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """x: (..., H, W) tensor. k = rot90 count, flip = mirror H."""
    out = torch.rot90(x, k=k, dims=(-2, -1))
    if flip:
        out = torch.flip(out, dims=(-1,))
    return out


def _reverse_op(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Inverse of _apply_op."""
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, k=-k, dims=(-2, -1))


@torch.no_grad()
def airv_self_consistency_predict(
    predict_fn,
    image_t: torch.Tensor,
    grid_size: int,
    n_colors: int = 10,
    n_samples: int = 4,
    enable_dropout_fn=None,
) -> torch.Tensor:
    """AIRV + self-consistency: 8 augmentations × n_samples each (with dropout
    enabled if `enable_dropout_fn` provided), majority vote over all outputs.

    enable_dropout_fn: callable(bool) toggling model.train() state for dropout.
    Returns: (grid_size, grid_size) majority-voted predicted grid.
    """
    votes = torch.zeros(n_colors, grid_size, grid_size, device=image_t.device)
    for k, flip in DIHEDRAL_OPS:
        aug_img = _apply_op(image_t, k, flip)
        for s in range(n_samples):
            if enable_dropout_fn is not None:
                enable_dropout_fn(s > 0)        # first sample = deterministic, rest stochastic
            logits = predict_fn(aug_img)
            pred = logits.argmax(dim=0)
            rev_pred = _reverse_op(pred, k, flip)
            for c in range(n_colors):
                votes[c] += (rev_pred == c).float()
    if enable_dropout_fn is not None:
        enable_dropout_fn(False)
    return votes.argmax(dim=0)


@torch.no_grad()
def beam_search_predict(
    logits: torch.Tensor,
    grid_in: torch.Tensor | None = None,
    beam_width: int = 3,
    consistency_threshold: float = 0.5,
) -> torch.Tensor:
    """
    logits: (n_colors, H, W) decoder logits for one sample.
    grid_in: (H, W) optional input grid for identity fallback when low confidence.

    Top-k argmax per pixel. Pick most-likely; if confidence below threshold AND
    grid_in present, fall back to grid_in's pixel value.
    """
    n_colors, H, W = logits.shape
    probs = logits.softmax(dim=0)
    top_p, top_i = probs.topk(beam_width, dim=0)
    pred = top_i[0]
    if grid_in is not None and grid_in.shape == pred.shape:
        confidence = top_p[0]
        low_conf = confidence < consistency_threshold
        pred = torch.where(low_conf, grid_in.to(pred.device), pred)
    return pred


@torch.no_grad()
def airv_predict_grid(
    predict_fn,
    image_t: torch.Tensor,
    grid_size: int,
    n_colors: int = 10,
) -> torch.Tensor:
    """
    predict_fn: callable(image) -> (n_colors, grid_size, grid_size) logits
    image_t:    (3, H, W) input image tensor
    Returns: (grid_size, grid_size) majority-voted predicted grid.
    """
    votes = torch.zeros(n_colors, grid_size, grid_size, device=image_t.device)
    for k, flip in DIHEDRAL_OPS:
        aug_img = _apply_op(image_t, k, flip)
        logits = predict_fn(aug_img)                       # (n_colors, gs, gs)
        pred = logits.argmax(dim=0)                        # (gs, gs)
        rev_pred = _reverse_op(pred, k, flip)              # (gs, gs)
        # Add 1-vote at each pixel for the predicted color
        for c in range(n_colors):
            votes[c] += (rev_pred == c).float()
    return votes.argmax(dim=0)
