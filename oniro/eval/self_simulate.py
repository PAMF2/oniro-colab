"""Problem-level self-simulate verification (v40.1).

Implements Self-Execution Sim (arxiv:2604.03253) + FFDC Imagination Trust
(arxiv:2605.06222) at the TASK level, exactly per Pedro's clarification
"self simulate no nível de problema".

For an ARC task with N demo pairs and 1 test input, the model runs:
  1. For each demo (di, do), apply augmentation a -> predict pred = f(di_a),
     score = cell_acc(pred, do_a).
  2. Aggregate demo scores -> self_simulate_score(a) ∈ [0, 1].
  3. Score weights the test-time vote of augmentation a.

Augmentations with score < threshold are filtered out (model can't reproduce
demos under that augmentation, so its test prediction is untrusted).

Public functions:
    self_simulate_score(forward_fn, demos, aug_fn, aug_id, grid_size) -> float
    weighted_tta_majority(forward_fn, test_grid, demos, aug_fns, n_samples,
                          threshold=0.7, grid_size, n_colors, fallback="greedy")
        -> (H, W) int prediction
"""

from __future__ import annotations

from typing import Callable

import torch


def self_simulate_score(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    demos: list[tuple[torch.Tensor, torch.Tensor]],
    aug_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    aug_id: int = 0,
    grid_size: int = 30,
) -> float:
    """How well does the model reproduce demo outputs from demo inputs?

    Args:
        forward_fn: takes a grid (H, W) int and returns the predicted grid
                    (H, W) int. May be a closure that also handles augmentation
                    inversion.
        demos: list of (di, do) tensor pairs, each (H, W) int.
        aug_fn: optional function (grid, aug_id) -> aug_grid. If None, identity.
        aug_id: id passed to aug_fn for forward (and for inverting if needed).
        grid_size: target grid size for cell_acc denominator.

    Returns:
        Mean cell-accuracy across demos in [0, 1].
    """
    if not demos:
        return 0.0
    total = 0.0
    for di, do in demos:
        if aug_fn is not None:
            di_in = aug_fn(di, aug_id)
            do_target = aug_fn(do, aug_id)
        else:
            di_in = di
            do_target = do
        pred = forward_fn(di_in)
        # crop/pad mismatched shapes safely
        h = min(pred.shape[-2], do_target.shape[-2])
        w = min(pred.shape[-1], do_target.shape[-1])
        match = (pred[..., :h, :w] == do_target[..., :h, :w]).float().mean().item()
        total += float(match)
    return total / len(demos)


def weighted_tta_majority(
    forward_fn_with_aug: Callable[[torch.Tensor, int], tuple[torch.Tensor, float]],
    test_grid: torch.Tensor,
    n_colors: int,
    grid_size: int,
    n_samples: int = 128,
    threshold: float = 0.7,
    device: str = "cuda",
) -> torch.Tensor:
    """Weighted majority vote over n_samples test-time augmentations.

    forward_fn_with_aug(test_grid, aug_idx) must return:
        (pred_in_canonical_frame: (H, W) int,  score: float in [0, 1])
    where `score` is the self-simulate score for this augmentation.

    Augmentations with score < threshold are dropped from the vote (their
    weight set to 0). If ALL samples are below threshold, we fall back to
    plain majority (threshold=0).
    """
    votes = torch.zeros(n_colors, grid_size, grid_size, device=device)
    kept = 0
    preds_for_fallback = []
    for i in range(n_samples):
        pred, score = forward_fn_with_aug(test_grid, i)
        preds_for_fallback.append((pred, score))
        if score < threshold:
            continue
        # score acts as the weight
        for c in range(n_colors):
            votes[c] += score * (pred == c).float()
        kept += 1

    if kept == 0:
        # fall back: plain majority over everything
        for pred, _ in preds_for_fallback:
            for c in range(n_colors):
                votes[c] += (pred == c).float()
    return votes.argmax(dim=0)
