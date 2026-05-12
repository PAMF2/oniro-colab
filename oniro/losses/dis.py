"""Deep Improvement Supervision (DIS) — per-cycle target generation + loss.

Paper: arxiv:2511.16886 ("Your Latent Reasoning is Secretly Policy Improvement
Operator"). Each recursive cycle is supervised against a progressively cleaner
version of the ground-truth grid, forcing each forward pass to act as a
policy-improvement step rather than a black-box recurrence.

Novel extension in ONIRO: the corruption is applied at *training time only*,
acting as a curriculum over noise levels, and the decoder operates on slot
states at each cycle — making this slot-level discrete diffusion, not pixel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_dis_targets(
    grid_out: torch.Tensor,
    n_cycles: int,
    n_colors: int = 10,
    max_corruption: float = 0.7,
    min_corruption: float = 0.0,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Generate per-cycle corrupted targets.

    grid_out: (B, H, W) int64 ground-truth
    n_cycles: how many recursive cycles
    Returns: list of n_cycles tensors. cycle 0 = max_corruption, cycle N-1 = min.
    """
    if n_cycles < 1:
        return [grid_out]
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
    else:
        gen = None

    targets: list[torch.Tensor] = []
    B, H, W = grid_out.shape
    for c in range(n_cycles):
        if n_cycles == 1:
            corruption = min_corruption
        else:
            corruption = max_corruption - (max_corruption - min_corruption) * c / (n_cycles - 1)
        if corruption <= 0:
            targets.append(grid_out.clone())
            continue
        noise_kwargs = {"generator": gen} if gen is not None else {}
        mask = torch.rand(B, H, W, **noise_kwargs).to(grid_out.device) < corruption
        random_colors = torch.randint(0, n_colors, (B, H, W), **noise_kwargs).to(grid_out.device)
        corrupted = torch.where(mask, random_colors, grid_out)
        targets.append(corrupted)
    return targets


def dis_loss(
    grid_logits_per_cycle: list[torch.Tensor],
    grid_out: torch.Tensor,
    bg_weight: float = 0.15,
    n_colors: int = 10,
    max_corruption: float = 0.7,
    cycle_weight_alpha: float = 1.5,
    seed: int | None = None,
) -> dict:
    """Per-cycle CE against corruption-schedule targets.

    Final cycle has weight=1, earlier cycles have geometrically smaller weights
    (alpha^(distance_from_final)) so the model gets a smooth curriculum.
    """
    n_cycles = len(grid_logits_per_cycle)
    targets = make_dis_targets(grid_out, n_cycles, n_colors, max_corruption, seed=seed)

    w = torch.ones(n_colors, device=grid_out.device)
    w[0] = bg_weight

    total = grid_logits_per_cycle[0].new_zeros(())
    per_cycle_losses: list[float] = []
    for t, (logits, target) in enumerate(zip(grid_logits_per_cycle, targets)):
        ce = F.cross_entropy(logits, target, weight=w)
        weight = cycle_weight_alpha ** (-(n_cycles - 1 - t))
        total = total + weight * ce
        per_cycle_losses.append(float(ce.detach()))
    return {
        "total": total,
        "per_cycle": per_cycle_losses,
        "n_cycles": n_cycles,
    }
