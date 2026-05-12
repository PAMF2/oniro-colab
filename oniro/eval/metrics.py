"""Eval metrics: slot purity (ARI proxy), action-conditioned forward IoU."""

from __future__ import annotations

import torch


def slot_purity(slot_assignments: torch.Tensor, gt_segmentation: torch.Tensor) -> float:
    """Adjusted Rand Index between per-pixel slot assignment and GT segmentation.

    slot_assignments: (B, H, W) int — which slot each pixel went to (argmax of attn).
    gt_segmentation:  (B, H, W) int — ground truth object id (0..N-1).
    """
    B = slot_assignments.shape[0]
    aris: list[float] = []
    for b in range(B):
        a = slot_assignments[b].flatten().cpu().numpy()
        g = gt_segmentation[b].flatten().cpu().numpy()
        n = len(a)
        if n == 0:
            continue
        from collections import Counter
        ca = Counter(a)
        cg = Counter(g)
        cab = Counter(zip(a.tolist(), g.tolist()))
        sum_comb = sum(v * (v - 1) // 2 for v in cab.values())
        sum_a = sum(v * (v - 1) // 2 for v in ca.values())
        sum_g = sum(v * (v - 1) // 2 for v in cg.values())
        total = n * (n - 1) // 2
        if total == 0:
            continue
        expected = (sum_a * sum_g) / total
        max_index = (sum_a + sum_g) / 2
        if max_index - expected == 0:
            aris.append(0.0)
        else:
            aris.append((sum_comb - expected) / (max_index - expected))
    return sum(aris) / max(len(aris), 1)


def action_cond_iou(pred_slots: torch.Tensor, target_slots: torch.Tensor, eps: float = 1e-8) -> float:
    """Cosine-based IoU between predicted and target slot vectors, averaged over batch+slots."""
    pn = pred_slots / (pred_slots.norm(dim=-1, keepdim=True) + eps)
    tn = target_slots / (target_slots.norm(dim=-1, keepdim=True) + eps)
    sim = (pn * tn).sum(dim=-1).clamp(min=0.0)
    return float(sim.mean())
