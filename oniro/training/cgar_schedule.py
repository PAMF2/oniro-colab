"""CGAR — Curriculum-Guided Adaptive Recursion (v40.2).

Implements two ideas from arxiv:2511.08653 (Qasim & Zhang):

1. Progressive Depth Curriculum (PDC): start training with a SHALLOW URM
   (n_loops_eff = 1/3 of full) and ramp to full depth over a fraction of
   training. Saves compute on Phase A0 and helps early features converge.

2. Hierarchical Supervision Weighting (HSW): instead of the v37 fixed
   `1.5^(-(n_cycles-2-t))` weights, use a schedule that starts UNIFORM
   across cycles (encourages every cycle to learn) and SHIFTS to
   late-cycle weighting (consolidates final predictions) over training.
   Yields ~40% gradient variance reduction per the paper.
"""

from __future__ import annotations

import numpy as np


def pdc_loops(step: int, total_steps: int,
              n_loops_full: int = 12,
              shallow_frac: float = 0.20,
              mid_frac: float = 0.50) -> int:
    """Three-stage Progressive Depth Curriculum.

    Stage A0 (0 .. shallow_frac):  n_loops_eff = full / 3
    Stage A1 (shallow_frac .. mid_frac): n_loops_eff = 2 * full / 3
    Stage A2 (mid_frac .. 1.0):     n_loops_eff = full

    Total compute saved on Phase A by running shallower depth in early stages.
    """
    frac = step / max(total_steps, 1)
    if frac < shallow_frac:
        return max(1, n_loops_full // 3)
    if frac < mid_frac:
        return max(1, (2 * n_loops_full) // 3)
    return n_loops_full


def hsw_weights(n_cycles: int, step: int, total_steps: int,
                 decay: float = 0.5, ramp_frac: float = 0.5) -> np.ndarray:
    """Hierarchical Supervision Weighting.

    Schedule mixes UNIFORM weights (early) with late-cycle-heavy weights
    (late). Mix factor = min(1.0, step / (ramp_frac * total_steps)).

    Returns a (n_cycles,) numpy array summing to 1.0.
    """
    if n_cycles <= 0:
        return np.array([1.0])
    schedule = min(1.0, step / max(1.0, ramp_frac * total_steps))
    uniform = np.full(n_cycles, 1.0 / n_cycles)
    raw = np.array([decay ** (n_cycles - 1 - i) for i in range(n_cycles)])
    raw = raw / raw.sum()
    w = uniform * (1.0 - schedule) + raw * schedule
    return w / w.sum()
