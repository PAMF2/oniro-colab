"""Empirical Godel relaxation gate.

Schmidhuber's Godel machine (arXiv:cs/0309048) demands a *provable* improvement before
self-rewrite. ONIRO relaxes this to multi-split empirical consistency (DGM/SICA style):

    ACCEPT iff (#splits_improved >= 7) AND (mean_delta > 0.5 * sigma_noise)
    UNDECIDED iff splits_improved in [5,6] (schedule more seeds)
    REJECT otherwise

sigma_noise is estimated weekly from no-op control runs (same data, different seed,
no mutation). This calibrates the gate against intrinsic stochastic-training noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Verdict = Literal["ACCEPT", "REJECT", "UNDECIDED"]


@dataclass
class GateDecision:
    verdict: Verdict
    splits_improved: int
    mean_delta: float
    sigma_noise: float
    threshold: float
    notes: str = ""


@dataclass
class GodelGate:
    n_splits: int = 10
    min_splits_improved: int = 7
    sigma_multiplier: float = 0.5
    undecided_range: tuple[int, int] = (5, 6)
    sigma_noise: float = 1e-6
    noise_history: list[float] = field(default_factory=list)

    def update_sigma_from_noop(self, control_deltas: np.ndarray | list[float]) -> float:
        """Refresh sigma_noise from a no-op control run (same data, different seed).

        control_deltas[i] = loss(seed_b, split_i) - loss(seed_a, split_i)
        """
        arr = np.asarray(control_deltas, dtype=np.float64)
        if arr.size < 2:
            return self.sigma_noise
        self.sigma_noise = float(np.std(arr, ddof=1))
        self.noise_history.append(self.sigma_noise)
        return self.sigma_noise

    def evaluate(
        self,
        baseline_losses: np.ndarray | list[float],
        candidate_losses: np.ndarray | list[float],
    ) -> GateDecision:
        """
        baseline_losses[i]:  predictive loss of theta_0 on OOD split i
        candidate_losses[i]: predictive loss of theta_1 on OOD split i

        Positive delta = candidate beat baseline on that split.
        """
        b = np.asarray(baseline_losses, dtype=np.float64)
        c = np.asarray(candidate_losses, dtype=np.float64)

        if b.shape != c.shape:
            return GateDecision("REJECT", 0, 0.0, self.sigma_noise, 0.0, "shape mismatch")
        if b.size != self.n_splits:
            return GateDecision(
                "REJECT", 0, 0.0, self.sigma_noise, 0.0,
                f"expected {self.n_splits} splits, got {b.size}",
            )
        if not (np.isfinite(b).all() and np.isfinite(c).all()):
            return GateDecision("REJECT", 0, 0.0, self.sigma_noise, 0.0, "nan/inf in losses")

        deltas = b - c
        improved = int((deltas > 0).sum())
        mean_delta = float(deltas.mean())
        threshold = self.sigma_multiplier * self.sigma_noise

        if improved >= self.min_splits_improved and mean_delta > threshold:
            return GateDecision(
                "ACCEPT", improved, mean_delta, self.sigma_noise, threshold,
                f"{improved}/{self.n_splits} improved, mean_delta={mean_delta:.4f} > {threshold:.4f}",
            )

        lo, hi = self.undecided_range
        if lo <= improved <= hi:
            return GateDecision(
                "UNDECIDED", improved, mean_delta, self.sigma_noise, threshold,
                "borderline; schedule additional seeds",
            )

        return GateDecision(
            "REJECT", improved, mean_delta, self.sigma_noise, threshold,
            f"insufficient: {improved}/{self.n_splits} improved, mean_delta={mean_delta:.4f}",
        )
