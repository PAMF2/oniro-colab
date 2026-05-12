"""AlphaEvolve-style outer loop with Gödel acceptance gate.

Periodically during training, sample N candidate weight mutations, evaluate each
on a held-out batch, and ACCEPT the best ONLY if its score strictly improves on
the unmutated baseline. Accepted mutations are appended to an archive — exactly
the Gödel relaxation applied at the level of weight modifications, not edits.

Distinct from AlphaEvolve (which evolves code via LLM) and from CMA-ES (which
evolves continuous parameters via covariance). Here we directly perturb a
chosen weight tensor with gaussian noise scaled by its own std, and keep only
strictly-better outcomes — the simplest possible empirical Gödel machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import random

import torch


@dataclass
class MutationRecord:
    param_name: str
    delta_norm: float
    score_before: float
    score_after: float


@dataclass
class AlphaEvolveGodelArchive:
    accepted: list[MutationRecord] = field(default_factory=list)
    rejected: int = 0
    failed_param_count: dict[str, int] = field(default_factory=dict)
    accepted_param_count: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "accepted": len(self.accepted),
            "rejected": self.rejected,
            "best_score": max((r.score_after for r in self.accepted), default=float("nan")),
            "mean_lift": (
                sum(r.score_after - r.score_before for r in self.accepted)
                / max(len(self.accepted), 1)
            ),
            "params_succeeded": dict(self.accepted_param_count),
            "params_failed": dict(self.failed_param_count),
        }

    def sample_weight(self, param_name: str) -> float:
        """Weight for sampling this param: higher when historically successful."""
        s = self.accepted_param_count.get(param_name, 0)
        f = self.failed_param_count.get(param_name, 0)
        return (s + 1) / (s + f + 2)  # laplace smoothing


def _eligible_params(
    model: torch.nn.Module, name_filter: str | None = None,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Pick params worth mutating: weight matrices of Linear/Conv layers.

    name_filter: substring filter on param name (e.g. 'micro_learner' to mutate
    only the micro sub-net). None = all eligible params.
    """
    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            continue
        if min(p.shape) < 4:
            continue
        if name_filter is not None and name_filter not in name:
            continue
        out.append((name, p))
    return out


@torch.no_grad()
def alphaevolve_godel_round(
    model: torch.nn.Module,
    score_fn: Callable[[], float],
    n_candidates: int = 3,
    sigma: float = 1e-3,
    archive: AlphaEvolveGodelArchive | None = None,
    rng: random.Random | None = None,
    name_filter: str | None = None,
) -> tuple[bool, float, float, AlphaEvolveGodelArchive]:
    """One AlphaEvolve-Gödel round.

    1. Get baseline score.
    2. Sample N candidate weight mutations on random eligible params.
    3. Apply, eval, revert each.
    4. Accept the strictly-best candidate iff it beats baseline.
    Returns (accepted, baseline_score, best_score, archive).
    """
    archive = archive or AlphaEvolveGodelArchive()
    rng = rng or random.Random()
    params = _eligible_params(model, name_filter=name_filter)
    if not params:
        return False, 0.0, 0.0, archive

    baseline = float(score_fn())
    best_score = baseline
    best_choice: tuple[str, torch.nn.Parameter, torch.Tensor] | None = None

    weights = [archive.sample_weight(n) for n, _ in params]
    tried_param_names: list[str] = []

    for _ in range(n_candidates):
        # Failure-aware sampling: bias toward params with prior success
        name, p = rng.choices(params, weights=weights, k=1)[0]
        tried_param_names.append(name)
        scale = max(float(p.std().item()), 1e-6)
        delta = torch.randn_like(p) * sigma * scale
        p.data.add_(delta)
        try:
            s = float(score_fn())
        finally:
            p.data.sub_(delta)
        if s > best_score:
            best_score = s
            best_choice = (name, p, delta.clone())

    if best_choice is None:
        archive.rejected += 1
        for n in tried_param_names:
            archive.failed_param_count[n] = archive.failed_param_count.get(n, 0) + 1
        return False, baseline, best_score, archive

    name, p, delta = best_choice
    p.data.add_(delta)
    archive.accepted.append(MutationRecord(
        param_name=name,
        delta_norm=float(delta.norm().item()),
        score_before=baseline,
        score_after=best_score,
    ))
    archive.accepted_param_count[name] = archive.accepted_param_count.get(name, 0) + 1
    # also penalize the unsuccessful tried params from this round
    for n in tried_param_names:
        if n != name:
            archive.failed_param_count[n] = archive.failed_param_count.get(n, 0) + 1
    return True, baseline, best_score, archive
