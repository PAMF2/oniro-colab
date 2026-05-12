"""Evolutionary latent search with Gödel-gated acceptance.

Pillar: **evolutionary** + **Gödel**.

Per task, evolve a population of latent z vectors in the dynamics conditioning
space. Score each candidate by grid-match on the task's demo pairs. Mutate via
gaussian noise, select elites, crossover. Each generation, the new elite is
ACCEPTED only if its score is strictly better than the parent (Gödel gate
relaxed to empirical fitness on held-out demos).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class EvolveResult:
    best_z: torch.Tensor
    best_score: float
    history: list[float]


class EvolutionaryLatentSearch:
    """CMA-ES-lite over latent z. Diagonal covariance, gaussian mutation."""

    def __init__(
        self,
        z_dim: int,
        pop_size: int = 32,
        n_elite: int = 4,
        sigma_init: float = 0.5,
        sigma_decay: float = 0.95,
        device: str = "cuda",
        rng_seed: int = 0,
    ):
        self.z_dim = z_dim
        self.pop_size = pop_size
        self.n_elite = max(1, min(n_elite, pop_size))
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(rng_seed)

    def _sample_initial(self) -> torch.Tensor:
        return torch.randn(self.pop_size, self.z_dim, generator=self.gen).to(self.device)

    def search(
        self,
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        n_generations: int = 20,
        z_init: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> EvolveResult:
        """
        score_fn: takes (pop, z_dim) and returns (pop,) fitness (higher is better).
        Returns the best z found across all generations + history.
        """
        pop = z_init.clone() if z_init is not None else self._sample_initial()
        if pop.shape != (self.pop_size, self.z_dim):
            raise ValueError(f"pop must be ({self.pop_size}, {self.z_dim})")

        sigma = self.sigma_init
        best_z = pop[0].clone()
        best_score = -float("inf")
        history: list[float] = []

        for g in range(n_generations):
            with torch.no_grad():
                scores = score_fn(pop).detach()
            top_vals, top_idx = scores.topk(self.n_elite, largest=True)
            elites = pop[top_idx]

            gen_best = float(top_vals[0].item())
            history.append(gen_best)
            # Gödel-relaxed accept: only update best if strictly better
            if gen_best > best_score:
                best_score = gen_best
                best_z = pop[top_idx[0]].clone()

            # Build next population: keep elites, mutate around their mean
            mean = elites.mean(dim=0, keepdim=True)
            noise = torch.randn(self.pop_size - self.n_elite, self.z_dim,
                                generator=self.gen).to(self.device) * sigma
            children = mean + noise
            pop = torch.cat([elites, children], dim=0)
            sigma *= self.sigma_decay

            if verbose:
                print(f"gen {g:3d}  best={best_score:.4f}  σ={sigma:.4f}")

        return EvolveResult(best_z=best_z, best_score=best_score, history=history)
