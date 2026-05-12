"""Curiosity ensemble (Pathak-style RND/disagreement).

K small bootstrapped predictors of next-slot from (slot, action). Variance across
members is the intrinsic reward — high disagreement = under-explored transition.
Each member trains on a different bootstrap sample of the replay batch.
"""

from __future__ import annotations

import torch
from torch import nn


class _CuriosityHead(nn.Module):
    def __init__(self, slot_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, slot_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a.unsqueeze(1).expand(-1, s.shape[1], -1)], dim=-1)
        return self.net(x)


class CuriosityEnsemble(nn.Module):
    def __init__(
        self,
        K: int = 5,
        slot_dim: int = 128,
        action_dim: int = 1024,
        hidden: int = 512,
    ):
        super().__init__()
        self.K = K
        self.members = nn.ModuleList(
            [_CuriosityHead(slot_dim, action_dim, hidden) for _ in range(K)]
        )

    def predict_all(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """(B, K_slots, slot_dim) → stacked predictions (K_ensemble, B, K_slots, slot_dim)."""
        return torch.stack([m(s, a) for m in self.members], dim=0)

    def intrinsic_reward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Variance across ensemble = uncertainty bonus, returned per-batch scalar."""
        preds = self.predict_all(s, a)
        var = preds.var(dim=0)
        return var.mean(dim=(-1, -2))

    def loss(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Per-member MSE on detached target. Returned as mean loss across members."""
        target = s_next.detach()
        total = s.new_zeros(())
        for m in self.members:
            total = total + ((m(s, a) - target) ** 2).mean()
        return total / self.K
