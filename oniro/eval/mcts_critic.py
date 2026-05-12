"""Learned MCTS critic — small MLP that scores (state, candidate_pred).

Replaces the hand-rolled self_simulate scoring as a learnable signal for the
MCTS search at eval time. Trained offline against demo agreement: given the
URM final state and a candidate predicted grid, predict the demo-fit score.

Lightweight (~50k params), can be plugged into oniro/eval/mcts_search.py
when available.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MCTSCritic(nn.Module):
    def __init__(self, d_model: int = 768, n_colors: int = 10,
                 grid_size: int = 30, hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_colors = n_colors
        self.grid_size = grid_size
        self.hidden = hidden

        # state side: mean of URM final state
        self.state_proj = nn.Linear(d_model, hidden)
        # candidate side: one-hot grid -> small CNN summary
        self.cand_conv = nn.Sequential(
            nn.Conv2d(n_colors, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cand_proj = nn.Linear(32, hidden)
        # fusion
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, urm_state: torch.Tensor, candidate_grid: torch.Tensor) -> torch.Tensor:
        """urm_state: (B, T, d_model). candidate_grid: (B, H, W) int in [0, n_colors-1].
        Returns (B,) sigmoid score in [0, 1].
        """
        s = self.state_proj(urm_state.mean(dim=1))
        # one-hot the candidate
        B, H, W = candidate_grid.shape
        oh = F.one_hot(candidate_grid.clamp(0, self.n_colors - 1),
                        num_classes=self.n_colors).permute(0, 3, 1, 2).float()
        c = self.cand_conv(oh).flatten(1)
        c = self.cand_proj(c)
        h = torch.cat([s, c], dim=-1)
        return torch.sigmoid(self.head(h).squeeze(-1))
