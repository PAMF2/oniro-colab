"""Action encoder for ARC-AGI-3 (v41.1).

ARC-3 actions: 5 discrete (ACTION1..5) + 1 CLICK(x, y) continuous 2D.
Fuse into a single d_model token to be appended to URM input sequence.
"""

from __future__ import annotations

import torch
from torch import nn


N_DISCRETE_ACTIONS = 5
ACTION_CLICK = 5         # special discrete id for "click" (treats x,y as args)
NULL_ACTION = 6          # for ARC-1/2 mode (no action)


class ActionEncoder(nn.Module):
    """(action_id, click_xy) → (B, 1, d_model) token.

    action_id: long (B,). Values 0..N_DISCRETE_ACTIONS-1 = discrete,
        N_DISCRETE_ACTIONS = click, NULL_ACTION = no action (ARC-1/2 mode).
    click_xy: float (B, 2) in [0, 1]. Ignored unless action_id == ACTION_CLICK.
    """

    def __init__(self, d_model: int = 896, hidden: int = 128,
                 n_actions: int = N_DISCRETE_ACTIONS + 2):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.action_embed = nn.Embedding(n_actions, hidden)
        self.click_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.fuse = nn.Linear(hidden * 2, d_model)

    def forward(self, action_id: torch.Tensor,
                click_xy: torch.Tensor | None = None) -> torch.Tensor:
        B = action_id.shape[0]
        a = self.action_embed(action_id.clamp(0, self.n_actions - 1))   # (B, h)
        if click_xy is None:
            click_xy = torch.zeros(B, 2, device=action_id.device)
        c = self.click_mlp(click_xy.float())                              # (B, h)
        tok = self.fuse(torch.cat([a, c], dim=-1))                        # (B, d)
        return tok.unsqueeze(1)                                            # (B, 1, d)
