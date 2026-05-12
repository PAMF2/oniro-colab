"""HRM-style two-speed module wrapper.

Splits a recursive cycle into:
- L (fast worker): runs every cycle, takes (slot_state, H_state, action_emb).
- H (slow planner): runs every K cycles, takes pooled L state, updates H.

H provides a slowly-evolving global context that L conditions on. Forces
hierarchy of reasoning timescales — proven 40% ARC-AGI-1 with 27M params (HRM
arxiv:2506.21734).

Drops in over an existing dynamics core; the wrapper applies multi-cycle
recursion with the H/L split internally.
"""

from __future__ import annotations

import torch
from torch import nn


class TwoSpeedRecurrence(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        action_dim: int,
        h_blocks: int = 2,
        l_blocks: int = 2,
        n_heads: int = 4,
        h_period: int = 2,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.h_period = h_period

        def block(d: int) -> nn.Module:
            return nn.TransformerEncoderLayer(
                d_model=d, nhead=n_heads, dim_feedforward=4 * d,
                batch_first=True, activation="gelu", norm_first=True,
            )

        self.l_module = nn.ModuleList([block(slot_dim) for _ in range(l_blocks)])
        self.h_module = nn.ModuleList([block(slot_dim) for _ in range(h_blocks)])

        self.action_proj = nn.Linear(action_dim, slot_dim)
        self.h_to_l = nn.Linear(slot_dim, slot_dim)
        self.l_to_h = nn.Linear(slot_dim, slot_dim)
        self.norm_l = nn.LayerNorm(slot_dim)
        self.norm_h = nn.LayerNorm(slot_dim)

    def forward(
        self,
        slots: torch.Tensor,
        action_emb: torch.Tensor,
        n_cycles: int = 8,
    ) -> dict:
        """
        slots: (B, K, slot_dim)
        action_emb: (B, action_dim)
        n_cycles: total recursive cycles (L runs each cycle, H every h_period cycles)
        """
        B, K, D = slots.shape
        a_proj = self.action_proj(action_emb).unsqueeze(1)               # (B, 1, D)

        L_state = slots
        H_state = slots.mean(dim=1, keepdim=True).expand(-1, K, -1).clone()
        trajectory = [L_state]

        for cycle in range(n_cycles):
            inp_l = self.norm_l(L_state + a_proj + self.h_to_l(H_state))
            for blk in self.l_module:
                inp_l = blk(inp_l)
            L_state = inp_l

            if (cycle + 1) % self.h_period == 0:
                inp_h = self.norm_h(H_state + self.l_to_h(L_state))
                for blk in self.h_module:
                    inp_h = blk(inp_h)
                H_state = inp_h

            trajectory.append(L_state)

        return {
            "slots_final": L_state,
            "H_final": H_state,
            "trajectory": trajectory,
        }
