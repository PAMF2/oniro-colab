"""Slot Attention (arXiv:2006.15055).

Iterative cross-attention bottleneck that segments a feature map into K object-centric
slots. ONIRO uses K=6, d=128. Slots feed the SAE and the JEPA dynamics predictor.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_slots: int = 6,
        dim: int = 128,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 256,
        input_dim: int = 768,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(input_dim, dim, bias=False)
        self.to_v = nn.Linear(input_dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor, num_slots: int | None = None) -> torch.Tensor:
        """
        inputs: (B, N, D_in) feature tokens (e.g. SigLIP patches)
        returns: (B, K, dim) slot embeddings
        """
        b, n, _ = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, K, -1)
        sigma = self.slots_log_sigma.exp().expand(b, K, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots_n = self.norm_slots(slots)
            q = self.to_q(slots_n)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps          # competitive over slots
            attn = attn / attn.sum(dim=-1, keepdim=True)   # weighted mean over inputs

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, self.dim),
                slots_prev.reshape(-1, self.dim),
            ).reshape(b, K, self.dim)

            slots = slots + self.mlp(self.norm_pre_mlp(slots))

        return slots
