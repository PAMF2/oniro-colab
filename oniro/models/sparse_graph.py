"""Sparse Causal Slot Graph.

Pillar: **sparse attention** + **world model**.

Learnable adjacency A in R^{K x K} between slots. Slot-to-slot attention is
gated by sigmoid(A) AND TopK truncated per source slot. L1 penalty on sigmoid(A)
pushes most edges to 0, leaving a sparse causal interaction graph.

Extends standard cross-slot attention with explicit structured sparsity, the
SPARTAN-style discovery of which slots actually influence each other.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SparseCausalSlotGraph(nn.Module):
    def __init__(
        self,
        K: int,
        slot_dim: int,
        topk: int = 3,
        n_heads: int = 4,
        l1_lambda: float = 0.01,
    ):
        super().__init__()
        assert slot_dim % n_heads == 0
        self.K = K
        self.D = slot_dim
        self.H = n_heads
        self.Dh = slot_dim // n_heads
        self.topk = max(1, min(topk, K))
        self.l1_lambda = l1_lambda
        self.scale = self.Dh ** -0.5

        self.adj_logits = nn.Parameter(torch.zeros(K, K))

        self.q_proj = nn.Linear(slot_dim, slot_dim)
        self.k_proj = nn.Linear(slot_dim, slot_dim)
        self.v_proj = nn.Linear(slot_dim, slot_dim)
        self.o_proj = nn.Linear(slot_dim, slot_dim)
        self.norm = nn.LayerNorm(slot_dim)

    def adjacency(self) -> torch.Tensor:
        """σ(adj_logits): K×K matrix in (0, 1)."""
        return torch.sigmoid(self.adj_logits)

    def l1_penalty(self) -> torch.Tensor:
        return self.l1_lambda * self.adjacency().sum()

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.H, self.Dh)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        slots: (B, K, D)
        returns: (B, K, D) -- residual update routed through sparse adjacency.
        """
        B, K, D = slots.shape
        assert K == self.K, f"expected K={self.K} slots, got {K}"

        x = self.norm(slots)
        q = self._heads(self.q_proj(x))                          # (B, K, H, Dh)
        k_h = self._heads(self.k_proj(x))                        # (B, K, H, Dh)
        v_h = self._heads(self.v_proj(x))                        # (B, K, H, Dh)

        scores = torch.einsum("bihd,bjhd->bhij", q, k_h) * self.scale     # (B, H, K_q, K_k)
        adj = self.adjacency().unsqueeze(0).unsqueeze(0)                  # (1, 1, K, K)
        gated = scores + torch.log(adj + 1e-9)                            # additive log-gate

        # TopK per source slot
        topk_vals, _ = gated.topk(self.topk, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        mask = gated < threshold
        masked = gated.masked_fill(mask, float("-inf"))
        attn = masked.softmax(dim=-1)

        out = torch.einsum("bhij,bjhd->bihd", attn, v_h)                  # (B, K, H, Dh)
        out = out.reshape(B, K, D)
        return self.o_proj(out)
