"""Sparse relational memory.

A small learnable memory bank that the dynamics core attends to when predicting
the next slot state. Two roles:

    1. **Past -> Future relational chain.** Slot prototypes accumulate via EMA
       across observed transitions, so each forward pass can attend back to
       transitions seen earlier in the training run.

    2. **Sparse routing.** Each query (current slot) attends to only TopK memory
       slots — keeps the lookup interpretable and matches ONIRO's sparse-aligned
       representation pillar.

The module is residual: `pred_next = dynamics(s, a) + alpha * memory(s)`. With
alpha=0 it is a no-op; useful for ablations.
"""

from __future__ import annotations

import torch
from torch import nn


class SparseMemoryAttention(nn.Module):
    def __init__(
        self,
        slot_dim: int = 128,
        memory_size: int = 64,
        topk: int = 4,
        ema: float = 0.99,
        n_heads: int = 4,
    ):
        super().__init__()
        assert slot_dim % n_heads == 0
        self.D = slot_dim
        self.M = memory_size
        self.topk = max(1, min(topk, memory_size))
        self.ema = ema
        self.H = n_heads
        self.scale = (slot_dim // n_heads) ** -0.5

        self.register_buffer("memory", torch.randn(memory_size, slot_dim) * 0.02)
        self.register_buffer("usage", torch.zeros(memory_size))

        self.q_proj = nn.Linear(slot_dim, slot_dim)
        self.k_proj = nn.Linear(slot_dim, slot_dim)
        self.v_proj = nn.Linear(slot_dim, slot_dim)
        self.o_proj = nn.Linear(slot_dim, slot_dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., D) -> (..., H, D/H)
        return x.reshape(*x.shape[:-1], self.H, self.D // self.H)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        slots: (B, K, D)
        returns: (B, K, D)  -- TopK sparse memory-attended residual.
        """
        B, K, D = slots.shape
        mem = self.memory.detach()
        q = self._heads(self.q_proj(slots))                # (B, K, H, Dh)
        k_h = self._heads(self.k_proj(mem))                # (M, H, Dh)
        v_h = self._heads(self.v_proj(mem))                # (M, H, Dh)

        scores = torch.einsum("bkhd,mhd->bkhm", q, k_h) * self.scale

        # Sparse: keep only TopK along memory axis. Mask others to -inf so the
        # softmax zeros them out, then renormalise over the surviving entries.
        topk_vals, _ = scores.topk(self.topk, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        mask = scores < threshold
        masked = scores.masked_fill(mask, float("-inf"))
        attn = masked.softmax(dim=-1)

        out = torch.einsum("bkhm,mhd->bkhd", attn, v_h)
        out = out.reshape(B, K, D)
        return self.o_proj(out)

    @torch.no_grad()
    def ema_update(
        self,
        slots: torch.Tensor,
        held_out_score_fn: "callable | None" = None,
    ) -> int:
        """Update memory bank toward the most-similar observed slot via EMA.

        Gödel-gated extension: if `held_out_score_fn` provided, score the bank
        before AND after the candidate update and revert if score did not strictly
        improve. score_fn() returns a scalar tensor — higher = better.
        """
        flat = slots.flatten(0, -2).detach()
        if flat.numel() == 0:
            return 0

        snapshot = self.memory.clone() if held_out_score_fn is not None else None
        usage_snapshot = self.usage.clone() if held_out_score_fn is not None else None

        scores = flat @ self.memory.T
        nearest = scores.argmax(dim=-1)
        updated = 0
        for i in range(self.M):
            mask = nearest == i
            if not mask.any():
                continue
            mean = flat[mask].mean(dim=0)
            self.memory[i] = self.ema * self.memory[i] + (1 - self.ema) * mean
            self.usage[i] += float(mask.sum())
            updated += 1

        if held_out_score_fn is not None:
            v_new = held_out_score_fn()
            new_score = float(v_new.item()) if hasattr(v_new, "item") else float(v_new)
            new_bank = self.memory.clone()
            new_usage = self.usage.clone()
            self.memory.copy_(snapshot)
            self.usage.copy_(usage_snapshot)
            v_base = held_out_score_fn()
            base_score = float(v_base.item()) if hasattr(v_base, "item") else float(v_base)
            if new_score > base_score:
                self.memory.copy_(new_bank)
                self.usage.copy_(new_usage)
            else:
                updated = 0
        return updated

    def reset_usage(self) -> None:
        self.usage.zero_()
