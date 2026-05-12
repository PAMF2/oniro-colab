"""Action-conditioned dynamics core.

Default: Mamba2 (arXiv:2405.21060) via `mamba-ssm`. Falls back to a Transformer of
matched-param-count when mamba-ssm is not installed (CPU/Colab/dev).

Predicts s_{t+1} given (slots_t, action_t). Slots are flattened along the K axis and
fed as a length-K sequence; cross-attention layers at depths [n_blocks//2, n_blocks-1]
inject action embeddings as a single token.
"""

from __future__ import annotations

import torch
from torch import nn


def _action_token(a_emb: torch.Tensor, n: int) -> torch.Tensor:
    """Broadcast action embedding to length-n token sequence."""
    if a_emb.dim() == 2:
        return a_emb.unsqueeze(1).expand(-1, n, -1)
    return a_emb


class _MambaBlockFallback(nn.Module):
    """Tiny Transformer block standing in for Mamba2 when mamba-ssm absent."""

    def __init__(self, d: int, n_heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.ff(self.norm2(x))


class DynamicsCore(nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        n_blocks: int = 24,
        slot_dim: int = 128,
        action_dim: int = 1024,
        cross_attn_at: tuple[int, ...] = (12, 23),
        use_mamba: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.cross_attn_at = set(cross_attn_at)

        self.slot_proj_in = nn.Linear(slot_dim, d_model)
        self.slot_proj_out = nn.Linear(d_model, slot_dim)
        self.action_proj = nn.Linear(action_dim, d_model)

        if use_mamba:
            try:
                from mamba_ssm import Mamba2
                self.blocks = nn.ModuleList(
                    [Mamba2(d_model=d_model) for _ in range(n_blocks)]
                )
                self._using_mamba = True
            except Exception:
                self.blocks = nn.ModuleList(
                    [_MambaBlockFallback(d_model) for _ in range(n_blocks)]
                )
                self._using_mamba = False
        else:
            self.blocks = nn.ModuleList(
                [_MambaBlockFallback(d_model) for _ in range(n_blocks)]
            )
            self._using_mamba = False

        self.cross_attn = nn.ModuleDict(
            {
                str(i): nn.MultiheadAttention(d_model, 8, batch_first=True)
                for i in cross_attn_at
            }
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, slots: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """
        slots:      (B, K, slot_dim)
        action_emb: (B, action_dim)

        returns predicted next slots (B, K, slot_dim).
        """
        x = self.slot_proj_in(slots)
        a = self.action_proj(action_emb)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.cross_attn_at:
                a_tok = _action_token(a, x.shape[1])
                ctx, _ = self.cross_attn[str(i)](x, a_tok, a_tok, need_weights=False)
                x = x + ctx

        x = self.norm_out(x)
        return self.slot_proj_out(x)
