"""Hierarchical ViT patch encoder with 2D RoPE (v41.1).

ARChitects-style two-stage ViT:
    Stage 1 (inner): cells → 4x4 inner patches via small transformer (D_inner=256, 2 layers).
    Stage 2 (outer): inner patches → 4x4 outer patches via larger transformer (D_outer=D_model, 2-4 layers).

2D Rotary Positional Embeddings on both stages: per-dim rotation splits the
embedding into x-coord and y-coord halves, each rotated by its own theta.

Public:
    Hierarchical2DViTEncoder(grid_size, n_colors, d_model, inner_layers,
                              outer_layers, d_inner=256, patch_size=4)
"""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def _rope_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Standard RoPE inverse freqs (dim/2,)."""
    half = dim // 2
    return 1.0 / (base ** (torch.arange(0, half).float() / half))


def _apply_rope_1d(x: torch.Tensor, positions: torch.Tensor,
                   inv_freqs: torch.Tensor) -> torch.Tensor:
    """Apply 1D RoPE along last dim. x: (..., L, D). positions: (..., L) int.
    inv_freqs: (D/2,)."""
    L = x.shape[-2]
    D = x.shape[-1]
    half = D // 2
    angles = positions.float().unsqueeze(-1) * inv_freqs.to(positions.device)  # (..., L, D/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


def apply_2d_rope(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
                  inv_freqs_row: torch.Tensor,
                  inv_freqs_col: torch.Tensor) -> torch.Tensor:
    """2D RoPE: split feature dim half-x half-y, rotate each by its coord.

    x: (..., L, D). rows, cols: (L,) int.
    """
    D = x.shape[-1]
    half = D // 2
    x_row = x[..., :half]
    x_col = x[..., half:]
    x_row = _apply_rope_1d(x_row, rows, inv_freqs_row)
    x_col = _apply_rope_1d(x_col, cols, inv_freqs_col)
    return torch.cat([x_row, x_col], dim=-1)


class _ViTBlock(nn.Module):
    """Standard pre-norm Transformer block with optional 2D RoPE on Q,K."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 use_2d_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_2d_rope = use_2d_rope
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        h = ffn_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, h, bias=False),
            nn.GELU(),
            nn.Linear(h, d_model, bias=False),
        )
        if use_2d_rope:
            self.register_buffer("inv_freqs_row",
                                  _rope_freqs(self.head_dim // 2),
                                  persistent=False)
            self.register_buffer("inv_freqs_col",
                                  _rope_freqs(self.head_dim // 2),
                                  persistent=False)

    def forward(self, x: torch.Tensor,
                rows: torch.Tensor | None = None,
                cols: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, n_heads, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.use_2d_rope and rows is not None and cols is not None:
            q = apply_2d_rope(q, rows, cols, self.inv_freqs_row, self.inv_freqs_col)
            k = apply_2d_rope(k, rows, cols, self.inv_freqs_row, self.inv_freqs_col)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        x = x + self.o(attn)
        x = x + self.ffn(self.norm2(x))
        return x


class Hierarchical2DViTEncoder(nn.Module):
    """Two-stage ViT with 2D RoPE.

    Input: grid (B, H, W) int in [0, n_colors-1].
    Output: tokens (B, n_outer_patches, d_model).

    Inner stage: each 4x4 cell block → flatten → cell embed → 2 ViT layers
        at d_inner=256 → mean-pool to 1 inner-patch vector.
    Outer stage: arrange inner patches as 2D grid → ViT n layers at d_model.
    """

    def __init__(self, grid_size: int = 30, n_colors: int = 10,
                 d_model: int = 896, d_inner: int = 256,
                 patch_size: int = 4,
                 inner_layers: int = 2, outer_layers: int = 4,
                 inner_heads: int = 4, outer_heads: int = 8):
        super().__init__()
        # Pad grid to multiple of patch_size if needed (we already pad input
        # to grid_size in build_micro, so assume grid_size % patch_size == 0
        # OR pad here).
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.n_colors = n_colors
        self.d_model = d_model
        self.d_inner = d_inner

        # Pad to multiple of patch_size
        rem = grid_size % patch_size
        self.pad = (patch_size - rem) if rem else 0
        self.padded_size = grid_size + self.pad
        self.n_inner_side = self.padded_size // patch_size  # e.g. 30->32/4=8
        self.n_outer_patches = self.n_inner_side ** 2

        # Inner: per-cell embedding
        self.cell_embed = nn.Embedding(n_colors + 1, d_inner)   # +1 pad colour
        self.inner_blocks = nn.ModuleList([
            _ViTBlock(d_inner, inner_heads, ffn_mult=4, use_2d_rope=True)
            for _ in range(inner_layers)
        ])
        self.inner_norm = nn.LayerNorm(d_inner)

        # Projection inner → outer dim
        self.inner_to_outer = nn.Linear(d_inner, d_model, bias=False)
        # Outer ViT
        self.outer_blocks = nn.ModuleList([
            _ViTBlock(d_model, outer_heads, ffn_mult=4, use_2d_rope=True)
            for _ in range(outer_layers)
        ])
        self.outer_norm = nn.LayerNorm(d_model)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: (B, H, W) int in [0, n_colors-1] (or pad colour n_colors).
        Returns: (B, n_outer_patches, d_model).
        """
        B = grid.shape[0]
        # Pad to padded_size
        if self.pad > 0:
            grid = F.pad(grid, (0, self.pad, 0, self.pad), value=self.n_colors)
        H = W = self.padded_size
        P = self.patch_size

        # Reshape into patches: (B, n_inner_side, P, n_inner_side, P)
        g = grid.view(B, self.n_inner_side, P, self.n_inner_side, P)
        # Permute to (B, n_inner_side*n_inner_side, P*P)
        g = g.permute(0, 1, 3, 2, 4).contiguous()
        g = g.view(B * self.n_outer_patches, P * P)

        # Inner ViT: each patch of P*P cells -> P*P tokens of dim d_inner
        toks = self.cell_embed(g.clamp(0, self.n_colors))           # (B*n_outer, P*P, d_inner)
        # 2D positions within the inner patch: 0..P-1 for rows and cols
        rows_in = torch.arange(P, device=grid.device).repeat_interleave(P)  # (P*P,)
        cols_in = torch.arange(P, device=grid.device).repeat(P)
        for blk in self.inner_blocks:
            toks = blk(toks, rows=rows_in, cols=cols_in)
        toks = self.inner_norm(toks)
        # Mean-pool inner patch -> single (d_inner,) vector
        patch_vec = toks.mean(dim=1)                                  # (B*n_outer, d_inner)
        # Reshape to (B, n_outer_patches, d_inner)
        patch_vec = patch_vec.view(B, self.n_outer_patches, self.d_inner)
        # Project to outer dim
        outer_tokens = self.inner_to_outer(patch_vec)                  # (B, n_outer, d_model)

        # Outer 2D positions
        rows_out = torch.arange(self.n_inner_side, device=grid.device).repeat_interleave(self.n_inner_side)
        cols_out = torch.arange(self.n_inner_side, device=grid.device).repeat(self.n_inner_side)
        for blk in self.outer_blocks:
            outer_tokens = blk(outer_tokens, rows=rows_out, cols=cols_out)
        outer_tokens = self.outer_norm(outer_tokens)
        return outer_tokens
