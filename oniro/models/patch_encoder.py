"""Patch encoder - ViT-style visual tokens for grids.

Pairs with GridTokenEncoder. Cell encoder gives per-cell numeric tokens
(fine-grained), patch encoder gives per-3x3-block visual tokens (macro).
Concat as URM input -> "visão + número" dual signal.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PatchEncoder(nn.Module):
    """Extract patch tokens from a grid.

    For grid_size=30 and patch_size=3, produces 10x10=100 patch tokens.
    Each patch is embedded by a Linear over the flattened patch values cast
    to one-hot over n_colors.
    """

    def __init__(self, grid_size: int = 30, n_colors: int = 10,
                 patch_size: int = 3, d_model: int = 768):
        super().__init__()
        assert grid_size % patch_size == 0, \
            f"grid_size {grid_size} not divisible by patch_size {patch_size}"
        self.grid_size = grid_size
        self.n_colors = n_colors
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches_side = grid_size // patch_size
        n_patches = self.n_patches_side ** 2
        # one-hot encode each cell, flatten patch -> Linear
        flat_dim = (patch_size ** 2) * n_colors
        self.proj = nn.Linear(flat_dim, d_model)
        self.pos = nn.Parameter(
            torch.randn(1, n_patches, d_model) * 0.02
        )
        self.role = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: (B, H, W) int. Returns (B, n_patches, d_model)."""
        B, H, W = grid.shape
        P = self.patch_size
        # one-hot: (B, H, W, n_colors)
        oh = F.one_hot(grid.clamp(0, self.n_colors - 1), num_classes=self.n_colors).float()
        # split into patches: (B, n_h, P, n_w, P, n_colors) -> (B, n_h, n_w, P, P, n_colors)
        nh = H // P
        nw = W // P
        oh = oh.view(B, nh, P, nw, P, self.n_colors).permute(0, 1, 3, 2, 4, 5).contiguous()
        # flatten patch dims
        oh = oh.view(B, nh * nw, P * P * self.n_colors)
        tokens = self.proj(oh) + self.pos + self.role
        return tokens
