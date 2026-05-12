"""Spatial slot-broadcast grid decoder.

Each slot decodes to its own (color_logits, alpha_mask). The slot-wise alphas
softmax over slots per pixel → composite. Lets one slot "own" each grid region.

Replaces the previous mean-pooled decoder which collapsed all slots into one
spatial code (which was why the model could only learn background pixels).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GridDecoder(nn.Module):
    def __init__(
        self,
        slot_dim: int = 96,
        grid_size: int = 32,
        n_colors: int = 10,
        feat_dim: int = 128,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.grid_size = grid_size
        self.n_colors = n_colors
        self.feat_dim = feat_dim

        base = max(4, grid_size // 4)
        self.base = base

        self.slot_to_feat = nn.Linear(slot_dim, feat_dim)
        self.pos = nn.Parameter(torch.randn(1, 1, feat_dim, base, base) * 0.02)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, feat_dim, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(feat_dim, feat_dim, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, 1, 1),
            nn.GELU(),
        )
        self.head_color = nn.Conv2d(feat_dim, n_colors, 1)
        self.head_alpha = nn.Conv2d(feat_dim, 1, 1)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        slots: (B, K, slot_dim)
        returns: (B, n_colors, grid_size, grid_size) composite logits.
        """
        B, K, D = slots.shape
        feat = self.slot_to_feat(slots)                       # (B, K, feat_dim)

        spatial = self.pos + feat[:, :, :, None, None]        # (B, K, feat_dim, b, b)
        spatial = spatial.reshape(B * K, self.feat_dim, self.base, self.base)
        h = self.up(spatial)                                  # (B*K, feat_dim, H, W)
        if h.shape[-1] != self.grid_size:
            h = F.interpolate(h, size=(self.grid_size, self.grid_size), mode="nearest")

        color = self.head_color(h)                            # (B*K, n_colors, H, W)
        alpha = self.head_alpha(h)                            # (B*K, 1, H, W)

        color = color.reshape(B, K, self.n_colors, self.grid_size, self.grid_size)
        alpha = alpha.reshape(B, K, 1, self.grid_size, self.grid_size)

        attn = alpha.softmax(dim=1)                           # softmax over slots
        combined = (color * attn).sum(dim=1)                  # (B, n_colors, H, W)
        return combined

    def decode_grid(self, slots: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(slots)
            return logits.argmax(dim=1)
