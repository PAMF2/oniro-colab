"""Identity-Bias Grid Decoder.

Decodes (slot_state, input_grid) → output grid where:
    final[i,j] = trust_gate[i,j] * input[i,j] + (1-trust_gate[i,j]) * argmax(logits[i,j])

trust_gate is a sigmoid per-pixel "should we keep input?" head. ARC tasks
typically change ~20% pixels — identity bias gives ~80% pixels for free,
forcing the model to learn ONLY the delta.

Game-changer for exact-match: most ARC tasks fail because 1-2 background pixels
get predicted wrong. Identity gate fixes that automatically.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class IdentityBiasDecoder(nn.Module):
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
        self.head_trust = nn.Conv2d(feat_dim, 1, 1)

    def forward(self, slots: torch.Tensor, input_grid: torch.Tensor | None = None) -> torch.Tensor:
        """Default returns logits only (drop-in replacement for GridDecoder)."""
        return self.full(slots, input_grid)["logits"]

    def full(self, slots: torch.Tensor, input_grid: torch.Tensor | None = None) -> dict:
        """
        slots: (B, K, slot_dim)
        input_grid: (B, H, W) int64 with values 0..n_colors-1 (optional)
        Returns dict with:
            logits:   (B, n_colors, H, W) raw color logits
            trust:    (B, H, W) sigmoid in [0,1]
            pred:     (B, H, W) int64 final grid (identity-mixed if input_grid given)
        """
        B, K, D = slots.shape
        feat = self.slot_to_feat(slots)
        spatial = self.pos + feat[:, :, :, None, None]
        spatial = spatial.reshape(B * K, self.feat_dim, self.base, self.base)
        h = self.up(spatial)
        if h.shape[-1] != self.grid_size:
            h = F.interpolate(h, size=(self.grid_size, self.grid_size), mode="nearest")

        color = self.head_color(h)                            # (B*K, n_colors, H, W)
        alpha = self.head_alpha(h)                            # (B*K, 1, H, W)
        trust = self.head_trust(h)                            # (B*K, 1, H, W)

        color = color.reshape(B, K, self.n_colors, self.grid_size, self.grid_size)
        alpha = alpha.reshape(B, K, 1, self.grid_size, self.grid_size)
        trust = trust.reshape(B, K, 1, self.grid_size, self.grid_size)

        attn = alpha.softmax(dim=1)
        logits = (color * attn).sum(dim=1)                    # (B, n_colors, H, W)
        trust_pix = (trust * attn).sum(dim=1).squeeze(1)      # (B, H, W)
        trust_pix = trust_pix.sigmoid()

        argmax_pred = logits.argmax(dim=1)                    # (B, H, W)
        if input_grid is not None and input_grid.shape == argmax_pred.shape:
            # Stochastic-style routing: where trust > 0.5, keep input pixel
            keep = trust_pix > 0.5
            final = torch.where(keep, input_grid, argmax_pred)
        else:
            final = argmax_pred

        return {
            "logits": logits,
            "trust": trust_pix,
            "pred": final,
            "argmax_pred": argmax_pred,
        }
