"""Math-side patch encoder for v40 dual-encoder split.

For non-ARC samples (math, sudoku, CA, compose), the visual 3x3 PatchEncoder
is replaced with this row-wise feature encoder. Each row of the grid is
summarised by a small fixed feature vector that captures arithmetic-style
structure: count of non-zero cells, dominant color, parity, mean value,
max value. These features are normalised and projected to d_model. The
output token count matches PatchEncoder (100 tokens) so the URM sequence
length is unchanged at 1001 = [op, patches, cells].

Why this works for math:
  - count_colored, histogram, parity_row, sort_rows all need per-row
    "how many non-zero cells" - directly encoded
  - gravity needs per-column non-zero count but works via row reading too
  - arithmetic-style tasks need dominant color signal

Param cost: small. Linear(5 -> d_model) + pos embeds.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MathPatchEncoder(nn.Module):
    def __init__(self, grid_size: int = 30, n_colors: int = 10,
                 d_model: int = 768, n_out_tokens: int = 100,
                 n_features: int = 5):
        super().__init__()
        self.grid_size = grid_size
        self.n_colors = n_colors
        self.d_model = d_model
        self.n_out_tokens = n_out_tokens
        self.n_features = n_features
        # Linear projects 5 row features -> d_model
        self.proj = nn.Linear(n_features, d_model)
        # Positional embedding for each output token
        self.pos = nn.Parameter(torch.randn(1, n_out_tokens, d_model) * 0.02)
        # Role token to signal "this is a math patch" (distinct from vision)
        self.role = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: (B, H, W) int. Returns (B, n_out_tokens, d_model)."""
        B, H, W = grid.shape
        clamped = grid.clamp(0, self.n_colors - 1).float()
        nonzero_mask = (clamped > 0).float()

        # Per-row features (B, H, n_features):
        f_count = nonzero_mask.sum(dim=2) / float(W)              # fraction non-zero
        f_mean = clamped.mean(dim=2) / float(self.n_colors - 1)   # normalized mean
        f_max = clamped.max(dim=2).values / float(self.n_colors - 1)
        f_parity = (nonzero_mask.sum(dim=2) % 2)                  # parity of count
        # dominant color in row via simple argmax over color histogram
        # one-hot (B, H, W, C) -> sum over W -> (B, H, C) -> argmax -> (B, H)
        oh = F.one_hot(grid.clamp(0, self.n_colors - 1), num_classes=self.n_colors).float()
        f_dominant = oh.sum(dim=2).argmax(dim=2).float() / float(self.n_colors - 1)

        row_features = torch.stack(
            [f_count, f_mean, f_max, f_parity, f_dominant], dim=2
        )  # (B, H, 5)

        # Project rows to tokens. We have H row features. If n_out_tokens > H,
        # repeat / pad; if n_out_tokens < H, truncate. Standard config:
        # grid_size=30, n_out_tokens=100 — repeat each row ~3 times with offset.
        proj = self.proj(row_features)  # (B, H, d_model)
        if H >= self.n_out_tokens:
            tokens = proj[:, :self.n_out_tokens]
        else:
            repeats = (self.n_out_tokens + H - 1) // H
            tokens = proj.repeat(1, repeats, 1)[:, :self.n_out_tokens]

        tokens = tokens + self.pos + self.role
        return tokens
