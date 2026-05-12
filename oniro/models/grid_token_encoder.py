"""Grid token encoder: cell-as-token embedding for ARC-style grids.

Replaces slot attention. Each grid cell becomes one token with:
    color_embed[cell_value] + row_pos + col_pos.

Supports concatenation of test grid + demo pairs into a single sequence.
"""

from __future__ import annotations

import torch
from torch import nn


class GridTokenEncoder(nn.Module):
    def __init__(
        self,
        grid_size: int = 32,
        n_colors: int = 10,
        d_model: int = 256,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_colors = n_colors
        self.d_model = d_model

        self.color_embed = nn.Embedding(n_colors + 1, d_model)  # +1 for padding
        self.row_pos = nn.Parameter(torch.randn(grid_size, d_model) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(grid_size, d_model) * 0.02)
        # Role tokens to mark demo input/output/test_input regions
        self.role_embed = nn.Embedding(8, d_model)              # 0..7 for roles

    def encode_grid(self, grid: torch.Tensor, role_id: int = 0) -> torch.Tensor:
        """grid: (B, H, W) int64.  Returns (B, H*W, d_model)."""
        B, H, W = grid.shape
        c = self.color_embed(grid.clamp(0, self.n_colors))
        rows = self.row_pos[:H].unsqueeze(0).unsqueeze(2)        # (1, H, 1, d)
        cols = self.col_pos[:W].unsqueeze(0).unsqueeze(1)        # (1, 1, W, d)
        role = self.role_embed(torch.tensor(role_id, device=grid.device))
        tokens = c + rows + cols + role
        return tokens.reshape(B, H * W, self.d_model)

    def forward(
        self,
        test_grid: torch.Tensor,
        demos: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict:
        """
        test_grid: (B, H, W)
        demos: optional list of (demo_in, demo_out) pairs, each (B, H, W)
        Returns dict with 'tokens' (B, T, d_model) and slicing offsets.
        """
        chunks = []
        offsets = {}

        if demos:
            for i, (di, do) in enumerate(demos[:3]):
                offsets[f"demo_{i}_in"] = sum(c.shape[1] for c in chunks)
                chunks.append(self.encode_grid(di, role_id=2 * i + 1))
                offsets[f"demo_{i}_out"] = sum(c.shape[1] for c in chunks)
                chunks.append(self.encode_grid(do, role_id=2 * i + 2))

        offsets["test_in"] = sum(c.shape[1] for c in chunks)
        chunks.append(self.encode_grid(test_grid, role_id=0))

        tokens = torch.cat(chunks, dim=1)
        return {"tokens": tokens, "offsets": offsets, "test_len": chunks[-1].shape[1]}


class GridTokenDecoder(nn.Module):
    """Project token states back to grid color logits."""

    def __init__(self, d_model: int = 256, n_colors: int = 10):
        super().__init__()
        self.head = nn.Linear(d_model, n_colors)

    def forward(self, tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
        """
        tokens: (B, T, d_model) — assume last grid_size*grid_size tokens are test region
        Returns (B, n_colors, grid_size, grid_size) logits.
        """
        B, T, D = tokens.shape
        test_tokens = tokens[:, -grid_size * grid_size:]
        logits = self.head(test_tokens)                       # (B, H*W, n_colors)
        return logits.transpose(1, 2).reshape(B, -1, grid_size, grid_size)
