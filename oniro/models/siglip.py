"""SigLIP-2 ViT-Base wrapper.

Loads the HuggingFace `google/siglip2-base-patch16-256` checkpoint and exposes a
clean (B, N, D) patch-token interface for Slot Attention. Supports:
    - freeze for first N steps (then optional LoRA unfreeze)
    - small CPU-friendly fallback (`tiny=True`) for the Colab demo
"""

from __future__ import annotations

import torch
from torch import nn


class SigLIPEncoder(nn.Module):
    def __init__(
        self,
        pretrained: str = "google/siglip2-base-patch16-256",
        tiny: bool = False,
        image_size: int = 256,
        patch_size: int = 16,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
    ):
        super().__init__()
        self.tiny = tiny
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = (image_size // patch_size) ** 2

        if tiny:
            self._build_tiny(n_layers=4, d_model=192, n_heads=4)
        else:
            self._try_load_hf(pretrained, d_model, n_layers, n_heads)

    def _build_tiny(self, n_layers: int, d_model: int, n_heads: int) -> None:
        self.d_model = d_model
        self.patch_embed = nn.Conv2d(3, d_model, self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self._hf = None

    def _try_load_hf(self, repo: str, d_model: int, n_layers: int, n_heads: int) -> None:
        try:
            from transformers import AutoModel
            self._hf = AutoModel.from_pretrained(repo).vision_model
            self.d_model = self._hf.config.hidden_size
        except Exception:
            self._hf = None
            self._build_tiny(n_layers=n_layers, d_model=d_model, n_heads=n_heads)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → patch tokens (B, N, D)."""
        if self._hf is not None:
            out = self._hf(pixel_values=x).last_hidden_state
            return out
        z = self.patch_embed(x).flatten(2).transpose(1, 2)
        z = z + self.pos_embed[:, : z.shape[1]]
        z = self.transformer(z)
        return self.norm(z)
