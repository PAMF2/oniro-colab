"""TopK Sparse Autoencoder (arXiv:2406.04093).

Applied to EMA-frozen slot activations (tau=0.99). Extracts ~32 monosemantic features
per slot from a dictionary of 4096. The slot grad does NOT flow through the SAE; SAE
is a read-out for interpretability and for editable handles at the orchestrator level.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(
        self,
        d_in: int = 128,
        dict_size: int = 4096,
        topk: int = 32,
        bias: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.dict_size = dict_size
        self.topk = topk

        self.encoder = nn.Linear(d_in, dict_size, bias=bias)
        self.decoder = nn.Linear(dict_size, d_in, bias=bias)

        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            w = self.decoder.weight
            self.decoder.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))

        self.register_buffer("feature_usage", torch.zeros(dict_size))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def topk_mask(self, z: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = z.topk(self.topk, dim=-1)
        f = torch.zeros_like(z)
        f.scatter_(-1, topk_idx, topk_vals)
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (..., d_in)
        returns: (recon, sparse_features) both shaped (..., d_in) and (..., dict_size)
        """
        z = self.encode(x)
        f = self.topk_mask(z)
        recon = self.decode(f)

        with torch.no_grad():
            self.feature_usage += (f.flatten(0, -2) > 0).float().sum(dim=0)

        return recon, f

    def dead_features(self, threshold: int = 0) -> torch.Tensor:
        return (self.feature_usage <= threshold).nonzero(as_tuple=True)[0]

    def resample_dead(self, x_sample: torch.Tensor, threshold: int = 0) -> int:
        """Resample dead features with random directions from a data sample.

        Returns number of features resampled.
        """
        dead = self.dead_features(threshold)
        if dead.numel() == 0:
            return 0
        with torch.no_grad():
            idx = torch.randint(0, x_sample.shape[0], (dead.numel(),))
            new_dirs = x_sample[idx]
            new_dirs = new_dirs / (new_dirs.norm(dim=-1, keepdim=True) + 1e-8)
            self.encoder.weight[dead] = new_dirs * 0.2
            self.decoder.weight[:, dead] = new_dirs.T
            self.feature_usage[dead] = 0
        return int(dead.numel())

    def reset_usage(self) -> None:
        self.feature_usage.zero_()
