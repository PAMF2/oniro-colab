"""Vector Symbolic Algebra bindings (arxiv:2511.08747 Joffe & Eliasmith).

Lightweight HRR (Holographic Reduced Representation) bind/unbind ops as
torch modules. Used as an optional symbolic-style intermediate layer that
can be wired into ONIRO eval / MCTS to mark abstract object roles.

bind(a, b) = circular convolution of a and b   = ifft(fft(a) * fft(b))
unbind(c, a) = circular correlation             = ifft(fft(c) * conj(fft(a)))

These satisfy unbind(bind(a, b), a) ≈ b for orthogonal-like distributed
vectors. Lossy but distributive and bounded-norm.
"""

from __future__ import annotations

import torch
from torch import nn


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular convolution along last dim. a, b: (..., D)."""
    A = torch.fft.fft(a, dim=-1)
    B = torch.fft.fft(b, dim=-1)
    return torch.fft.ifft(A * B, dim=-1).real


def unbind(c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Circular correlation: recover b approximately from bind(a, b)."""
    C = torch.fft.fft(c, dim=-1)
    A = torch.fft.fft(a, dim=-1)
    return torch.fft.ifft(C * A.conj(), dim=-1).real


def bundle(*xs: torch.Tensor) -> torch.Tensor:
    """Superposition (sum). Normalised to unit norm along last dim."""
    s = torch.stack(xs, dim=0).sum(dim=0)
    norm = s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return s / norm


class VSALayer(nn.Module):
    """Learnable VSA layer.

    Maintains a small bank of "role" vectors r_1..r_K. Given an input
    sequence (B, T, D), interprets each position-i as a (role_i, value_i)
    pair via a learned projection, then bundles bind(role_i, value_i) into
    a single (B, D) symbolic summary. Optional read-out projects back to D.
    """

    def __init__(self, d_model: int, n_roles: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_roles = n_roles
        self.roles = nn.Parameter(torch.randn(n_roles, d_model) * (d_model ** -0.5))
        # gate selects which role each position binds with
        self.role_logits = nn.Linear(d_model, n_roles)
        # value projection
        self.value_proj = nn.Linear(d_model, d_model)
        self.readout = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, D) bundled symbolic summary."""
        B, T, D = x.shape
        role_w = torch.softmax(self.role_logits(x), dim=-1)        # (B, T, K)
        # per-position role: weighted sum of role vectors
        role_vec = role_w @ self.roles                              # (B, T, D)
        val_vec = self.value_proj(x)                                # (B, T, D)
        bound = bind(role_vec, val_vec)                             # (B, T, D)
        bundled = bound.mean(dim=1)                                 # (B, D)
        return self.readout(bundled)
