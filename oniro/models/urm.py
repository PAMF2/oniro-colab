"""URM — Universal Reasoning Model.

Reimplementation of the URM architecture (arxiv:2512.14693, Dec 2025).
Reports SOTA on ARC-AGI-1 (53.8%) and ARC-AGI-2 (16%) for neural baselines.

Key components:
    1. Token-level grid encoding (cell-as-token, drops slot attention).
    2. ConvSwiGLU FFN — depthwise short conv inside SwiGLU FFN for local
       contextual interactions.
    3. Universal Transformer style: single block applied N_loops times
       (weight-tied recursion).
    4. TBPTL — Truncated Backprop Through Loops: first K loops are no_grad,
       gradient flows only through later loops.
    5. ACT halting (optional) via entropy gate.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvSwiGLU(nn.Module):
    """SwiGLU FFN augmented with a depthwise short-conv on the gate path.

    Standard SwiGLU: h = (W1 x) * silu(W2 x); out = W3 h
    ConvSwiGLU: same but apply 1D depthwise conv (kernel 3) on the gate before
    silu, injecting local context between adjacent tokens.
    """

    def __init__(self, d_model: int, ffn_hidden: int = None, conv_kernel: int = 3):
        super().__init__()
        h = ffn_hidden or 2 * d_model
        self.w1 = nn.Linear(d_model, h, bias=False)
        self.w2 = nn.Linear(d_model, h, bias=False)
        self.dwconv = nn.Conv1d(h, h, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=h)
        self.w3 = nn.Linear(h, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)."""
        x_norm = self.norm(x)
        a = self.w1(x_norm)
        b = self.w2(x_norm)
        # depthwise short conv on the gate path (b transposed for conv1d)
        b_conv = self.dwconv(b.transpose(1, 2)).transpose(1, 2)
        h = a * F.silu(b_conv)
        return self.w3(h)


class URMBlock(nn.Module):
    """One UT block: attention + ConvSwiGLU, applied repeatedly (weight-tied)."""

    def __init__(self, d_model: int, n_heads: int = 8, ffn_hidden: int = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = ConvSwiGLU(d_model, ffn_hidden=ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ffn(x)
        return x


class RIMAReweighter(nn.Module):
    """RIMA-style reweighter (arxiv:2603.05234).

    Learned gate α = σ(LinLayer(x_new)) interpolates between previous and new state:
        x = α * x_new + (1 - α) * x_prev

    Stabilises recursive trajectories; RIMA reports +5.3pp ARC-2 over plain TRM.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x_new: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate(x_new))
        return alpha * x_new + (1 - alpha) * x_prev


class URM(nn.Module):
    """Universal Reasoning Model — recursive transformer for grid reasoning.

    Args:
        d_model: hidden dim (e.g. 256)
        n_heads: attention heads
        n_loops: total recursive applications of the shared block
        n_forward_only: first K loops run under torch.no_grad (TBPTL)
        ffn_hidden: ConvSwiGLU hidden dim
        use_rima: enable RIMA reweighter between cycles (default False)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_loops: int = 8,
        n_forward_only: int = 2,
        ffn_hidden: int = None,
        use_rima: bool = False,
    ):
        super().__init__()
        assert n_forward_only < n_loops, "n_forward_only must be < n_loops"
        self.d_model = d_model
        self.n_loops = n_loops
        self.n_forward_only = n_forward_only
        self.use_rima = use_rima

        self.block = URMBlock(d_model, n_heads=n_heads, ffn_hidden=ffn_hidden)
        self.reweighter = RIMAReweighter(d_model) if use_rima else None
        self.final_norm = nn.LayerNorm(d_model)

    def _step(self, cur: torch.Tensor) -> torch.Tensor:
        new = self.block(cur)
        if self.reweighter is not None:
            new = self.reweighter(new, cur)
        return new

    def forward(self, x: torch.Tensor) -> dict:
        states = [x]
        cur = x
        with torch.no_grad():
            for _ in range(self.n_forward_only):
                cur = self._step(cur)
                states.append(cur)
        cur = cur.detach()
        for _ in range(self.n_loops - self.n_forward_only):
            cur = self._step(cur)
            states.append(cur)
        final = self.final_norm(cur)
        return {"states_per_loop": states, "final_state": final}
