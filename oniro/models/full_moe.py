"""Full FFN Mixture-of-Experts for URM.

Heavier alternative to MoL when capacity is the bottleneck. Each expert
is a full ConvSwiGLU FFN (not a LoRA delta). Top-1 routing per batch sample
via the same (mean_tokens, op_embed) router. Default disabled in URM.

Param cost: K * full_FFN. For d=768, ffn=3584, K=2: ~14M extra.
v40 keeps this OFF by default; switch on if MoL ceiling is hit.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _SingleFFN(nn.Module):
    """One ConvSwiGLU FFN expert (matches the base ConvSwiGLU shape)."""

    def __init__(self, d_model: int, ffn_hidden: int, conv_kernel: int = 3):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w2 = nn.Linear(d_model, ffn_hidden, bias=False)
        self.dwconv = nn.Conv1d(ffn_hidden, ffn_hidden, kernel_size=conv_kernel,
                                padding=conv_kernel // 2, groups=ffn_hidden)
        self.w3 = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        a = self.w1(x_norm)
        b = self.w2(x_norm)
        b_conv = self.dwconv(b.transpose(1, 2)).transpose(1, 2)
        h = a * F.silu(b_conv)
        return self.w3(h)


class FullMoEConvSwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int | None = None,
                 n_experts: int = 2, conv_kernel: int = 3):
        super().__init__()
        h = ffn_hidden or 2 * d_model
        self.d_model = d_model
        self.h = h
        self.n_experts = n_experts
        self.norm = nn.LayerNorm(d_model)
        self.experts = nn.ModuleList([
            _SingleFFN(d_model, h, conv_kernel) for _ in range(n_experts)
        ])
        self.router = nn.Linear(2 * d_model, n_experts, bias=True)
        self.last_expert_usage: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, op_embed: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = self.norm(x)
        mean_tok = x_norm.mean(dim=1)
        if op_embed is None:
            op_vec = torch.zeros_like(mean_tok)
        else:
            op_vec = op_embed.squeeze(1) if op_embed.dim() == 3 else op_embed
        logits = self.router(torch.cat([mean_tok, op_vec], dim=-1))
        top1 = logits.argmax(dim=-1)
        usage = F.one_hot(top1, num_classes=self.n_experts).float().mean(dim=0)
        self.last_expert_usage = usage.detach()

        outs = []
        for b in range(x.shape[0]):
            outs.append(self.experts[int(top1[b].item())](x_norm[b:b + 1]))
        return torch.cat(outs, dim=0)

    def load_balance_loss(self) -> torch.Tensor:
        if self.last_expert_usage is None:
            return torch.zeros(())
        target = 1.0 / self.n_experts
        return ((self.last_expert_usage - target) ** 2).sum()
