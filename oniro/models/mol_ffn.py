"""Mixture of LoRAs (MoL) for the URM ConvSwiGLU FFN.

Implementation of arxiv:2512.12880 "Improving Recursive Transformers with
Mixture of LoRAs", adapted for the ConvSwiGLU FFN used in URM.

Design:
  - Shared base ConvSwiGLU is unchanged.
  - K LoRA experts (default 4), each providing low-rank deltas on the W1
    projection of ConvSwiGLU (lightest delta path; W3 untouched to keep
    output dim stable).
  - Router: a small MLP that takes (mean over tokens, op_embed) -> K logits.
    Top-1 routing per BATCH (not per token) to keep the implementation
    simple and the recursive trunk weight-tied across loops.
  - Load-balancing aux signal: usage variance is computed but only exposed;
    the caller decides whether to add it to the training loss.

Param cost (d=768, h=3584, K=4, rank=16):
    LoRA A: 768 * 16 = 12,288   per expert
    LoRA B: 16 * 3584 = 57,344  per expert
    per-expert total: 69,632
    K=4: 278,528 ≈ 280k extra params (~1.4% of URM)
    Router: 768 * K = 3,072  + bias K
    Total MoL overhead: ~282k.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MoLConvSwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int | None = None,
                 conv_kernel: int = 3, n_experts: int = 4, lora_rank: int = 16):
        super().__init__()
        h = ffn_hidden or 2 * d_model
        self.d_model = d_model
        self.h = h
        self.n_experts = n_experts
        self.lora_rank = lora_rank

        # Shared base (same as ConvSwiGLU)
        self.w1 = nn.Linear(d_model, h, bias=False)
        self.w2 = nn.Linear(d_model, h, bias=False)
        self.dwconv = nn.Conv1d(h, h, kernel_size=conv_kernel,
                                padding=conv_kernel // 2, groups=h)
        self.w3 = nn.Linear(h, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # K LoRA experts: low-rank delta on W1
        # delta_W1 = B_k @ A_k,  A_k: (rank, d_model), B_k: (h, rank)
        self.lora_A = nn.Parameter(torch.zeros(n_experts, lora_rank, d_model))
        self.lora_B = nn.Parameter(torch.zeros(n_experts, h, lora_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B initialized to zero so initial delta is zero (LoRA convention)
        # leave lora_B at zeros

        # Router: input is concat(mean_tokens, op_embed) → K logits
        # If no op_embed provided, router reads just mean_tokens (zero pad)
        self.router = nn.Linear(2 * d_model, n_experts, bias=True)

        # Track last router decision for load-balance loss
        self.last_expert_usage: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, op_embed: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, d_model). op_embed: optional (B, 1, d_model) or (B, d_model)."""
        x_norm = self.norm(x)

        # Compute router input
        mean_tok = x_norm.mean(dim=1)   # (B, d_model)
        if op_embed is None:
            op_vec = torch.zeros_like(mean_tok)
        else:
            op_vec = op_embed.squeeze(1) if op_embed.dim() == 3 else op_embed
        router_in = torch.cat([mean_tok, op_vec], dim=-1)
        router_logits = self.router(router_in)               # (B, K)
        top1_idx = router_logits.argmax(dim=-1)              # (B,)
        # store one-hot usage for load-balance aux signal
        usage = F.one_hot(top1_idx, num_classes=self.n_experts).float().mean(dim=0)
        self.last_expert_usage = usage.detach()

        # Base path
        a = self.w1(x_norm)            # (B, T, h)
        b = self.w2(x_norm)
        b_conv = self.dwconv(b.transpose(1, 2)).transpose(1, 2)
        h_base = a * F.silu(b_conv)
        out_base = self.w3(h_base)

        # LoRA delta path: gather selected expert per batch sample
        # delta_a = (x_norm @ A_k.T) @ B_k.T   for the k chosen per sample
        # Implement via batched gather: pick A[top1_idx] (B, rank, d) and B[top1_idx] (B, h, rank)
        A_sel = self.lora_A[top1_idx]                       # (B, rank, d)
        B_sel = self.lora_B[top1_idx]                       # (B, h, rank)
        # delta_a = einsum('btd, brd -> btr', x_norm, A_sel)
        delta_a = torch.einsum('btd,brd->btr', x_norm, A_sel)
        # delta_a_proj = einsum('btr, bhr -> bth', delta_a, B_sel)
        delta_a_full = torch.einsum('btr,bhr->bth', delta_a, B_sel)
        h_delta = delta_a_full * F.silu(b_conv)
        out_delta = self.w3(h_delta)

        return out_base + out_delta

    def load_balance_loss(self) -> torch.Tensor:
        """Variance of expert usage. Caller multiplies and adds to total loss."""
        if self.last_expert_usage is None:
            return torch.zeros(())
        # encourage uniform usage: low variance is good
        target = 1.0 / self.n_experts
        return ((self.last_expert_usage - target) ** 2).sum()
