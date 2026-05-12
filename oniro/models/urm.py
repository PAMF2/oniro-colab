"""URM v35 — Universal Reasoning Model with GQA + Flash + Cross-cycle KV cache.

Upgrades over v34:
    1. GQA (Grouped Query Attention) — n_kv_heads << n_heads, reduces KV memory
       by group_factor (4×). Llama-2/3, Mistral, DeepSeek pattern.
    2. Flash attention via torch.nn.functional.scaled_dot_product_attention —
       O(N) memory enables larger grids on T4.
    3. Cross-cycle KV cache — within a weight-tied group, K,V recomputed every
       kv_refresh_every cycles. Saves ~50% attention compute when refresh=2.
    4. URMGrouped — N untied URM blocks, each weight-tied for group_loops
       cycles (n_loops_total = n_groups * group_loops). Bulks params without
       breaking the recursive inductive bias.

Pillars retained:
    - ConvSwiGLU FFN (depthwise short-conv on gate)
    - RIMA reweighter between cycles
    - TBPTL — first K loops forward-only (no grad)
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvSwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int = None, conv_kernel: int = 3):
        super().__init__()
        h = ffn_hidden or 2 * d_model
        self.w1 = nn.Linear(d_model, h, bias=False)
        self.w2 = nn.Linear(d_model, h, bias=False)
        self.dwconv = nn.Conv1d(h, h, kernel_size=conv_kernel,
                                padding=conv_kernel // 2, groups=h)
        self.w3 = nn.Linear(h, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        a = self.w1(x_norm)
        b = self.w2(x_norm)
        b_conv = self.dwconv(b.transpose(1, 2)).transpose(1, 2)
        h = a * F.silu(b_conv)
        return self.w3(h)


class GQAAttention(nn.Module):
    """Grouped Query Attention + Flash kernel + KV cache hook.

    n_kv_heads < n_heads -> each KV head shared across (n_heads / n_kv_heads) Q heads.
    Uses F.scaled_dot_product_attention (Flash when CUDA + fp16/bf16).

    When kv_cache dict is passed with valid=True, reuses cached K,V.
    Else recomputes K,V and stores into the cache.
    Q is always recomputed (cheap relative to K,V projections + attention math).
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.group_size = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None and kv_cache.get("valid", False):
            k = kv_cache["k"]
            v = kv_cache["v"]
        else:
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            if kv_cache is not None:
                kv_cache["k"] = k
                kv_cache["v"] = v
                kv_cache["valid"] = True

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)


class URMBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, n_kv_heads: int | None = None,
                 ffn_hidden: int | None = None):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads=n_heads, n_kv_heads=n_kv_heads)
        self.ffn = ConvSwiGLU(d_model, ffn_hidden=ffn_hidden)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, kv_cache=kv_cache)
        x = x + h
        x = x + self.ffn(x)
        return x


class RIMAReweighter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x_new: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate(x_new))
        return alpha * x_new + (1 - alpha) * x_prev


class URM(nn.Module):
    """Universal Reasoning Model — recursive transformer with GQA + KV cache.

    Modes:
        n_groups=1: classic Universal Transformer — single block weight-tied
            for n_loops cycles.
        n_groups>1: N untied URM blocks, each weight-tied for group_loops
            (= n_loops // n_groups) cycles.

    KV cache reuse:
        Within each group, K,V are recomputed every `kv_refresh_every` cycles
        and reused at intermediate cycles. refresh=1 disables reuse.
        refresh=2 saves ~50% K,V projection + reshape cost.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        n_loops: int = 8,
        n_forward_only: int = 2,
        ffn_hidden: int | None = None,
        use_rima: bool = False,
        n_groups: int = 1,
        kv_refresh_every: int = 1,
    ):
        super().__init__()
        assert n_forward_only < n_loops, "n_forward_only must be < n_loops"
        assert n_loops % n_groups == 0, "n_loops must be divisible by n_groups"
        n_kv_heads = n_kv_heads or n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_loops = n_loops
        self.n_forward_only = n_forward_only
        self.use_rima = use_rima
        self.n_groups = n_groups
        self.group_loops = n_loops // n_groups
        self.kv_refresh_every = max(1, kv_refresh_every)

        self.blocks = nn.ModuleList([
            URMBlock(d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, ffn_hidden=ffn_hidden)
            for _ in range(n_groups)
        ])
        self.reweighter = RIMAReweighter(d_model) if use_rima else None
        self.final_norm = nn.LayerNorm(d_model)
        # legacy alias for tests / loaders that reach for `.block`
        if n_groups == 1:
            self.block = self.blocks[0]

    def _step(self, block: URMBlock, cur: torch.Tensor, kv_cache: dict | None) -> torch.Tensor:
        new = block(cur, kv_cache=kv_cache)
        if self.reweighter is not None:
            new = self.reweighter(new, cur)
        return new

    def forward(self, x: torch.Tensor) -> dict:
        states = [x]
        cur = x
        global_cycle = 0
        for block in self.blocks:
            kv_cache = {"valid": False}
            for c in range(self.group_loops):
                if c % self.kv_refresh_every == 0:
                    kv_cache = {"valid": False}
                # Invalidate cache at no_grad/grad boundary so the first grad
                # cycle recomputes K,V with a grad-tracked path (otherwise
                # cached K,V are detached and the first grad cycle would lose
                # K,V gradient flow until the next scheduled refresh).
                if global_cycle == self.n_forward_only:
                    kv_cache = {"valid": False}
                if global_cycle < self.n_forward_only:
                    with torch.no_grad():
                        cur = self._step(block, cur, kv_cache=kv_cache)
                    if global_cycle + 1 == self.n_forward_only:
                        cur = cur.detach()
                else:
                    cur = self._step(block, cur, kv_cache=kv_cache)
                states.append(cur)
                global_cycle += 1
        final = self.final_norm(cur)
        return {"states_per_loop": states, "final_state": final}
