"""Byte-level autoregressive text head for ONIRO.

Cross-attends over URM final state and emits next-token logits over 256-byte
vocab. No external tokenizer required - works for digits, math symbols, ASCII
letters, punctuation directly.

Designed to share the URM hidden width (d_model) for free cross-attention.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


BYTE_VOCAB = 256
BOS = 1     # start-of-text byte
EOS = 4     # end-of-text byte
PAD = 0


def text_to_bytes(s: str, max_len: int = 64) -> torch.Tensor:
    b = s.encode("utf-8")[:max_len - 2]
    ids = [BOS] + list(b) + [EOS]
    if len(ids) < max_len:
        ids = ids + [PAD] * (max_len - len(ids))
    return torch.tensor(ids[:max_len], dtype=torch.long)


def bytes_to_text(ids: torch.Tensor) -> str:
    out = []
    for i in ids.tolist():
        if i == EOS:
            break
        if i in (BOS, PAD):
            continue
        if 0 <= i < 256:
            out.append(bytes([i]))
    try:
        return b"".join(out).decode("utf-8", errors="replace")
    except Exception:
        return "<decode_err>"


class TextHead(nn.Module):
    """Small byte-level decoder with cross-attn to URM state.

    Sized to add ~1-2M params on top of a 20M URM trunk.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        max_len: int = 64,
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok_embed = nn.Embedding(BYTE_VOCAB, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, BYTE_VOCAB, bias=False)

    def forward(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """memory: (B, T_mem, d_model)  tokens: (B, T) -> logits (B, T, V)."""
        T = tokens.shape[1]
        x = self.tok_embed(tokens) + self.pos_embed[:, :T]
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tokens.device), diagonal=1
        )
        h = self.decoder(x, memory, tgt_mask=causal)
        return self.lm_head(self.norm(h))

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, max_new: int = 32,
                 temperature: float = 0.0) -> torch.Tensor:
        """Greedy / temperature sample from BOS."""
        B = memory.shape[0]
        device = memory.device
        seq = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        for _ in range(max_new):
            logits = self.forward(memory, seq[:, -self.max_len:])
            next_logits = logits[:, -1] / max(temperature, 1e-6)
            if temperature <= 0:
                nxt = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            seq = torch.cat([seq, nxt], dim=1)
            if (nxt == EOS).all():
                break
        return seq
