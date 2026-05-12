"""VLM grounding head.

Small transformer decoder cross-attended over slot tokens. Two roles:
    1. Aux training loss: next-token CE on captions / instructions (forces slots to
       remain semantically aligned with language).
    2. Eval-time goal scoring: log-prob that current slots correspond to a textual
       goal description ("a red square in the corner"). Used by MPC for extrinsic
       reward at ARC-AGI-3 inference.
"""

from __future__ import annotations

import torch
from torch import nn


class VLMHead(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        slot_dim: int = 128,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.slot_proj = nn.Linear(slot_dim, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        slots: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        slots:  (B, K, slot_dim)
        tokens: (B, T) input ids

        returns logits (B, T, vocab_size).
        """
        memory = self.slot_proj(slots)
        x = self.tok_embed(tokens) + self.pos_embed[:, : tokens.shape[1]]
        T = tokens.shape[1]
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tokens.device), diagonal=1
        )
        h = self.decoder(x, memory, tgt_mask=causal)
        return self.lm_head(self.norm(h))

    def score(
        self,
        slots: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Mean log-prob of `tokens` given `slots`. Used as goal-likelihood at inference."""
        logits = self(slots, tokens[:, :-1])
        targets = tokens[:, 1:]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return logp.mean(dim=-1)
