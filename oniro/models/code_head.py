"""CodeHead — math-as-code DSL program prediction (v40.2).

Auxiliary head that predicts a length-3 DSL primitive sequence given the URM
final state. Trained on synthetic DSL_COMPOSE samples (where the ground-truth
program is known) with CE loss; ARC samples target NULL_PROGRAM (no specific
program) at a low weight so the head can stay undecided.

Output shape: (B, seq_len, n_prims + 1). Class `n_prims` is NULL_PROGRAM.

Param cost (d=768, n_prims=146, seq_len=3):
    inp_proj: 2*768 -> 256  = 393k
    seq head: 256 -> 3 * (146+1) = ~113k
    Total: ~500k. (Smaller hidden width avoids bloating the model.)
"""

from __future__ import annotations

import torch
from torch import nn


class CodeHead(nn.Module):
    def __init__(self, d_model: int = 768, n_prims: int = 146, seq_len: int = 3,
                 hidden: int = 256):
        super().__init__()
        self.d_model = d_model
        self.n_prims = n_prims
        self.seq_len = seq_len
        self.hidden = hidden
        # Reads concat of op_token + mean(cell_tokens), each (B, d_model)
        self.inp = nn.Linear(2 * d_model, hidden)
        self.act = nn.GELU()
        # Output: seq_len * (n_prims + 1) logits, reshape to (B, seq_len, n_prims+1)
        self.out = nn.Linear(hidden, seq_len * (n_prims + 1))

    @property
    def null_class(self) -> int:
        return self.n_prims  # last index reserved for NULL_PROGRAM

    def forward(self, urm_final_state: torch.Tensor,
                op_token_idx: int = 0,
                cell_start_idx: int = 101) -> torch.Tensor:
        """urm_final_state: (B, T, d_model) with op token at index 0 and
        cell tokens at indices cell_start_idx .. cell_start_idx + 900.

        Returns logits (B, seq_len, n_prims + 1).
        """
        op_tok = urm_final_state[:, op_token_idx]                       # (B, d)
        cell_mean = urm_final_state[:, cell_start_idx:].mean(dim=1)      # (B, d)
        x = torch.cat([op_tok, cell_mean], dim=-1)                       # (B, 2d)
        h = self.act(self.inp(x))
        flat = self.out(h)
        return flat.view(-1, self.seq_len, self.n_prims + 1)
