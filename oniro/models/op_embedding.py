"""Op embedding for v40 - math-as-code conditioning.

Each sample carries an op_id from OP_VOCAB. The embedded op token gets
prepended at position 0 of the URM input sequence so the model can condition
its "execution" on the operation kind.

OP_VOCAB (32 ids):
    0  ARC_GENERIC
    1  ARC_RE        (RE-ARC augmented)
    2  ARC_CONCEPT
    3  ARC_MINI
    4  ARC_HEAVY
    5..25 MATH_* (matches math_gen_v2.ALL_GENERATORS order)
    26 SUDOKU
    27 CA_CONWAY
    28 CA_BS
    29 CA_RULE110
    30 DSL_COMPOSE
    31 UNKNOWN_OP
"""

from __future__ import annotations

import torch
from torch import nn


N_OPS = 32

OP_NAMES = [
    "ARC_GENERIC", "ARC_RE", "ARC_CONCEPT", "ARC_MINI", "ARC_HEAVY",
    "MATH_ADD", "MATH_SUB", "MATH_MUL", "MATH_DIV", "MATH_MOD",
    "MATH_DOUBLE", "MATH_HALVE", "MATH_ARITH_SEQ", "MATH_FIB",
    "MATH_PRIME_MARK", "MATH_MAX", "MATH_MIN", "MATH_EQUAL",
    "MATH_SORT", "MATH_GRAVITY", "MATH_MIRROR_H", "MATH_ROTATE",
    "MATH_COUNT", "MATH_HISTOGRAM", "MATH_PARITY", "MATH_ARITH_CHAIN",
    "SUDOKU", "CA_CONWAY", "CA_BS", "CA_RULE110",
    "DSL_COMPOSE", "UNKNOWN_OP",
]
assert len(OP_NAMES) == N_OPS

OP_ID = {name: i for i, name in enumerate(OP_NAMES)}


class OpEmbedding(nn.Module):
    """Maps op_id (B,) int64 → (B, 1, d_model) token."""

    def __init__(self, n_ops: int = N_OPS, d_model: int = 768):
        super().__init__()
        self.n_ops = n_ops
        self.d_model = d_model
        self.embed = nn.Embedding(n_ops, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, op_id: torch.Tensor) -> torch.Tensor:
        """op_id: (B,) long → (B, 1, d_model)."""
        if op_id.dim() == 0:
            op_id = op_id.unsqueeze(0)
        return self.embed(op_id.clamp(0, self.n_ops - 1)).unsqueeze(1)
