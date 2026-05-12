"""EMA (Exponential Moving Average) shadow weights for training stability.

URM paper (arxiv:2512.14693) reports EMA on parameters is essential for
recursive transformer stability. Implementation here mirrors the standard
trick: maintain a shadow copy of params, updated each step as

    ema_w = decay * ema_w + (1 - decay) * w

At evaluation, swap params -> shadow, predict, swap back.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import torch
from torch import nn


class EMA:
    def __init__(self, modules: list[nn.Module], decay: float = 0.999):
        self.decay = decay
        self.modules = modules
        self.shadow: list[dict] = []
        for m in modules:
            self.shadow.append({k: v.detach().clone() for k, v in m.state_dict().items()})

    @torch.no_grad()
    def update(self) -> None:
        for m, sh in zip(self.modules, self.shadow):
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point:
                    sh[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    sh[k].copy_(v.detach())

    @contextmanager
    def swap_in(self):
        """Temporarily swap module weights with EMA shadow. Restore on exit."""
        backups = []
        for m, sh in zip(self.modules, self.shadow):
            backups.append({k: v.detach().clone() for k, v in m.state_dict().items()})
            m.load_state_dict(sh, strict=True)
        try:
            yield
        finally:
            for m, bk in zip(self.modules, backups):
                m.load_state_dict(bk, strict=True)
