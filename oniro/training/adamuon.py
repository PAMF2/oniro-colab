"""AdaMuon optimizer (arxiv:2507.11005).

Muon + per-parameter Adam-style 2nd-moment + RMS-aligned rescaling.
~30-40% step-efficiency over AdamW at small/medium scale, drop-in
replacement.

Muon orthogonalises the update via Newton-Schulz iteration on the
momentum buffer. AdaMuon adds element-wise 2nd-moment scaling so
parameters with high gradient variance get smaller updates.

Use on 2D weight matrices (Linear/Conv weights). For embeddings, lm_head,
biases, LayerNorm: fall back to AdamW.

Reference:
    Muon original: kellerjordan/Muon
    AdaMuon: arxiv:2507.11005 (Liu et al.)
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def _newton_schulz_5(g: torch.Tensor, steps: int = 5,
                     eps: float = 1e-7) -> torch.Tensor:
    """5th-order Newton-Schulz iteration to orthogonalise a 2D matrix.

    Stable in bf16/fp16 (no SVD). Returns U such that U has approximately
    orthogonal columns, ||g - U||_F minimised under the constraint.
    """
    assert g.dim() == 2, "Newton-Schulz expects a 2D tensor"
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.float()
    norm = x.norm() + eps
    x = x / norm
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class AdaMuon(Optimizer):
    """AdaMuon: Muon + Adam 2nd-moment rescaling.

    Args:
        params: iterable of 2D Linear/Conv weight parameters only.
        lr: learning rate (default 1e-3 for typical small models)
        momentum: SGD-style momentum coefficient (default 0.95)
        beta2: 2nd-moment EMA coefficient (default 0.95)
        eps: numerical stability (default 1e-8)
        weight_decay: decoupled weight decay (default 0.0)
        ns_steps: Newton-Schulz iterations (default 5)
        rms_align: divide update by RMS to align magnitude with Adam (default True)
    """

    def __init__(self, params, lr: float = 1e-3,
                 momentum: float = 0.95, beta2: float = 0.95,
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 ns_steps: int = 5, rms_align: bool = True):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, ns_steps=ns_steps,
                        rms_align=rms_align)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            b2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            rms_align = group["rms_align"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.dim() != 2:
                    # AdaMuon is for 2D weights only. Caller should keep
                    # biases / embeddings on AdamW. We fall back to AdamW
                    # here as a safety net.
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    m = state["exp_avg"]
                    v = state["exp_avg_sq"]
                    m.mul_(mom).add_(g, alpha=1 - mom)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1 = 1 - mom ** state["step"]
                    bc2 = 1 - b2 ** state["step"]
                    denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
                    p.addcdiv_(m / bc1, denom, value=-lr)
                    if wd > 0:
                        p.add_(p, alpha=-lr * wd)
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["step"] = 0
                state["step"] += 1
                buf = state["momentum_buffer"]
                v = state["v"]

                # SGD-momentum on raw grad
                buf.mul_(mom).add_(g, alpha=1 - mom)
                # 2nd-moment EMA (Adam style)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                # Bias-correct momentum buffer (Adam style) before NS
                bc1 = 1 - mom ** state["step"]
                buf_corrected = buf / bc1

                # Newton-Schulz orthogonalisation on bias-corrected momentum
                update = _newton_schulz_5(buf_corrected, steps=ns_steps)

                # RMS-aligned rescale with bias correction
                if rms_align:
                    bc2 = 1 - b2 ** state["step"]
                    v_corrected = v / bc2
                    # Use scalar RMS (not per-element) for orthogonal update;
                    # per-element would destroy the orthogonalisation.
                    rms = v_corrected.sqrt().mean().add_(eps)
                    update = update / rms

                # Decoupled weight decay
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)

                p.add_(update, alpha=-lr)
        return loss
