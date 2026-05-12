"""TTAT-TRM: test-time adapt of the recursive trunk (arxiv:2511.02886, McGovern).

For each evaluation task, snapshot model weights, fine-tune trunk +
per-augmentation task embeddings on the task's demo pairs for N steps,
predict the test grid, then restore the snapshot. Paper finding: LoRA
or embedding-only adaptation is significantly inferior to full trunk
fine-tuning under tight compute budgets.

Public:
    ttat_finetune(modules: list, optimizer_factory, demos, loss_fn,
                  n_steps, lr) -> snapshot_dict
    restore(modules, snapshot)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch


def ttat_snapshot(modules: list[torch.nn.Module]) -> list[dict]:
    """Return a deep-copy of each module's state_dict."""
    return [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in modules]


def ttat_restore(modules: list[torch.nn.Module], snapshot: list[dict]) -> None:
    for m, snap in zip(modules, snapshot):
        m.load_state_dict(snap, strict=True)


def ttat_finetune(
    modules: list[torch.nn.Module],
    demos: list[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    n_steps: int = 30,
    lr: float = 1e-4,
    device: str = "cuda",
) -> list[dict]:
    """Snapshot, fine-tune on demos, return snapshot. Caller MUST restore.

    Args:
        modules: list of nn.Module to adapt (trunk + embeddings).
        demos: [(di, do), ...] each (H, W) int tensors.
        loss_fn(pred_logits, target_grid) -> scalar.
        forward_fn(di) -> pred_logits (B, C, H, W) for one demo.
        n_steps: gradient steps per task.
        lr: AdamW LR for adaptation.
        device: 'cuda' or 'cpu'.
    """
    snap = ttat_snapshot(modules)
    params = [p for m in modules for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=lr)
    for m in modules:
        m.train()
    for _ in range(n_steps):
        for di, do in demos:
            di = di.to(device)
            do = do.to(device)
            logits = forward_fn(di)
            loss = loss_fn(logits, do)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    for m in modules:
        m.eval()
    return snap
