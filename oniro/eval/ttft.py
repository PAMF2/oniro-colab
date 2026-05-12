"""Test-time fine-tuning (TTFT) for ARC-AGI tasks.

Paper finding: TTFT lifts LLM solve rate 5% → 39% on ARC.

Procedure per task:
    1. Snapshot model state_dict.
    2. For N inner steps, fine-tune on the task's `train` demo pairs only
       (the test pair's output is held out / unseen).
    3. Predict on the test pair.
    4. Restore model state.

This adds runtime per eval task but is the established way to beat baseline
ARC scores with small models.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
import torch.nn.functional as F


def ttft_finetune_task(
    model,
    demo_pairs: list[dict],
    n_steps: int = 30,
    lr: float = 1e-4,
    loss_fn: Callable[[dict, dict], torch.Tensor] | None = None,
    device: str = "cuda",
) -> dict:
    """Returns snapshot needed to restore model after this task's eval.

    demo_pairs: list of dicts with {image, next_image, action_disc, action_click, grid_out}
    loss_fn: callable (out, batch) -> scalar loss. Default = grid CE on grid_out.
    """
    snapshot = deepcopy(model.state_dict())

    if loss_fn is None:
        from oniro.losses.grid_ce import grid_ce_loss

        def loss_fn(out: dict, batch: dict) -> torch.Tensor:
            g = batch["grid_out"].to(device)
            return grid_ce_loss(out["grid_logits_pred"], g)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for step in range(n_steps):
        for batch in demo_pairs:
            out = model(
                image=batch["image"].to(device),
                next_image=batch["next_image"].to(device),
                action_disc=batch["action_disc"].to(device),
                action_click=batch["action_click"].to(device),
            )
            l = loss_fn(out, batch)
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if "_slots_for_memory_update" in out:
                model.memory.ema_update(out["_slots_for_memory_update"])

    model.eval()
    return snapshot


def restore_snapshot(model, snapshot: dict) -> None:
    model.load_state_dict(snapshot)


def pairs_from_task_json(task: dict, image_size: int, grid_target_side: int) -> list[dict]:
    """Build TTFT batch list from a single ARC task JSON, train pairs only."""
    from oniro.data.arc2_loader import _grid_to_image, _grid_to_int_tensor, task_to_action

    out = []
    a_idx = task_to_action(task.get("_task_id", ""), action_vocab=1024)
    for p in task.get("train", []):
        img = _grid_to_image(p["input"], image_size).unsqueeze(0)
        nxt = _grid_to_image(p["output"], image_size).unsqueeze(0)
        gout = _grid_to_int_tensor(p["output"], grid_target_side).unsqueeze(0)
        out.append({
            "image": img,
            "next_image": nxt,
            "action_disc": torch.tensor([a_idx], dtype=torch.long),
            "action_click": torch.zeros(1, 2),
            "grid_out": gout,
        })
    return out
