"""ARC-AGI-3 episode runner.

Drives one game (5 levels) end-to-end:
    - Two-phase budget split (probe / exploit)
    - Online Gödel-gated adaptation every N env steps
    - MPC dream rollout for action selection

Returns a structured per-level score dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from oniro.data.arc3_env import ARC3Env, GameAction, FrameData
from oniro.models.oniro import Oniro
from oniro.planner.mpc import MPCPlanner, MPCConfig
from oniro.orchestrator.godel_gate import GodelGate
from oniro.eval.ood_splits import OODBuffer
from oniro.train.phase3_online import phase3_online_episode, OnlineAdaptConfig


@dataclass
class ARC3RunResult:
    game_id: str
    per_level: dict[int, float] = field(default_factory=dict)
    actions_used: dict[int, int] = field(default_factory=dict)
    final_state: str = ""
    gate_accepts: int = 0
    gate_rejects: int = 0


def _grid_to_image(grid: torch.Tensor, size: int = 256) -> torch.Tensor:
    """64×64 int grid (0..15) → 3×size×size RGB float tensor."""
    palette = torch.tensor(
        [
            [0, 0, 0], [30, 144, 255], [220, 20, 60], [50, 205, 50],
            [255, 255, 0], [128, 128, 128], [255, 105, 180], [255, 165, 0],
            [135, 206, 235], [165, 42, 42], [148, 0, 211], [0, 255, 255],
            [255, 0, 255], [255, 255, 255], [70, 130, 180], [0, 100, 0],
        ],
        dtype=torch.float32,
    ) / 255.0
    flat = palette[grid.long().flatten()]
    img = flat.view(grid.shape[0], grid.shape[1], 3).permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=(size, size), mode="bicubic", align_corners=False)
    return img.clamp(0.0, 1.0)


def run_arc3_episode(
    env: ARC3Env,
    model: Oniro,
    planner: MPCPlanner,
    gate: GodelGate,
    ood_buffer: OODBuffer,
    human_baseline_actions: int = 40,
    probe_fraction: float = 0.30,
    adapt_every: int = 5,
    device: str = "cuda",
    image_size: int | None = None,
) -> ARC3RunResult:
    if image_size is None:
        image_size = getattr(getattr(model, "cfg", None), "image_size", 256)
    result = ARC3RunResult(game_id=env.game_id)
    frame: FrameData = env.reset()
    last_img = _grid_to_image(torch.from_numpy(frame.grid), size=image_size).to(device)
    last_action_disc: torch.Tensor | None = None
    last_action_click: torch.Tensor | None = None

    max_actions = int(5 * human_baseline_actions)
    probe_budget = int(probe_fraction * human_baseline_actions)

    while frame.state == "running" and env.step_count < max_actions:
        slots = model.encode_slots(last_img)

        if env.step_count < probe_budget:
            ad = torch.randint(0, len(frame.available_actions), (1,), device=device)
            ac = torch.rand(1, 2, device=device) * 2 - 1
        else:
            plan = planner.plan(slots)
            ad = plan["action_disc"]
            ac = plan["action_click"]

        chosen = frame.available_actions[int(ad.item()) % len(frame.available_actions)]
        data = None
        if chosen == GameAction.CLICK:
            xy = ((ac[0] + 1) * 0.5 * 63).clamp(0, 63).long()
            data = {"x": int(xy[0].item()), "y": int(xy[1].item())}
        try:
            frame = env.step(chosen, data=data)
        except NotImplementedError:
            break

        nxt_img = _grid_to_image(torch.from_numpy(frame.grid), size=image_size).to(device)
        result.per_level[frame.level] = frame.score
        result.actions_used[frame.level] = env.step_count

        if env.step_count % adapt_every == 0:
            decision = phase3_online_episode(
                model, gate, ood_buffer.sample(),
                {
                    "image": last_img, "next_image": nxt_img,
                    "action_disc": ad, "action_click": ac,
                },
                cfg=OnlineAdaptConfig(),
                device=device,
            )
            if decision.verdict == "ACCEPT":
                result.gate_accepts += 1
            else:
                result.gate_rejects += 1

        last_img = nxt_img
        last_action_disc, last_action_click = ad, ac

    result.final_state = frame.state
    return result
