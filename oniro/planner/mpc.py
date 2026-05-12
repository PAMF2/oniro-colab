"""Model-Predictive Control over dream rollouts.

Score = extrinsic_reward(final slots, goal text via VLM head)
      + curiosity_weight * sum(curiosity over rollout)
      - action_count_penalty * H

Branching: B candidates sampled per timestep. Pick the candidate whose first action
maximizes the score; execute only that first action, then re-plan next step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from oniro.models.oniro import Oniro
from oniro.planner.dream_rollout import dream_rollout


@dataclass
class MPCConfig:
    horizon: int = 8
    branching: int = 12
    temperature: float = 0.7
    curiosity_weight: float = 0.1
    action_count_penalty: float = 0.05
    n_discrete_actions: int = 5
    click_grid: int = 8


class MPCPlanner:
    def __init__(self, model: Oniro, cfg: MPCConfig | None = None):
        self.model = model
        self.cfg = cfg or MPCConfig()

    @torch.no_grad()
    def sample_actions(
        self, batch: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H = self.cfg.horizon
        B = self.cfg.branching * batch
        disc = torch.randint(0, self.cfg.n_discrete_actions, (B, H), device=device)
        click = torch.rand(B, H, 2, device=device) * 2.0 - 1.0
        return disc, click

    @torch.no_grad()
    def plan(
        self,
        slots_init: torch.Tensor,
        goal_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        slots_init: (B, K, slot_dim)  current slots
        goal_tokens: (B, T) optional textual goal; if None, only curiosity drives.

        returns dict with first chosen action: {"action_disc": (B,), "action_click": (B,2)}
        """
        B = slots_init.shape[0]
        device = slots_init.device
        cfg = self.cfg

        slots_expanded = slots_init.repeat_interleave(cfg.branching, dim=0)
        disc, click = self.sample_actions(B, device)

        roll = dream_rollout(
            self.model, slots_expanded,
            action_seq_disc=disc, action_seq_click=click,
            horizon=cfg.horizon, tau=cfg.temperature,
        )
        final = roll["slots_traj"][:, -1]
        cur_sum = roll["curiosity"].sum(dim=-1)

        if goal_tokens is not None:
            goal_rep = goal_tokens.repeat_interleave(cfg.branching, dim=0)
            extrinsic = self.model.vlm.score(final, goal_rep)
        else:
            extrinsic = torch.zeros(B * cfg.branching, device=device)

        score = (
            extrinsic
            + cfg.curiosity_weight * cur_sum
            - cfg.action_count_penalty * cfg.horizon
        )
        score = score.view(B, cfg.branching)
        best = score.argmax(dim=-1)

        idx = (torch.arange(B, device=device) * cfg.branching) + best
        return {
            "action_disc": disc[idx, 0],
            "action_click": click[idx, 0],
            "score": score.gather(1, best.unsqueeze(1)).squeeze(1),
        }
