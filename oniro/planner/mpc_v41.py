"""MPC planner for ARC-AGI-3 (v41.1).

Dream-rollout planning: from current URM state, simulate up to H steps
ahead using DynamicsPredictor + RewardHead + CuriosityEnsemble, score
each branch, return the FIRST action of the best branch.

Distinct from oniro/planner/mpc.py which is older slot-attention-based.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def mpc_plan(
    state_tokens: torch.Tensor,
    urm_pool_fn,
    dynamics,
    policy,
    reward,
    value,
    curiosity,
    action_encoder,
    horizon: int = 5,
    branching: int = 3,
    curiosity_weight: float = 0.1,
    value_weight: float = 1.0,
    discount: float = 0.95,
) -> dict:
    """Greedy-after-root MPC. Branches at the root only.

    Returns {'best_action_id', 'best_click_xy', 'best_score'} all (B,).
    """
    B = state_tokens.shape[0]
    device = state_tokens.device
    root_pool = urm_pool_fn(state_tokens)

    pol = policy(root_pool)
    disc_logits = pol["discrete_logits"]
    click_mean = pol["click_xy_mean"]

    top_idx = disc_logits.topk(min(branching, disc_logits.shape[-1]), dim=-1).indices

    best_score = torch.full((B,), -float("inf"), device=device)
    best_action = torch.zeros(B, dtype=torch.long, device=device)
    best_click = click_mean.clone()

    for k in range(top_idx.shape[-1]):
        action_id = top_idx[:, k]
        is_click = (action_id == 5)
        click_xy = torch.where(is_click.unsqueeze(-1), click_mean,
                                torch.zeros_like(click_mean))
        action_token = action_encoder(action_id, click_xy)

        r0 = reward(root_pool, action_token)
        c0 = curiosity.curiosity_bonus(root_pool, action_token)
        score = r0 + curiosity_weight * c0
        cur_pool = dynamics(state_tokens, action_token)

        for h in range(1, horizon):
            pol_h = policy(cur_pool)
            a_h = pol_h["discrete_logits"].argmax(dim=-1)
            click_h = torch.where(
                (a_h == 5).unsqueeze(-1),
                pol_h["click_xy_mean"],
                torch.zeros_like(pol_h["click_xy_mean"]),
            )
            act_tok_h = action_encoder(a_h, click_h)
            r_h = reward(cur_pool, act_tok_h)
            c_h = curiosity.curiosity_bonus(cur_pool, act_tok_h)
            score = score + (discount ** h) * (r_h + curiosity_weight * c_h)
            cur_pool = dynamics(cur_pool.unsqueeze(1), act_tok_h)

        score = score + value_weight * (discount ** horizon) * value(cur_pool)

        improved = score > best_score
        best_score = torch.where(improved, score, best_score)
        best_action = torch.where(improved, action_id, best_action)
        best_click = torch.where(improved.unsqueeze(-1), click_xy, best_click)

    return {
        "best_action_id": best_action,
        "best_click_xy": best_click,
        "best_score": best_score,
    }
