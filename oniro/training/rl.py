"""REINFORCE-style RL fine-tuning for ONIRO URM.

Sample N predictions per input via dropout-based stochasticity, score each by
cell_match (or pair_exact), use REINFORCE with leave-one-out baseline.

Reward = mean(pred == gt) for cell-level OR (pred == gt).all() for exact.
Loss = -(reward - baseline) * log_prob_of_sampled_grid

Auto-evolving: per-step AlphaEvolve mutation accepted if reward improves on
held-out, archive accumulates accepted deltas.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def reinforce_step(
    encoder, urm, decoder, opt,
    g_in: torch.Tensor, g_out: torch.Tensor,
    n_samples: int = 4,
    temperature: float = 1.0,
    reward_type: str = "cell",     # "cell" or "exact"
    entropy_bonus: float = 0.01,
) -> dict:
    """
    Single REINFORCE update.

    g_in:  (B, H, W) int64
    g_out: (B, H, W) int64

    Returns dict with loss, mean_reward, baseline.
    """
    B, H, W = g_in.shape
    device = g_in.device

    encoder.train(); urm.train(); decoder.train()
    enc_out = encoder(g_in)
    urm_out = urm(enc_out["tokens"])
    logits = decoder(urm_out["final_state"], H)             # (B, C, H, W)

    # Sample N grids per item; compute log_prob of each
    log_probs = F.log_softmax(logits / max(temperature, 1e-3), dim=1)   # (B, C, H, W)
    probs = log_probs.exp()
    rewards_all = []
    logp_sum_all = []

    for _ in range(n_samples):
        # Categorical sample per pixel
        flat = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
        sampled = torch.multinomial(flat, 1).reshape(B, H, W)
        # Log prob of sample
        lp = log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)      # (B, H, W)
        logp_sum = lp.sum(dim=(1, 2))                                  # (B,)
        # Reward
        match = (sampled == g_out).float()
        if reward_type == "exact":
            reward = (match.view(B, -1).mean(dim=-1) >= 0.999).float()
        else:
            reward = match.view(B, -1).mean(dim=-1)
        rewards_all.append(reward)
        logp_sum_all.append(logp_sum)

    rewards = torch.stack(rewards_all, dim=0)                # (N, B)
    logp = torch.stack(logp_sum_all, dim=0)                  # (N, B)
    baseline = rewards.mean(dim=0, keepdim=True)             # (1, B) leave-one-out aprox
    advantage = (rewards - baseline).detach()

    # REINFORCE loss + entropy bonus
    loss_rl = -(advantage * logp).mean()
    entropy = -(probs * log_probs).sum(dim=1).mean()
    loss = loss_rl - entropy_bonus * entropy

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters())
                                   + list(urm.parameters())
                                   + list(decoder.parameters()), 1.0)
    opt.step()
    return {
        "loss": float(loss),
        "loss_rl": float(loss_rl),
        "mean_reward": float(rewards.mean()),
        "max_reward": float(rewards.max()),
        "entropy": float(entropy),
    }
