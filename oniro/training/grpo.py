"""GRPO — Group Relative Policy Optimization (DeepSeek-R1 style).

Sample G outputs per prompt. Advantage = (reward - mean) / std within the group.
Loss = -clip(ratio * advantage) - β · KL(policy, ref_policy).

Better than REINFORCE for sparse rewards (ARC exact match) — group baseline
reduces variance without learning separate value network.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
import torch.nn.functional as F


def grpo_step(
    encoder, urm, decoder, opt,
    g_in: torch.Tensor, g_out: torch.Tensor,
    ref_encoder, ref_urm, ref_decoder,
    n_group: int = 4,
    eps_clip: float = 0.2,
    kl_beta: float = 0.04,
    temperature: float = 1.0,
    reward_type: str = "cell",
) -> dict:
    """One GRPO update.

    g_in:  (B, H, W) int64
    g_out: (B, H, W) int64 target grid

    ref_*: frozen reference policy snapshot (for KL).
    """
    B, H, W = g_in.shape
    device = g_in.device

    encoder.train(); urm.train(); decoder.train()
    ref_encoder.eval(); ref_urm.eval(); ref_decoder.eval()

    # Current policy forward (with grad)
    enc_out = encoder(g_in)
    urm_out = urm(enc_out["tokens"])
    logits = decoder(urm_out["final_state"], H)
    log_probs = F.log_softmax(logits / max(temperature, 1e-3), dim=1)
    probs = log_probs.exp()

    # Reference policy forward (no grad)
    with torch.no_grad():
        r_enc = ref_encoder(g_in)
        r_urm = ref_urm(r_enc["tokens"])
        r_logits = ref_decoder(r_urm["final_state"], H)
        ref_log_probs = F.log_softmax(r_logits / max(temperature, 1e-3), dim=1)

    rewards_all = []
    logp_sum_all = []
    ref_logp_sum_all = []
    samples_all = []

    for _ in range(n_group):
        flat = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
        sampled = torch.multinomial(flat, 1).reshape(B, H, W)
        samples_all.append(sampled)

        lp = log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
        rlp = ref_log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
        logp_sum_all.append(lp.sum(dim=(1, 2)))
        ref_logp_sum_all.append(rlp.sum(dim=(1, 2)))

        match = (sampled == g_out).float()
        if reward_type == "exact":
            r = (match.view(B, -1).mean(dim=-1) >= 0.999).float()
        else:
            r = match.view(B, -1).mean(dim=-1)
        rewards_all.append(r)

    rewards = torch.stack(rewards_all, dim=0)         # (G, B)
    logp = torch.stack(logp_sum_all, dim=0)           # (G, B)
    ref_logp = torch.stack(ref_logp_sum_all, dim=0)   # (G, B)

    # Group-relative advantage
    mean_r = rewards.mean(dim=0, keepdim=True)
    std_r = rewards.std(dim=0, keepdim=True) + 1e-6
    advantage = ((rewards - mean_r) / std_r).detach()

    # Policy ratio
    ratio = (logp - logp.detach()).exp()              # always 1 here; trivially valid
    # PPO-style clip (degenerate here since ratio=1 always — kept for shape)
    pg_loss = -(advantage * logp).mean()

    # KL penalty (forward KL: policy || ref)
    kl_term = (logp - ref_logp).mean()
    loss = pg_loss + kl_beta * kl_term

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters())
                                   + list(urm.parameters())
                                   + list(decoder.parameters()), 1.0)
    opt.step()

    return {
        "loss": float(loss),
        "pg_loss": float(pg_loss),
        "kl": float(kl_term),
        "mean_reward": float(rewards.mean()),
        "max_reward": float(rewards.max()),
        "advantage_std": float(advantage.std()),
    }


def snapshot_policy(encoder, urm, decoder):
    """Deepcopy current policy as a reference policy. Returns tuple of frozen modules."""
    re = deepcopy(encoder); ru = deepcopy(urm); rd = deepcopy(decoder)
    for m in (re, ru, rd):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
    return re, ru, rd
