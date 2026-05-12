"""Dream rollout in latent space (Ha/Schmidhuber 2018).

Given current slots s_0 and a sequence of candidate actions, roll the dynamics core
forward H steps, applying temperature noise τ to each step's prediction. Returns the
full latent trajectory plus per-step curiosity scores.
"""

from __future__ import annotations

import torch

from oniro.models.oniro import Oniro


@torch.no_grad()
def dream_rollout(
    model: Oniro,
    slots_init: torch.Tensor,
    action_seq_disc: torch.Tensor | None = None,
    action_seq_click: torch.Tensor | None = None,
    horizon: int = 8,
    tau: float = 0.7,
) -> dict[str, torch.Tensor]:
    """
    slots_init:        (B, K, slot_dim)
    action_seq_disc:   (B, H) discrete action ids or None
    action_seq_click:  (B, H, 2) click coords or None

    returns:
        slots_traj:   (B, H+1, K, slot_dim)
        curiosity:    (B, H)
    """
    device = slots_init.device
    B, K, D = slots_init.shape

    if action_seq_disc is None and action_seq_click is None:
        raise ValueError("must pass at least one action sequence")

    H = horizon
    if action_seq_disc is not None:
        H = action_seq_disc.shape[1]
    elif action_seq_click is not None:
        H = action_seq_click.shape[1]

    slots = slots_init
    traj = [slots]
    cur = torch.zeros(B, H, device=device)

    for t in range(H):
        ad = action_seq_disc[:, t] if action_seq_disc is not None else None
        ac = action_seq_click[:, t] if action_seq_click is not None else None
        a = model.embed_action(ad, ac)

        nxt = model.dynamics(slots, a)
        if tau > 0:
            nxt = nxt + tau * torch.randn_like(nxt) * nxt.std()
        traj.append(nxt)

        preds = model.curiosity.predict_all(slots, a)
        cur[:, t] = preds.var(dim=0).mean(dim=(-1, -2))

        slots = nxt

    return {"slots_traj": torch.stack(traj, dim=1), "curiosity": cur}
