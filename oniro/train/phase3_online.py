"""Phase 3: online Gödel-gated adaptation during ARC-AGI-3 play.

For every transition observed during an ARC episode:
    1. Snapshot weights θ_0.
    2. N inner gradient steps on the just-observed transition → θ_1.
    3. Compute predictive loss on 10 OOD splits.
    4. ACCEPT only if Gödel gate passes; else rollback.

Pairs with `oniro/eval/arc3_runner.py` which loops env-step → adapt-call.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import torch
import torch.nn.functional as F

from oniro.models.oniro import Oniro
from oniro.losses import jepa_loss
from oniro.orchestrator.godel_gate import GodelGate, GateDecision


@dataclass
class OnlineAdaptConfig:
    inner_steps: int = 3
    inner_lr: float = 1e-5
    snapshot_every_n_steps: int = 5


def _pred_loss(
    model: Oniro,
    image: torch.Tensor,
    next_image: torch.Tensor,
    action_disc: torch.Tensor | None = None,
    action_click: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        out = model(
            image=image, next_image=next_image,
            action_disc=action_disc, action_click=action_click,
        )
    return F.mse_loss(out["pred_next_slots"], out["target_next_slots"])


def phase3_online_episode(
    model: Oniro,
    gate: GodelGate,
    ood_buffer: list[dict],
    transition: dict,
    cfg: OnlineAdaptConfig | None = None,
    device: str = "cuda",
) -> GateDecision:
    """One online-adaptation step. Mutates `model` in-place iff gate accepts.

    transition: {image, next_image, action_disc?, action_click?}
    ood_buffer: list of dicts (≥ gate.n_splits) used to evaluate predictive loss.
    """
    cfg = cfg or OnlineAdaptConfig()
    if len(ood_buffer) < gate.n_splits:
        return GateDecision(
            "REJECT", 0, 0.0, gate.sigma_noise, 0.0,
            f"insufficient OOD splits: {len(ood_buffer)}/{gate.n_splits}",
        )

    baseline = torch.tensor(
        [
            float(_pred_loss(model, **{k: v.to(device) for k, v in s.items()}))
            for s in ood_buffer[: gate.n_splits]
        ]
    )

    snapshot = deepcopy(model.state_dict())

    inner_opt = torch.optim.SGD(model.parameters(), lr=cfg.inner_lr, momentum=0.0)
    img = transition["image"].to(device)
    nxt = transition["next_image"].to(device)
    ad = transition.get("action_disc")
    ac = transition.get("action_click")
    if ad is not None:
        ad = ad.to(device)
    if ac is not None:
        ac = ac.to(device)

    for _ in range(cfg.inner_steps):
        out = model(image=img, next_image=nxt, action_disc=ad, action_click=ac)
        loss = jepa_loss(out["pred_next_slots"], out["target_next_slots"], out["slots"])
        inner_opt.zero_grad(set_to_none=True)
        loss["total"].backward()
        inner_opt.step()

    candidate = torch.tensor(
        [
            float(_pred_loss(model, **{k: v.to(device) for k, v in s.items()}))
            for s in ood_buffer[: gate.n_splits]
        ]
    )

    decision = gate.evaluate(baseline.numpy(), candidate.numpy())

    if decision.verdict != "ACCEPT":
        model.load_state_dict(snapshot)

    return decision
