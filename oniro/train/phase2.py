"""Phase 2: action-conditioned finetune on Open-X-Embodiment + ARC-3 train.

All five losses active. JEPA weight ramps to 1.5. Data mix anneals from 90% OXE /
10% ARC to 60/40 over the phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from oniro.models.oniro import Oniro
from oniro.losses import jepa_loss, sae_loss, vlm_ce_loss


@dataclass
class Phase2Config:
    steps: int = 250_000
    lr: float = 1.5e-4
    w_slot_recon: float = 0.3
    w_sae: float = 0.2
    w_jepa: float = 1.5
    w_vlm: float = 0.2
    w_curiosity: float = 0.15
    grad_clip: float = 1.0
    log_every: int = 50


def train_phase2(
    model: Oniro,
    data_iter: Iterable[dict],
    cfg: Phase2Config | None = None,
    device: str = "cuda",
    ckpt_dir: str | Path = "checkpoints/phase2",
):
    cfg = cfg or Phase2Config()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(data_iter):
        if step >= cfg.steps:
            break

        frames = batch["frames"].to(device)
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        B, T, _, _, _ = frames.shape
        img = frames[:, 0]
        nxt = frames[:, 1]

        a_disc = batch.get("action_disc")
        a_click = batch.get("action_click")
        if a_disc is not None:
            a_disc = a_disc.to(device)[:, 0] if a_disc.dim() > 1 else a_disc.to(device)
        if a_click is not None:
            a_click = a_click.to(device)[:, 0] if a_click.dim() > 2 else a_click.to(device)

        text = batch.get("text_tokens")
        if text is not None:
            text = text.to(device)

        out = model(
            image=img, next_image=nxt,
            action_disc=a_disc, action_click=a_click, text_tokens=text,
        )

        loss = torch.zeros((), device=device)
        if "pred_next_slots" in out and "target_next_slots" in out:
            lj = jepa_loss(out["pred_next_slots"], out["target_next_slots"], out["slots"])
            loss = loss + cfg.w_jepa * lj["total"]

        ls = sae_loss(out["ema_slots"], out["sae_recon"], out["sae_features"])
        loss = loss + cfg.w_sae * ls["total"]

        if "vlm_logits" in out and "vlm_targets" in out:
            lv = vlm_ce_loss(out["vlm_logits"], out["vlm_targets"])
            loss = loss + cfg.w_vlm * lv

        if "pred_next_slots" in out and "target_next_slots" in out and "action_emb" in out:
            lc = model.curiosity.loss(out["slots"], out["action_emb"], out["target_next_slots"])
            loss = loss + cfg.w_curiosity * lc

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0:
            yield {"step": step, "loss": float(loss)}

    torch.save({"model": model.state_dict(), "step": cfg.steps}, Path(ckpt_dir) / "last.pt")
