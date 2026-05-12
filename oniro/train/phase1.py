"""Phase 1: passive pretrain on Ego4D + HowTo100M.

Active losses: slot-recon (downsampled MSE) + SAE + VLM (where captions exist).
JEPA + curiosity warmup at `warmup_jepa_at` step. Encoder frozen for the first
`freeze_steps`, then LoRA unfreeze.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time

import torch
from torch import nn
import torch.nn.functional as F

from oniro.models.oniro import Oniro, OniroConfig
from oniro.losses import jepa_loss, sae_loss, vlm_ce_loss


@dataclass
class Phase1Config:
    steps: int = 600_000
    warmup_jepa_at: int = 100_000
    encoder_freeze_steps: int = 50_000
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.05
    w_slot_recon: float = 0.5
    w_sae: float = 0.2
    w_jepa: float = 1.0
    w_vlm: float = 0.3
    w_curiosity: float = 0.1
    log_every: int = 50
    grad_clip: float = 1.0


def _slot_recon_loss(slots: torch.Tensor, frame_lowres: torch.Tensor) -> torch.Tensor:
    """Tiny spatial-broadcast decoder regularizer (no learned weights):
    average slots → 8x8 broadcast → linear pull toward downsampled frame mean.
    Stand-in until a proper spatial-broadcast decoder lands.
    """
    mean = slots.mean(dim=1).mean(dim=-1, keepdim=True)
    target = frame_lowres.mean(dim=(2, 3))
    return F.mse_loss(mean.squeeze(-1).expand_as(target), target)


def train_phase1(
    model: Oniro,
    data_iter: Iterable[dict],
    cfg: Phase1Config | None = None,
    device: str = "cuda",
    ckpt_dir: str | Path = "checkpoints/phase1",
) -> dict:
    cfg = cfg or Phase1Config()
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
    )
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    model.encoder.freeze()
    last_log: float = time.time()

    metrics: dict[str, float] = {}
    for step, batch in enumerate(data_iter):
        if step >= cfg.steps:
            break
        if step == cfg.encoder_freeze_steps:
            model.encoder.unfreeze()

        frames = batch["frames"].to(device)
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        B, T, C, H, W = frames.shape
        img = frames[:, 0]
        nxt = frames[:, 1] if T > 1 else None

        text = batch.get("text_tokens")
        if text is not None:
            text = text.to(device)

        out = model(image=img, next_image=nxt, text_tokens=text)

        loss = torch.zeros((), device=device)
        loss_recon = _slot_recon_loss(out["slots"], F.adaptive_avg_pool2d(img, 8))
        loss = loss + cfg.w_slot_recon * loss_recon
        loss_s = sae_loss(out["ema_slots"], out["sae_recon"], out["sae_features"])
        loss = loss + cfg.w_sae * loss_s["total"]

        if step >= cfg.warmup_jepa_at and "pred_next_slots" in out and "target_next_slots" in out:
            lj = jepa_loss(out["pred_next_slots"], out["target_next_slots"], out["slots"])
            loss = loss + cfg.w_jepa * lj["total"]

        if "vlm_logits" in out and "vlm_targets" in out:
            l_v = vlm_ce_loss(out["vlm_logits"], out["vlm_targets"])
            loss = loss + cfg.w_vlm * l_v

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0:
            now = time.time()
            metrics = {"step": step, "loss": float(loss), "sec_per_step": (now - last_log) / max(cfg.log_every, 1)}
            last_log = now
            yield metrics

        if step > 0 and step % 25_000 == 0:
            torch.save({"model": model.state_dict(), "step": step}, Path(ckpt_dir) / f"step_{step}.pt")

    torch.save({"model": model.state_dict(), "step": step}, Path(ckpt_dir) / "last.pt")
    return metrics
