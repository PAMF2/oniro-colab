"""SigLIP-2 distillation.

Distill a ViT-Large teacher into a ViT-Base student via feature-matching MSE on
patch embeddings. If `transformers` / network is unavailable, falls back to a
no-op stub that warns and exits cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import torch
import torch.nn.functional as F


@dataclass
class DistillConfig:
    teacher: str = "google/siglip2-large-patch16-256"
    student: str = "google/siglip2-base-patch16-256"
    steps: int = 50_000
    lr: float = 1e-4
    batch_size: int = 64
    image_size: int = 256
    out: str = "checkpoints/siglip_b.pt"


def distill(cfg: DistillConfig | None = None, device: str = "cuda") -> dict | None:
    cfg = cfg or DistillConfig()
    try:
        from transformers import AutoModel
        teacher = AutoModel.from_pretrained(cfg.teacher).vision_model.to(device).eval()
        student = AutoModel.from_pretrained(cfg.student).vision_model.to(device).train()
    except Exception as e:
        warnings.warn(f"transformers/network unavailable; distillation skipped ({e})")
        return None

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr)
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)

    for step in range(cfg.steps):
        x = torch.rand(cfg.batch_size, 3, cfg.image_size, cfg.image_size, device=device)
        with torch.no_grad():
            t = teacher(pixel_values=x).last_hidden_state
        s = student(pixel_values=x).last_hidden_state
        s_proj = F.interpolate(
            s.permute(0, 2, 1).unsqueeze(-1), size=(t.shape[1], 1), mode="bilinear"
        ).squeeze(-1).permute(0, 2, 1) if s.shape[1] != t.shape[1] else s

        loss = F.mse_loss(s_proj, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"distill step {step}  mse={float(loss):.4f}")

    torch.save({"student": student.state_dict(), "cfg": cfg.__dict__}, cfg.out)
    return {"final_mse": float(loss), "out": cfg.out}


if __name__ == "__main__":
    distill()
