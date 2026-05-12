"""Test-time training (TTFT) for URM. Akyurek 2024 — paper shows 5% → 53% lift.

Per evaluation task:
    1. Snapshot encoder/urm/decoder weights.
    2. Fine-tune on the task's 3 demo pairs for N steps with low LR.
    3. Predict test pair.
    4. Restore snapshot for next task.

Combined with AIRV (8 dihedral) and best-of-N sampling.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
import torch.nn.functional as F


def ttft_finetune_urm(
    encoder, urm, decoder,
    demo_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    grid_size: int,
    n_steps: int = 30,
    lr: float = 1e-4,
    device: str = "cuda",
) -> dict:
    """Snapshot + fine-tune model on demos. Caller must restore snapshot after."""
    snap = {
        "encoder": deepcopy(encoder.state_dict()),
        "urm": deepcopy(urm.state_dict()),
        "decoder": deepcopy(decoder.state_dict()),
    }
    params = list(encoder.parameters()) + list(urm.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    encoder.train(); urm.train(); decoder.train()
    for step in range(n_steps):
        for di, do in demo_pairs:
            di = di.to(device)
            do = do.to(device)
            enc_out = encoder(di)
            urm_out = urm(enc_out["tokens"])
            logits = decoder(urm_out["final_state"], grid_size)
            loss = F.cross_entropy(logits, do)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    encoder.eval(); urm.eval(); decoder.eval()
    return snap


def restore_urm(encoder, urm, decoder, snap: dict) -> None:
    encoder.load_state_dict(snap["encoder"])
    urm.load_state_dict(snap["urm"])
    decoder.load_state_dict(snap["decoder"])


@torch.no_grad()
def airv_predict(
    encoder, urm, decoder,
    image_grid: torch.Tensor,
    grid_size: int,
    n_colors: int = 10,
) -> torch.Tensor:
    """AIRV: 8 dihedral augmentations, majority vote per pixel.

    image_grid: (1, H, W) int64
    Returns: (H, W) int64 final majority-voted grid.
    """
    DIHEDRAL = [(0, False), (1, False), (2, False), (3, False),
                (0, True), (1, True), (2, True), (3, True)]
    votes = torch.zeros(n_colors, grid_size, grid_size, device=image_grid.device)
    for k, flip in DIHEDRAL:
        aug = torch.rot90(image_grid, k=k, dims=(-2, -1))
        if flip:
            aug = torch.flip(aug, dims=(-1,))
        enc_out = encoder(aug)
        urm_out = urm(enc_out["tokens"])
        logits = decoder(urm_out["final_state"], grid_size)
        pred = logits.argmax(dim=1)[0]
        # reverse augmentation
        if flip:
            pred = torch.flip(pred, dims=(-1,))
        pred = torch.rot90(pred, k=-k, dims=(-2, -1))
        for c in range(n_colors):
            votes[c] += (pred == c).float()
    return votes.argmax(dim=0)


@torch.no_grad()
def best_of_n_predict(
    encoder, urm, decoder,
    image_grid: torch.Tensor,
    grid_size: int,
    n_samples: int = 8,
    temperature: float = 0.8,
) -> torch.Tensor:
    """Sample N grids, pick by self-consistency (max likelihood under own dist)."""
    enc_out = encoder(image_grid)
    urm_out = urm(enc_out["tokens"])
    logits = decoder(urm_out["final_state"], grid_size)
    log_probs = F.log_softmax(logits / max(temperature, 1e-3), dim=1)
    probs = log_probs.exp()
    flat = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
    best_score = -float("inf")
    best_pred = None
    for _ in range(n_samples):
        sampled = torch.multinomial(flat, 1).reshape(1, grid_size, grid_size)
        score = log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1).sum().item()
        if score > best_score:
            best_score = score
            best_pred = sampled[0]
    # Always include greedy argmax in the candidate pool
    greedy = logits.argmax(dim=1)[0]
    greedy_score = log_probs.gather(1, greedy.unsqueeze(0).unsqueeze(0)).squeeze().sum().item()
    if greedy_score > best_score:
        best_pred = greedy
    return best_pred
