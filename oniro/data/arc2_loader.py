"""ARC-AGI-2 dataloader (real, github.com/arcprize/ARC-AGI-2).

Each task is JSON with 3 demo pairs (`train`) + 1-2 test pairs (`test`).
Each pair is {input: grid, output: grid}; grids are 1x1..30x30 of ints 0-9.

ONIRO consumes pairs as (frame, next_frame, action_disc=task_idx).
`action_disc` is the deterministic SHA1 hash of the task_id mod `action_vocab`,
so the dynamics core learns a per-task transformation rule (1000 "actions").
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F


def task_to_action(task_id: str, action_vocab: int = 1024) -> int:
    """Deterministic stable mapping task_id -> action index."""
    h = hashlib.sha1(task_id.encode()).hexdigest()
    return int(h[:8], 16) % max(action_vocab, 1)


# ARC palette (10 colors). Rough match to arcprize.org viewer.
ARC_PALETTE = torch.tensor(
    [
        [  0,   0,   0],
        [ 30, 144, 255],
        [255,  20,  60],
        [ 60, 200,  60],
        [255, 230,  60],
        [128, 128, 128],
        [240, 100, 200],
        [255, 165,   0],
        [135, 206, 235],
        [165,  42,  42],
    ],
    dtype=torch.float32,
) / 255.0


def _grid_to_image(grid: list[list[int]] | np.ndarray, size: int) -> torch.Tensor:
    """ARC grid (1..30 sq) → (3, size, size) float tensor (0..1)."""
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2:
        arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
    h, w = arr.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.int64)
    canvas[:h, :w] = arr
    canvas = np.clip(canvas, 0, 9)

    flat = ARC_PALETTE[canvas.flatten()]
    img = flat.view(side, side, 3).permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=(size, size), mode="nearest")
    return img.squeeze(0)


def _grid_to_int_tensor(grid, target_side: int) -> torch.Tensor:
    """ARC grid → (target_side, target_side) int64 tensor, nearest-upsampled."""
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2:
        arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
    h, w = arr.shape
    side = max(h, w, 1)
    canvas = np.zeros((side, side), dtype=np.int64)
    canvas[:h, :w] = arr
    canvas = np.clip(canvas, 0, 9)
    t = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(target_side, target_side), mode="nearest")
    return t.squeeze(0).squeeze(0).long()


def _pairs_from_task(task: dict) -> list[tuple]:
    out = []
    for p in task.get("train", []):
        out.append((p["input"], p["output"]))
    for p in task.get("test", []):
        if "output" in p:
            out.append((p["input"], p["output"]))
    return out


def arc2_iter(
    data_root: str | Path,
    split: str = "training",
    image_size: int = 128,
    shuffle: bool = True,
    seed: int = 0,
    augment: bool = True,
    action_vocab: int = 1024,
    once: bool = False,
    grid_target_side: int = 32,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield dicts {image, next_image, action_disc, action_click, task_id} per pair.

    action_disc: deterministic int derived from task_id via SHA1 mod action_vocab,
                 so the dynamics core learns a per-task rule.
    once: if True, iterate the split once and stop (eval mode). If False, loops.
    """
    root = Path(data_root) / split
    if not root.exists():
        raise FileNotFoundError(f"ARC-AGI-2 split not found: {root}")

    rng = random.Random(seed)
    task_files = sorted(root.glob("*.json"))

    while True:
        order = task_files[:]
        if shuffle:
            rng.shuffle(order)

        for tf in order:
            with tf.open() as f:
                task = json.load(f)
            a_idx = task_to_action(tf.stem, action_vocab)
            for inp, out in _pairs_from_task(task):
                img = _grid_to_image(inp, image_size)
                nxt = _grid_to_image(out, image_size)
                grid_in = _grid_to_int_tensor(inp, grid_target_side)
                grid_out = _grid_to_int_tensor(out, grid_target_side)
                if augment:
                    k = rng.randint(0, 3)
                    img = torch.rot90(img, k, dims=(1, 2))
                    nxt = torch.rot90(nxt, k, dims=(1, 2))
                    grid_in = torch.rot90(grid_in, k, dims=(0, 1))
                    grid_out = torch.rot90(grid_out, k, dims=(0, 1))
                    if rng.random() < 0.5:
                        img = torch.flip(img, dims=(2,))
                        nxt = torch.flip(nxt, dims=(2,))
                        grid_in = torch.flip(grid_in, dims=(1,))
                        grid_out = torch.flip(grid_out, dims=(1,))
                yield {
                    "image": img,
                    "next_image": nxt,
                    "grid_in": grid_in,
                    "grid_out": grid_out,
                    "action_disc": torch.tensor(a_idx, dtype=torch.long),
                    "action_click": torch.zeros(2),
                    "task_id": tf.stem,
                }
        if once:
            return


def arc2_batch_iter(
    data_root: str | Path,
    split: str = "training",
    image_size: int = 128,
    batch_size: int = 16,
    shuffle: bool = True,
    seed: int = 0,
    augment: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    buf: list[dict] = []
    for item in arc2_iter(data_root, split, image_size, shuffle, seed, augment):
        buf.append(item)
        if len(buf) >= batch_size:
            yield {
                "image": torch.stack([b["image"] for b in buf]),
                "next_image": torch.stack([b["next_image"] for b in buf]),
                "grid_in": torch.stack([b["grid_in"] for b in buf]),
                "grid_out": torch.stack([b["grid_out"] for b in buf]),
                "action_disc": torch.stack([b["action_disc"] for b in buf]),
                "action_click": torch.stack([b["action_click"] for b in buf]),
                "task_id": [b["task_id"] for b in buf],
            }
            buf = []


def count_pairs(data_root: str | Path, split: str = "training") -> int:
    root = Path(data_root) / split
    n = 0
    for tf in root.glob("*.json"):
        with tf.open() as f:
            t = json.load(f)
        n += len(_pairs_from_task(t))
    return n
