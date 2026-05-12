"""Multi-source ARC corpus: ARC-AGI-1 + ARC-AGI-2 + (optional) RE-ARC.

All three share the same per-task JSON schema (train/test pairs with int grids
0-9). We yield interleaved pairs from all available roots, hashed action ids
namespaced per source so the dynamics core can learn source-aware rules without
collisions.

Why include ARC-1? Same format, ~400 additional tasks = 30%+ more training pairs.
Why include RE-ARC? Procedurally regenerates ARC-1 tasks with thousands of
variations — boosts coverage of each rule.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch

from oniro.data.arc2_loader import (
    _grid_to_image,
    _grid_to_int_tensor,
    _pairs_from_task,
    task_to_action,
)


def multi_arc_iter(
    roots: list[tuple[str, str | Path]],
    image_size: int = 128,
    grid_target_side: int = 32,
    shuffle: bool = True,
    seed: int = 0,
    augment: bool = True,
    action_vocab: int = 1024,
    once: bool = False,
    split: str = "training",
) -> Iterator[dict[str, torch.Tensor]]:
    """Interleaved iterator across multiple ARC-format dataset roots.

    roots: list of (source_tag, path_to_data_dir_or_split_dir).
           If the path ends in ``training`` / ``evaluation`` it is used as-is;
           otherwise we look inside for that split subfolder.
    """
    rng = random.Random(seed)
    rosters: list[tuple[str, list[Path]]] = []
    for tag, root in roots:
        rp = Path(root)
        if rp.name in ("training", "evaluation"):
            split_dir = rp
        else:
            split_dir = rp / split
        if not split_dir.exists():
            continue
        files = sorted(split_dir.glob("*.json"))
        if files:
            rosters.append((tag, files))

    if not rosters:
        raise FileNotFoundError("no ARC roots resolved to files")

    while True:
        if shuffle:
            for _, files in rosters:
                rng.shuffle(files)

        idx = [0] * len(rosters)
        while True:
            advanced = False
            for r_idx, (tag, files) in enumerate(rosters):
                i = idx[r_idx]
                if i >= len(files):
                    continue
                advanced = True
                tf = files[i]
                idx[r_idx] += 1
                with tf.open() as f:
                    task = json.load(f)
                action_key = f"{tag}::{tf.stem}"
                a_idx = task_to_action(action_key, action_vocab)
                for inp, out in _pairs_from_task(task):
                    img = _grid_to_image(inp, image_size)
                    nxt = _grid_to_image(out, image_size)
                    g_in = _grid_to_int_tensor(inp, grid_target_side)
                    g_out = _grid_to_int_tensor(out, grid_target_side)
                    if augment:
                        k = rng.randint(0, 3)
                        img = torch.rot90(img, k, dims=(1, 2))
                        nxt = torch.rot90(nxt, k, dims=(1, 2))
                        g_in = torch.rot90(g_in, k, dims=(0, 1))
                        g_out = torch.rot90(g_out, k, dims=(0, 1))
                        if rng.random() < 0.5:
                            img = torch.flip(img, dims=(2,))
                            nxt = torch.flip(nxt, dims=(2,))
                            g_in = torch.flip(g_in, dims=(1,))
                            g_out = torch.flip(g_out, dims=(1,))
                    yield {
                        "image": img,
                        "next_image": nxt,
                        "grid_in": g_in,
                        "grid_out": g_out,
                        "action_disc": torch.tensor(a_idx, dtype=torch.long),
                        "action_click": torch.zeros(2),
                        "task_id": action_key,
                        "source": tag,
                    }
            if not advanced:
                break
        if once:
            return


def multi_arc_batch_iter(
    roots: list[tuple[str, str | Path]],
    batch_size: int = 16,
    **kwargs,
) -> Iterator[dict]:
    buf: list[dict] = []
    for item in multi_arc_iter(roots, **kwargs):
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
                "source": [b["source"] for b in buf],
            }
            buf = []
