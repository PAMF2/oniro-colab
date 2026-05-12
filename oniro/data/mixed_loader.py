"""Mixed loader: ARC + Sudoku + Maze interleaved.

Sample-rate-controlled mix: each draw picks source by configured weights, e.g.
{'arc': 0.7, 'sudoku': 0.2, 'maze': 0.1}.

Wraps multi_arc_iter (for ARC family) + sudoku_iter + maze_iter.
"""

from __future__ import annotations

from typing import Iterator
import random

import torch

from oniro.data.multi_arc_loader import multi_arc_iter
from oniro.data.sudoku_gen import sudoku_iter
from oniro.data.maze_gen import maze_iter


def mixed_iter(
    arc_roots: list[tuple[str, str]],
    image_size: int = 96,
    grid_target_side: int = 32,
    weights: dict[str, float] | None = None,
    seed: int = 0,
    augment: bool = True,
    once: bool = False,
) -> Iterator[dict]:
    weights = weights or {"arc": 0.7, "sudoku": 0.2, "maze": 0.1}
    iters = {}
    if weights.get("arc", 0) > 0 and arc_roots:
        iters["arc"] = multi_arc_iter(
            arc_roots, image_size=image_size, grid_target_side=grid_target_side,
            shuffle=True, seed=seed, augment=augment, once=once,
        )
    if weights.get("sudoku", 0) > 0:
        iters["sudoku"] = sudoku_iter(
            image_size=image_size, grid_target_side=grid_target_side,
            seed=seed + 1, once=once,
        )
    if weights.get("maze", 0) > 0:
        iters["maze"] = maze_iter(
            image_size=image_size, grid_target_side=grid_target_side,
            seed=seed + 2, once=once,
        )

    sources = list(iters.keys())
    probs = [weights[k] for k in sources]
    total = sum(probs)
    probs = [p / total for p in probs]
    rng = random.Random(seed)

    while True:
        s = rng.choices(sources, weights=probs, k=1)[0]
        try:
            yield next(iters[s])
        except StopIteration:
            if once:
                return
            iters[s] = {"arc": multi_arc_iter, "sudoku": sudoku_iter, "maze": maze_iter}[s]


def mixed_batch_iter(
    arc_roots: list[tuple[str, str]],
    batch_size: int = 16,
    **kwargs,
) -> Iterator[dict]:
    buf: list[dict] = []
    for item in mixed_iter(arc_roots, **kwargs):
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
