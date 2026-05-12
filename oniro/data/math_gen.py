"""Procedural visual math problem generator → ONIRO grid pairs.

Tasks (each fits in N×N grid):
  - SUM: input shows two color-counts (rows of N1 dots + row of N2 dots).
         output shows row of (N1+N2) dots.
  - MAGIC: 3×3 magic square with one cell hidden → predict.
  - SEQ: arithmetic sequence row with one term hidden → predict.
  - COMPARE: two rows; output marks longer one.

Easy procedural, fits ARC grid framework (0-9 cell values).
"""

from __future__ import annotations

from typing import Iterator
import random

import numpy as np
import torch

from oniro.data.arc2_loader import _grid_to_int_tensor


def _empty(side: int) -> np.ndarray:
    return np.zeros((side, side), dtype=np.int8)


def gen_sum_pair(side: int = 16, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    a = rng.randint(1, side // 2 - 1)
    b = rng.randint(1, side // 2 - 1)
    inp = _empty(side); out = _empty(side)
    inp[2, :a] = 3
    inp[5, :b] = 5
    out[2, :a + b] = 4
    return inp, out


def gen_seq_pair(side: int = 16, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    start = rng.randint(1, 3)
    step = rng.randint(1, 2)
    seq = [start + i * step for i in range(min(6, side - 1))]
    hide_at = rng.randint(0, len(seq) - 1)
    inp = _empty(side); out = _empty(side)
    for i, v in enumerate(seq):
        if i == hide_at:
            inp[3, i] = 9  # marker for missing
        else:
            inp[3, i] = min(v, 9)
        out[3, i] = min(v, 9)
    return inp, out


def gen_compare_pair(side: int = 16, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    a = rng.randint(2, side - 2)
    b = rng.randint(2, side - 2)
    while a == b:
        b = rng.randint(2, side - 2)
    inp = _empty(side); out = _empty(side)
    inp[3, :a] = 3
    inp[6, :b] = 5
    winner = 3 if a > b else 5
    row = 3 if a > b else 6
    out[row, :max(a, b)] = winner
    return inp, out


def gen_math_pair(side: int = 16, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    task = rng.choice(["sum", "seq", "compare"])
    if task == "sum":
        return gen_sum_pair(side, rng)
    if task == "seq":
        return gen_seq_pair(side, rng)
    return gen_compare_pair(side, rng)


def math_iter(
    image_size: int = 96,
    grid_target_side: int = 16,
    seed: int = 0,
    once: bool = False,
    action_vocab: int = 1024,
    math_action_offset: int = 950,
) -> Iterator[dict]:
    from oniro.data.arc2_loader import _grid_to_image
    rng = random.Random(seed)
    n = 0
    while True:
        puzzle, solved = gen_math_pair(side=grid_target_side, rng=rng)
        img = _grid_to_image(puzzle.tolist(), image_size)
        nxt = _grid_to_image(solved.tolist(), image_size)
        g_in = _grid_to_int_tensor(puzzle.tolist(), grid_target_side)
        g_out = _grid_to_int_tensor(solved.tolist(), grid_target_side)
        a_idx = (math_action_offset + n) % action_vocab
        yield {
            "image": img, "next_image": nxt,
            "grid_in": g_in, "grid_out": g_out,
            "action_disc": torch.tensor(a_idx, dtype=torch.long),
            "action_click": torch.zeros(2),
            "task_id": f"math::{n}", "source": "math",
        }
        n += 1
        if once and n >= 200:
            return
