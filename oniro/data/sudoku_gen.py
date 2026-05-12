"""Procedural Sudoku grid generator → ONIRO-format pairs.

Each generated puzzle is a 9×9 grid with int values 1-9 and 0 (empty cell).
We yield (input=puzzle_with_blanks, output=solved_puzzle) pairs in the same
schema as ARC tasks. Casts to int8 for compatibility with grid_in/grid_out.

Generator strategy: backtrack solver fills random complete board, then mask
~30-50% of cells.
"""

from __future__ import annotations

from typing import Iterator
import random

import numpy as np
import torch

from oniro.data.arc2_loader import _grid_to_image, _grid_to_int_tensor


def _solve_backtrack(board: np.ndarray) -> bool:
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                vals = list(range(1, 10))
                random.shuffle(vals)
                for v in vals:
                    if _valid(board, r, c, v):
                        board[r, c] = v
                        if _solve_backtrack(board):
                            return True
                        board[r, c] = 0
                return False
    return True


def _valid(board: np.ndarray, r: int, c: int, v: int) -> bool:
    if v in board[r]:
        return False
    if v in board[:, c]:
        return False
    br, bc = (r // 3) * 3, (c // 3) * 3
    if v in board[br:br + 3, bc:bc + 3]:
        return False
    return True


def gen_sudoku_pair(mask_rate: float = 0.4, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    board = np.zeros((9, 9), dtype=np.int8)
    _solve_backtrack(board)
    solved = board.copy()
    mask = np.random.RandomState(rng.randint(0, 2**31 - 1)).rand(9, 9) < mask_rate
    puzzle = solved.copy()
    puzzle[mask] = 0
    return puzzle, solved


def sudoku_iter(
    image_size: int = 96,
    grid_target_side: int = 32,
    mask_rate: float = 0.4,
    seed: int = 0,
    once: bool = False,
    action_vocab: int = 1024,
    sudoku_action_offset: int = 700,
) -> Iterator[dict]:
    """Action ids namespaced to sudoku slot in [700..900).

    Maps Sudoku 9×9 with values 0-9 (we keep 0 as blank, 1-9 as ARC colors 1-9).
    """
    rng = random.Random(seed)
    n = 0
    while True:
        puzzle, solved = gen_sudoku_pair(mask_rate=mask_rate, rng=rng)
        img = _grid_to_image(puzzle.tolist(), image_size)
        nxt = _grid_to_image(solved.tolist(), image_size)
        g_in = _grid_to_int_tensor(puzzle.tolist(), grid_target_side)
        g_out = _grid_to_int_tensor(solved.tolist(), grid_target_side)
        a_idx = (sudoku_action_offset + n) % action_vocab
        yield {
            "image": img, "next_image": nxt,
            "grid_in": g_in, "grid_out": g_out,
            "action_disc": torch.tensor(a_idx, dtype=torch.long),
            "action_click": torch.zeros(2),
            "task_id": f"sudoku::{n}", "source": "sudoku",
        }
        n += 1
        if once and n >= 200:
            return
