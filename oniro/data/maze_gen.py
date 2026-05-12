"""Procedural maze grid generator → ONIRO-format pairs.

input  = maze with walls (1) + free cells (0) + start (2) + goal (3)
output = same maze with shortest path marked (4) along the route

Generator: random binary maze with guaranteed connected start→goal via DFS
carving. Path solver: BFS.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator
import random

import numpy as np
import torch

from oniro.data.arc2_loader import _grid_to_image, _grid_to_int_tensor


def _carve_maze(size: int, rng: random.Random) -> np.ndarray:
    """Random walls density ~30%, then ensure start-goal connectivity by carving."""
    grid = np.zeros((size, size), dtype=np.int8)
    for r in range(size):
        for c in range(size):
            if rng.random() < 0.3:
                grid[r, c] = 1
    grid[0, 0] = 2  # start
    grid[size - 1, size - 1] = 3  # goal
    # Carve a guaranteed path along outer L: top row + right col
    grid[0, :] = np.where(grid[0, :] == 1, 0, grid[0, :])
    grid[:, -1] = np.where(grid[:, -1] == 1, 0, grid[:, -1])
    grid[0, 0] = 2
    grid[size - 1, size - 1] = 3
    return grid


def _bfs_path(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    start = tuple(map(int, np.argwhere(grid == 2)[0]))
    goal = tuple(map(int, np.argwhere(grid == 3)[0]))
    visited = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if grid[nr, nc] != 1:
                    visited[(nr, nc)] = (r, c)
                    q.append((nr, nc))
    path = []
    cur = goal
    while cur is not None and cur in visited:
        path.append(cur)
        cur = visited[cur]
    return path[::-1] if path and path[0] == start else []


def gen_maze_pair(size: int = 12, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    grid = _carve_maze(size, rng)
    path = _bfs_path(grid)
    solved = grid.copy()
    for r, c in path[1:-1]:
        solved[r, c] = 4
    return grid, solved


def maze_iter(
    image_size: int = 96,
    grid_target_side: int = 32,
    size: int = 12,
    seed: int = 0,
    once: bool = False,
    action_vocab: int = 1024,
    maze_action_offset: int = 850,
) -> Iterator[dict]:
    rng = random.Random(seed)
    n = 0
    while True:
        puzzle, solved = gen_maze_pair(size, rng)
        img = _grid_to_image(puzzle.tolist(), image_size)
        nxt = _grid_to_image(solved.tolist(), image_size)
        g_in = _grid_to_int_tensor(puzzle.tolist(), grid_target_side)
        g_out = _grid_to_int_tensor(solved.tolist(), grid_target_side)
        a_idx = (maze_action_offset + n) % action_vocab
        yield {
            "image": img, "next_image": nxt,
            "grid_in": g_in, "grid_out": g_out,
            "action_disc": torch.tensor(a_idx, dtype=torch.long),
            "action_click": torch.zeros(2),
            "task_id": f"maze::{n}", "source": "maze",
        }
        n += 1
        if once and n >= 200:
            return
