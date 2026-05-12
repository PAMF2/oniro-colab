"""ARC-AGI-3 environment.

Two backends:

- **synthetic** (default, no external deps): a self-contained grid puzzle env that
  mirrors ARC-AGI-3's action/observation contract. Goal of each level: match a
  target pattern by moving a cursor + painting cells. 5 levels per game, harder
  patterns per level. Lets ONIRO train end-to-end on a CPU/Colab without any
  network access.

- **remote**: wraps `arcprize/ARC-AGI-3-Agents` over subprocess. Used at real
  Kaggle submission time. Not loaded by default.

The same `FrameData` / `GameAction` types serve both backends, so anything that
runs on synthetic transfers to remote unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class GameAction(IntEnum):
    ACTION1 = 1   # cursor up
    ACTION2 = 2   # cursor down
    ACTION3 = 3   # cursor left
    ACTION4 = 4   # cursor right
    ACTION5 = 5   # paint cell at cursor with current color, advance color
    CLICK = 6     # warp cursor to (x, y)


@dataclass
class FrameData:
    grid: np.ndarray            # (64, 64) int8, values in [0, 15]
    score: float                # cumulative best match fraction this game
    level: int                  # current level 1..5
    available_actions: list[GameAction]
    state: str                  # "running" | "level_complete" | "game_complete" | "dead"
    raw: dict[str, Any] = field(default_factory=dict)


class _SyntheticARC3Env:
    PALETTE = 16
    GRID = 64
    LEVELS = 5
    LEVEL_COMPLETE_THRESHOLD = 0.85

    def __init__(self, game_id: str = "synth-0", seed: int = 0):
        self.game_id = game_id
        self.rng = np.random.RandomState(seed)
        self._level = 1
        self._step = 0
        self._cursor = (0, 0)
        self._color_idx = 1
        self._grid = np.zeros((self.GRID, self.GRID), dtype=np.int8)
        self._target = self._grid.copy()
        self._state = "init"
        self._cum_score = 0.0
        self._actions_per_level: dict[int, int] = {}
        self._best_match: dict[int, float] = {}

    def _make_target(self, level: int) -> np.ndarray:
        t = np.zeros((self.GRID, self.GRID), dtype=np.int8)
        n_shapes = level + 1
        for _ in range(n_shapes):
            x = int(self.rng.randint(0, self.GRID - 8))
            y = int(self.rng.randint(0, self.GRID - 8))
            w = int(self.rng.randint(4, 12))
            h = int(self.rng.randint(4, 12))
            c = int(self.rng.randint(1, self.PALETTE))
            t[y : y + h, x : x + w] = c
        return t

    def _match_fraction(self) -> float:
        mask = self._target > 0
        total = int(mask.sum())
        if total == 0:
            return 0.0
        correct = int(((self._grid == self._target) & mask).sum())
        wrong = int(((self._grid > 0) & (self._grid != self._target)).sum())
        recall = correct / max(total, 1)
        precision_pen = wrong / max(total, 1)
        return float(max(0.0, recall - 0.5 * precision_pen))

    def _frame(self) -> FrameData:
        return FrameData(
            grid=self._grid.copy(),
            score=self._cum_score,
            level=self._level,
            available_actions=[
                GameAction.ACTION1, GameAction.ACTION2,
                GameAction.ACTION3, GameAction.ACTION4,
                GameAction.ACTION5, GameAction.CLICK,
            ],
            state=self._state,
            raw={
                "cursor": list(self._cursor),
                "color_idx": self._color_idx,
                "match": self._match_fraction(),
                "best_per_level": dict(self._best_match),
                "steps_per_level": dict(self._actions_per_level),
            },
        )

    def reset(self) -> FrameData:
        self._level = 1
        self._step = 0
        self._cursor = (self.GRID // 2, self.GRID // 2)
        self._color_idx = 1
        self._grid = np.zeros((self.GRID, self.GRID), dtype=np.int8)
        self._target = self._make_target(self._level)
        self._state = "running"
        self._cum_score = 0.0
        self._actions_per_level = {}
        self._best_match = {}
        return self._frame()

    def step(self, action: GameAction, data: dict[str, int] | None = None) -> FrameData:
        if self._state != "running":
            return self._frame()

        self._step += 1
        self._actions_per_level[self._level] = self._actions_per_level.get(self._level, 0) + 1
        x, y = self._cursor

        if action == GameAction.ACTION1:
            y = max(0, y - 1)
        elif action == GameAction.ACTION2:
            y = min(self.GRID - 1, y + 1)
        elif action == GameAction.ACTION3:
            x = max(0, x - 1)
        elif action == GameAction.ACTION4:
            x = min(self.GRID - 1, x + 1)
        elif action == GameAction.ACTION5:
            self._grid[y, x] = self._color_idx
            self._color_idx = (self._color_idx % (self.PALETTE - 1)) + 1
        elif action == GameAction.CLICK:
            if data is not None:
                x = int(data.get("x", x)) % self.GRID
                y = int(data.get("y", y)) % self.GRID

        self._cursor = (x, y)

        m = self._match_fraction()
        self._best_match[self._level] = max(self._best_match.get(self._level, 0.0), m)
        self._cum_score = sum(self._best_match.values()) / self.LEVELS

        if m >= self.LEVEL_COMPLETE_THRESHOLD:
            if self._level >= self.LEVELS:
                self._state = "game_complete"
            else:
                self._level += 1
                self._target = self._make_target(self._level)
                self._grid = np.zeros_like(self._grid)
                self._cursor = (self.GRID // 2, self.GRID // 2)
                self._color_idx = 1

        return self._frame()

    @property
    def step_count(self) -> int:
        return self._step

    def target_grid(self) -> np.ndarray:
        return self._target.copy()


class _RemoteARC3Env:
    """Subprocess wrapper around `arcprize/ARC-AGI-3-Agents`.

    Wired only at Kaggle submission time. Until then, raises with a clear error
    so callers do not silently fall through.
    """

    def __init__(self, game_id: str, seed: int = 0):
        self.game_id = game_id
        raise NotImplementedError(
            "Remote ARC-3 backend not wired in this checkout. "
            "Use backend='synthetic' (default) until you have the runtime."
        )


class ARC3Env:
    """Default ARC-AGI-3 env. `backend='synthetic'` is fully functional offline."""

    def __init__(self, game_id: str = "synth-0", backend: str = "synthetic", seed: int = 0):
        self.game_id = game_id
        self.backend = backend
        if backend == "synthetic":
            self._impl = _SyntheticARC3Env(game_id=game_id, seed=seed)
        elif backend in ("remote", "kaggle"):
            self._impl = _RemoteARC3Env(game_id=game_id, seed=seed)
        else:
            raise ValueError(f"unknown backend: {backend!r}")

    def reset(self) -> FrameData:
        return self._impl.reset()

    def step(self, action: GameAction, data: dict[str, int] | None = None) -> FrameData:
        return self._impl.step(action, data=data)

    @property
    def step_count(self) -> int:
        return self._impl.step_count

    def target_grid(self) -> np.ndarray:
        return self._impl.target_grid()
