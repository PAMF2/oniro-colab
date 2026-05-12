"""DSL solver: beam search over primitive sequences to fit demo pairs.

For each task:
    1. Try every length-1 and length-2 program from library.
    2. Filter by: applies cleanly to ALL demo inputs producing demo outputs.
    3. If found, apply to test input.

Falls back to neural URM prediction if no program found.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from oniro.dsl.primitives import (
    PRIMITIVES_GEOM, PRIMITIVES_STRUCT,
    all_color_swaps, all_recolors,
    grid_equal, identity,
)


def _apply(prog: list[Callable], grid: np.ndarray) -> np.ndarray:
    out = grid
    for fn in prog:
        try:
            out = fn(out)
        except Exception:
            return None
    return out


def _matches_all_demos(prog: list[Callable], demos: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    for di, do in demos:
        pred = _apply(prog, di)
        if pred is None or not grid_equal(pred, do):
            return False
    return True


def search_program(
    demos: list[tuple[np.ndarray, np.ndarray]],
    max_depth: int = 2,
    library: list[Callable] | None = None,
) -> list[Callable] | None:
    """Beam search: try identity → length-1 → length-2 programs."""
    if library is None:
        library = PRIMITIVES_GEOM + PRIMITIVES_STRUCT + all_color_swaps()

    # depth 0: identity
    if _matches_all_demos([identity], demos):
        return [identity]

    # depth 1
    for fn in library:
        if _matches_all_demos([fn], demos):
            return [fn]

    if max_depth < 2:
        return None

    # depth 2
    for f1 in library:
        for f2 in library:
            if _matches_all_demos([f1, f2], demos):
                return [f1, f2]

    return None


def solve_task(
    task: dict,
    neural_fallback: Callable | None = None,
    max_depth: int = 2,
) -> dict:
    """
    task: ARC-AGI task dict with 'train' (demos) and 'test'.
    neural_fallback: callable(grid_np) -> predicted_grid_np, used if no program found.

    Returns dict with:
        program: list of primitive names (or None)
        predictions: per test-pair predicted grids
        method: 'dsl' or 'neural'
    """
    demos_np = [(np.asarray(p["input"], dtype=np.int8),
                 np.asarray(p["output"], dtype=np.int8))
                for p in task.get("train", [])]

    prog = search_program(demos_np, max_depth=max_depth)

    predictions = []
    for tp in task.get("test", []):
        ti = np.asarray(tp["input"], dtype=np.int8)
        if prog is not None:
            pred = _apply(prog, ti)
            if pred is not None:
                predictions.append(pred)
                continue
        # fallback
        if neural_fallback is not None:
            predictions.append(neural_fallback(ti))
        else:
            predictions.append(ti.copy())  # identity fallback

    return {
        "program": [fn.__name__ for fn in prog] if prog else None,
        "predictions": predictions,
        "method": "dsl" if prog else "neural",
    }
