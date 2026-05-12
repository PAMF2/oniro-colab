"""Generic ARC-format JSON loader.

ARC-AGI-1, ARC-AGI-2, RE-ARC, ConceptARC, Mini-ARC, ARC-Heavy, BARC and most
of the neoneye/arc-dataset-collection share the same JSON format:
    {"train": [{"input":..., "output":...}, ...],
     "test":  [{"input":..., "output":...}, ...]}

This loader walks any directory tree and yields (input_grid, output_grid)
pairs from every JSON file it finds. Grids returned as np.int64 2D arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _pairs_from_task(task: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for section in ("train", "test"):
        for p in task.get(section, []):
            if "input" not in p or "output" not in p:
                continue
            inp = np.asarray(p["input"], dtype=np.int64)
            outp = np.asarray(p["output"], dtype=np.int64)
            if inp.ndim != 2 or outp.ndim != 2:
                continue
            out.append((inp, outp))
    return out


def load_arc_dir(root: str | Path, max_tasks: int | None = None) -> list[list]:
    """Return list[ list[(inp, out)] ] grouped by task file."""
    root = Path(root)
    files = sorted(root.rglob("*.json"))
    tasks = []
    for tf in files:
        try:
            with tf.open() as f:
                task = json.load(f)
        except Exception:
            continue
        pairs = _pairs_from_task(task)
        if pairs:
            tasks.append(pairs)
        if max_tasks is not None and len(tasks) >= max_tasks:
            break
    return tasks


def flat_pairs(tasks: list[list]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flatten task-grouped list into a single (inp, out) list."""
    out = []
    for t in tasks:
        out.extend(t)
    return out
