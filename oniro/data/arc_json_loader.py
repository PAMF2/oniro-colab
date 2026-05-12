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


def _pairs_from_task(task) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tolerant pair extractor.

    Accepts THREE on-disk shapes:
      1. Standard ARC: {"train": [{"input", "output"}, ...], "test": [...]}.
      2. RE-ARC: a flat list of pairs [{"input", "output"}, ...] (no train/test split).
      3. BARC HF format inside JSON: {"examples": [{"input", "output"}, ...]}.
    Returns: list[(inp_np, out_np)] always as 2D int64 arrays. Items that
    fail shape/key checks are silently dropped.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []

    def _take(items):
        for p in items:
            if not isinstance(p, dict):
                continue
            if "input" not in p or "output" not in p:
                continue
            try:
                inp = np.asarray(p["input"], dtype=np.int64)
                outp = np.asarray(p["output"], dtype=np.int64)
            except Exception:
                continue
            if inp.ndim != 2 or outp.ndim != 2:
                continue
            out.append((inp, outp))

    if isinstance(task, list):
        # RE-ARC: flat array of pairs
        _take(task)
    elif isinstance(task, dict):
        # ARC standard: train + test sections
        if "train" in task or "test" in task:
            for section in ("train", "test"):
                _take(task.get(section, []))
        # BARC / other: top-level "examples"
        if "examples" in task:
            _take(task["examples"])
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
