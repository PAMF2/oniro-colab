"""H-ARC loader (human ARC traces).

H-ARC (https://arc-visualizations.github.io/) is a corpus of >1700 humans
solving 400+400 ARC tasks, capturing step-by-step interactions. The raw
format publishes per-task solver attempts; for training purposes we only
need the final-state (i,o) pairs which are already ARC-format compatible.

This loader treats the H-ARC repo as a generic ARC-format JSON tree.
Many H-ARC dumps include a per-attempt status field; we keep only attempts
flagged as 'success' / 'solved' (loose match) to avoid learning wrong
solutions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_SUCCESS_TOKENS = {"success", "solved", "correct", "true", "1"}


def _is_success(task: dict) -> bool:
    # tolerant: many H-ARC dumps key the status differently
    has_status_key = False
    for k in ("status", "solved", "success", "is_correct", "label"):
        v = task.get(k)
        if v is None:
            continue
        has_status_key = True
        if isinstance(v, bool):
            if v: return True
            continue
        if str(v).lower() in _SUCCESS_TOKENS:
            return True
    # If any status key was present but none matched success, treat as failure
    if has_status_key:
        return False
    # No status keys at all → tolerate (treat as success if has train+test)
    return "train" in task and "test" in task


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


def load_harc_dir(root: str | Path, max_tasks: int | None = None,
                  require_success: bool = True) -> list[list]:
    root = Path(root)
    files = sorted(root.rglob("*.json"))
    tasks: list[list] = []
    for tf in files:
        try:
            with tf.open() as f:
                task = json.load(f)
        except Exception:
            continue
        if require_success and not _is_success(task):
            continue
        pairs = _pairs_from_task(task)
        if pairs:
            tasks.append(pairs)
        if max_tasks is not None and len(tasks) >= max_tasks:
            break
    return tasks
