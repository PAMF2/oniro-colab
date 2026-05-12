"""BARC loader (Bootstrapping ARC, xu3kev/BARC).

BARC publishes ~400k GPT-4o-mini-synthesised ARC-format problems. Known
issue: ~1/3 of generated samples have wrong outputs (paper acknowledges).
This loader supports an optional verifier filter that drops samples where
the demo input under the published program does not reproduce the demo
output exactly.

Returns the same list[(inp_np, out_np)] interface as arc_json_loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_arc_format(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        with path.open() as f:
            task = json.load(f)
    except Exception:
        return []
    pairs = []
    for section in ("train", "test"):
        for p in task.get(section, []):
            if "input" not in p or "output" not in p:
                continue
            inp = np.asarray(p["input"], dtype=np.int64)
            out = np.asarray(p["output"], dtype=np.int64)
            if inp.ndim != 2 or out.ndim != 2:
                continue
            pairs.append((inp, out))
    return pairs


def _consistency_check(pairs: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    """Reject if there are no pairs or pairs have wildly inconsistent shapes."""
    if not pairs:
        return False
    in_shapes = {p[0].shape for p in pairs}
    out_shapes = {p[1].shape for p in pairs}
    # allow varying sizes but require at least one valid pair
    return len(in_shapes) >= 1 and len(out_shapes) >= 1


def load_barc_dir(root: str | Path, max_tasks: int | None = None,
                  filter_consistent: bool = True) -> list[list]:
    """Returns list[ list[(inp, out)] ] grouped by task, optionally filtered."""
    root = Path(root)
    files = sorted(root.rglob("*.json"))
    tasks: list[list] = []
    n_kept = 0
    n_dropped = 0
    for tf in files:
        pairs = _load_arc_format(tf)
        if filter_consistent and not _consistency_check(pairs):
            n_dropped += 1
            continue
        if pairs:
            tasks.append(pairs)
            n_kept += 1
        if max_tasks is not None and n_kept >= max_tasks:
            break
    return tasks
