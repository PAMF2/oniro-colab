"""Rolling OOD-split buffer for the Gödel gate.

Holds short transition samples from:
    - different ARC-AGI-3 games than the one currently being played
    - held-out Open-X-Embodiment robot trajectories
    - synthetic distractor frames (noise, color swap, scrambled grids)

The Gödel gate queries this buffer for predictive-loss deltas before accepting any
online adaptation step.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable
import random

import torch


class OODBuffer:
    def __init__(self, capacity_per_split: int = 32, n_splits: int = 10, seed: int = 0):
        self.n_splits = n_splits
        self.cap = capacity_per_split
        self.splits: list[deque] = [deque(maxlen=capacity_per_split) for _ in range(n_splits)]
        self.rng = random.Random(seed)

    def add(self, split_idx: int, transition: dict) -> None:
        if 0 <= split_idx < self.n_splits:
            self.splits[split_idx].append(transition)

    def sample(self) -> list[dict]:
        out: list[dict] = []
        for s in self.splits:
            if s:
                out.append(self.rng.choice(list(s)))
        return out

    def __len__(self) -> int:
        return sum(len(s) for s in self.splits)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save([list(s) for s in self.splits], p)

    def load(self, path: str | Path) -> None:
        data = torch.load(path)
        self.splits = [deque(d, maxlen=self.cap) for d in data]
