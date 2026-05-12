"""Cellular Automata procedural generator.

Generates (state_t, state_t+1) grid pairs from:
    - Conway's Game of Life (B3/S23)
    - rule110 (1D, embedded into 2D row)
    - Variable B/S rules sampled uniformly (for diversity)

All grids encoded as int64 with values in {0, 1} (background + alive).
Used as pretraining signal for local-rule reasoning in ARC.
"""

from __future__ import annotations

import random
import numpy as np


def _conway_step(g: np.ndarray) -> np.ndarray:
    """One step of Conway's Game of Life (B3/S23). g: (H, W) int."""
    H, W = g.shape
    # 8-neighborhood sum via shifts
    n = np.zeros_like(g)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            n += np.roll(np.roll(g, di, axis=0), dj, axis=1)
    born = (g == 0) & (n == 3)
    survive = (g == 1) & ((n == 2) | (n == 3))
    return (born | survive).astype(g.dtype)


def _bs_step(g: np.ndarray, B: set[int], S: set[int]) -> np.ndarray:
    """Generic B/S cellular automaton step (Moore neighborhood)."""
    n = np.zeros_like(g)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            n += np.roll(np.roll(g, di, axis=0), dj, axis=1)
    born_mask = np.zeros_like(g, dtype=bool)
    survive_mask = np.zeros_like(g, dtype=bool)
    for b in B:
        born_mask |= (n == b)
    for s in S:
        survive_mask |= (n == s)
    born = (g == 0) & born_mask
    survive = (g == 1) & survive_mask
    return (born | survive).astype(g.dtype)


def _rule110_row(row: np.ndarray, rule: int = 110) -> np.ndarray:
    """1D elementary CA: 3-cell neighborhood, 8 patterns, rule encoded as bits."""
    n = row.shape[0]
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right
    out = np.zeros_like(row)
    for p in range(8):
        if (rule >> p) & 1:
            out[idx == p] = 1
    return out


def gen_conway_pair(rng: random.Random | None = None,
                    side: int = 16, density: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Random Conway init -> one-step output. Returns (state_t, state_t+1)."""
    if rng is None:
        rng = random.Random()
    seed = rng.randint(0, 2**31 - 1)
    nrng = np.random.RandomState(seed)
    g = (nrng.random((side, side)) < density).astype(np.int64)
    g1 = _conway_step(g)
    return g, g1


def gen_bs_pair(rng: random.Random | None = None, side: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Random B/S rule -> one-step. Adds diversity beyond Conway."""
    if rng is None:
        rng = random.Random()
    # sample 1-3 B values and 1-4 S values
    B = set(rng.sample(range(0, 9), k=rng.randint(1, 3)))
    S = set(rng.sample(range(0, 9), k=rng.randint(1, 4)))
    seed = rng.randint(0, 2**31 - 1)
    nrng = np.random.RandomState(seed)
    density = rng.uniform(0.2, 0.5)
    g = (nrng.random((side, side)) < density).astype(np.int64)
    g1 = _bs_step(g, B, S)
    return g, g1


def gen_rule110_pair(rng: random.Random | None = None, side: int = 16,
                     rule: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Random 1D row, evolve N steps as 2D grid (rows = time slices)."""
    if rng is None:
        rng = random.Random()
    if rule is None:
        rule = rng.choice([30, 90, 110, 184])  # interesting elementary rules
    seed = rng.randint(0, 2**31 - 1)
    nrng = np.random.RandomState(seed)
    row = (nrng.random(side) < 0.5).astype(np.int64)
    grid_in = np.zeros((side, side), dtype=np.int64)
    grid_out = np.zeros((side, side), dtype=np.int64)
    grid_in[0] = row
    cur = row
    for t in range(side):
        cur = _rule110_row(cur, rule=rule)
        if t < side:
            grid_out[t] = cur
    return grid_in, grid_out


def gen_ca_pair(rng: random.Random | None = None, side: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Random pick from {Conway, B/S, rule110}."""
    if rng is None:
        rng = random.Random()
    r = rng.random()
    if r < 0.5:
        return gen_conway_pair(rng, side=side)
    elif r < 0.8:
        return gen_bs_pair(rng, side=side)
    else:
        return gen_rule110_pair(rng, side=side)
