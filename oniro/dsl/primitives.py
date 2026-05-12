"""ARC DSL primitives — building blocks for symbolic grid transformations.

Each primitive operates on a numpy grid and returns transformed grid.
Composed via beam search to find program that maps demo input → demo output.
SOTA Kaggle ARC solvers use this neural+symbolic hybrid.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# Geometric transformations
def rot90(g: np.ndarray) -> np.ndarray:  return np.rot90(g, 1)
def rot180(g: np.ndarray) -> np.ndarray: return np.rot90(g, 2)
def rot270(g: np.ndarray) -> np.ndarray: return np.rot90(g, 3)
def flip_h(g: np.ndarray) -> np.ndarray: return np.flip(g, axis=1).copy()
def flip_v(g: np.ndarray) -> np.ndarray: return np.flip(g, axis=0).copy()
def transpose(g: np.ndarray) -> np.ndarray: return g.T.copy()
def identity(g: np.ndarray) -> np.ndarray: return g.copy()


# Color transformations
def make_swap_colors(a: int, b: int) -> Callable[[np.ndarray], np.ndarray]:
    def fn(g: np.ndarray) -> np.ndarray:
        out = g.copy()
        m_a = (g == a); m_b = (g == b)
        out[m_a] = b; out[m_b] = a
        return out
    fn.__name__ = f"swap_{a}_{b}"
    return fn


def make_recolor(src: int, dst: int) -> Callable[[np.ndarray], np.ndarray]:
    def fn(g: np.ndarray) -> np.ndarray:
        out = g.copy()
        out[out == src] = dst
        return out
    fn.__name__ = f"recolor_{src}_{dst}"
    return fn


# Structural transformations
def crop_to_bbox(g: np.ndarray) -> np.ndarray:
    """Crop to bounding box of non-zero pixels."""
    if not (g != 0).any():
        return g.copy()
    rows = np.any(g != 0, axis=1)
    cols = np.any(g != 0, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return g[r0:r1 + 1, c0:c1 + 1].copy()


def tile_2x2(g: np.ndarray) -> np.ndarray:
    return np.tile(g, (2, 2))


def double_size(g: np.ndarray) -> np.ndarray:
    return np.kron(g, np.ones((2, 2), dtype=g.dtype))


def half_size(g: np.ndarray) -> np.ndarray:
    """Downsample 2x via simple stride."""
    h, w = g.shape
    if h < 2 or w < 2:
        return g.copy()
    return g[::2, ::2].copy()


# Library
PRIMITIVES_GEOM = [identity, rot90, rot180, rot270, flip_h, flip_v, transpose]
PRIMITIVES_STRUCT = [crop_to_bbox, tile_2x2, double_size, half_size]


def all_color_swaps() -> list:
    out = []
    for a in range(10):
        for b in range(a + 1, 10):
            out.append(make_swap_colors(a, b))
    return out


def all_recolors() -> list:
    out = []
    for s in range(10):
        for d in range(10):
            if s != d:
                out.append(make_recolor(s, d))
    return out


def grid_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))
