"""Procedural geometry puzzle generator (v41.1).

Pedro: "eval em mat, sudoku, geometria, ARC-AGI 1, 2, 3".

Pure-numeric geometry tasks rendered as int8 grids. Tasks pair (input,
expected_output) so a recursive grid model can learn:

    Tasks:
    - draw_line: draw straight line between two endpoints
    - reflect_axis: reflect a shape across given axis
    - rotate_90: rotate input by 90° (input = original, output = rotated)
    - mirror_h / mirror_v: horizontal/vertical mirror
    - count_corners: count corners of a rectangle, output as bar
    - bbox: bounding-box extraction → fill rectangle of input footprint
    - centroid: mark centroid of input shape with single cell
    - symmetric_completion: complete symmetric pattern from half
    - convex_hull_fill: fill convex hull of scattered points
    - line_intersection: predict where two lines cross
    - parallel_translation: translate shape by fixed vector
    - shape_perimeter: number of boundary cells output as bar
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np


def _empty(side: int) -> np.ndarray:
    return np.zeros((side, side), dtype=np.int8)


def gen_draw_line(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Two endpoints marked → output draws line."""
    inp, out = _empty(side), _empty(side)
    r1, c1 = rng.randint(0, side - 1), rng.randint(0, side - 1)
    r2, c2 = rng.randint(0, side - 1), rng.randint(0, side - 1)
    inp[r1, c1] = 3
    inp[r2, c2] = 3
    # Bresenham
    dr, dc = abs(r2 - r1), abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc
    r, c = r1, c1
    while True:
        out[r, c] = 4
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return inp, out


def gen_mirror_h(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Random shape on left half → output is full grid with mirrored shape."""
    inp = _empty(side)
    n_cells = rng.randint(3, side // 2)
    for _ in range(n_cells):
        r = rng.randint(0, side - 1)
        c = rng.randint(0, side // 2 - 1)
        inp[r, c] = rng.randint(1, 8)
    out = inp.copy()
    out[:, side // 2:] = np.fliplr(inp[:, :side // 2])
    return inp, out


def gen_mirror_v(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    inp = _empty(side)
    n_cells = rng.randint(3, side // 2)
    for _ in range(n_cells):
        r = rng.randint(0, side // 2 - 1)
        c = rng.randint(0, side - 1)
        inp[r, c] = rng.randint(1, 8)
    out = inp.copy()
    out[side // 2:, :] = np.flipud(inp[:side // 2, :])
    return inp, out


def gen_rotate_90(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    inp = _empty(side)
    n_cells = rng.randint(3, min(8, side))
    for _ in range(n_cells):
        r = rng.randint(0, side - 1)
        c = rng.randint(0, side - 1)
        inp[r, c] = rng.randint(1, 8)
    out = np.rot90(inp).copy()
    return inp, out


def gen_bbox(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Scattered points → output is bounding box rectangle."""
    inp = _empty(side)
    n_pts = rng.randint(3, 8)
    rs = [rng.randint(2, side - 3) for _ in range(n_pts)]
    cs = [rng.randint(2, side - 3) for _ in range(n_pts)]
    color = rng.randint(1, 8)
    for r, c in zip(rs, cs):
        inp[r, c] = color
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    out = inp.copy()
    out[r0:r1 + 1, c0] = color
    out[r0:r1 + 1, c1] = color
    out[r0, c0:c1 + 1] = color
    out[r1, c0:c1 + 1] = color
    return inp, out


def gen_centroid(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    inp = _empty(side)
    n_pts = rng.randint(3, 8)
    rs = [rng.randint(0, side - 1) for _ in range(n_pts)]
    cs = [rng.randint(0, side - 1) for _ in range(n_pts)]
    for r, c in zip(rs, cs):
        inp[r, c] = 3
    cr, cc = int(round(np.mean(rs))), int(round(np.mean(cs)))
    out = _empty(side)
    out[cr, cc] = 5
    return inp, out


def gen_symmetric_completion(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Half-pattern → completion via 90° symmetry."""
    inp = _empty(side)
    half = side // 2
    n_cells = rng.randint(3, 7)
    color = rng.randint(1, 8)
    for _ in range(n_cells):
        r = rng.randint(0, half - 1)
        c = rng.randint(0, half - 1)
        inp[r, c] = color
    out = inp.copy()
    # 4-fold symmetric copy
    out[:half, half:] = np.fliplr(inp[:half, :half])
    out[half:, :half] = np.flipud(inp[:half, :half])
    out[half:, half:] = np.flipud(np.fliplr(inp[:half, :half]))
    return inp, out


def gen_count_corners(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Rectangle → output is bar of length 4 (corner count)."""
    inp = _empty(side)
    r0 = rng.randint(1, side - 5)
    r1 = rng.randint(r0 + 2, min(side - 2, r0 + 6))
    c0 = rng.randint(1, side - 5)
    c1 = rng.randint(c0 + 2, min(side - 2, c0 + 6))
    color = rng.randint(2, 8)
    inp[r0, c0:c1 + 1] = color
    inp[r1, c0:c1 + 1] = color
    inp[r0:r1 + 1, c0] = color
    inp[r0:r1 + 1, c1] = color
    out = _empty(side)
    out[0, :4] = color  # bar of 4
    return inp, out


def gen_shape_perimeter(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Solid rectangle → output is bar length = perimeter cell count (mod 9 if >9)."""
    inp = _empty(side)
    h = rng.randint(2, 5)
    w = rng.randint(2, 5)
    r0 = rng.randint(0, side - h - 1)
    c0 = rng.randint(0, side - w - 1)
    color = rng.randint(2, 8)
    inp[r0:r0 + h, c0:c0 + w] = color
    perim = 2 * (h + w) - 4
    out = _empty(side)
    out[0, :min(perim, side)] = color
    return inp, out


def gen_parallel_translate(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Shape + arrow → output shape translated."""
    inp = _empty(side)
    h, w = rng.randint(2, 4), rng.randint(2, 4)
    r0 = rng.randint(1, side // 2 - h)
    c0 = rng.randint(1, side // 2 - w)
    color = rng.randint(2, 7)
    inp[r0:r0 + h, c0:c0 + w] = color
    # Translation vector: shift by (dr, dc), both in [1, side//2 - 2]
    dr = rng.randint(1, side // 2 - h - 1)
    dc = rng.randint(1, side // 2 - w - 1)
    out = _empty(side)
    out[r0 + dr:r0 + dr + h, c0 + dc:c0 + dc + w] = color
    return inp, out


ALL_GENERATORS: list[Callable] = [
    gen_draw_line, gen_mirror_h, gen_mirror_v, gen_rotate_90,
    gen_bbox, gen_centroid, gen_symmetric_completion,
    gen_count_corners, gen_shape_perimeter, gen_parallel_translate,
]


def gen_geometry_pair(side: int = 20, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    fn = rng.choice(ALL_GENERATORS)
    return fn(side, rng)
