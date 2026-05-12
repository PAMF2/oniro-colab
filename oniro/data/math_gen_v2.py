"""Expanded procedural math grid generators (v2).

20+ visual math tasks rendered as int grids. All tasks fit in side×side,
values in [0, 9]. Designed for joint training with ARC and Sudoku - shares
the cell-as-token representation.

Tasks:
    arithmetic: add, sub, mul, div, mod, double, halve
    sequences:  arith_seq, fib, prime_mark
    comparison: max, min, equal, sort
    spatial:    gravity_down, mirror_h, mirror_v, rotate90
    counting:   count_colored, histogram, parity_row
    constraint: latin_row (mini-Latin), arithmetic_chain
"""

from __future__ import annotations

import random
import numpy as np


def _empty(side: int) -> np.ndarray:
    return np.zeros((side, side), dtype=np.int8)


# ============ ARITHMETIC ============

def gen_add(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    a = rng.randint(1, side // 2 - 1)
    b = rng.randint(1, side // 2 - 1)
    inp, out = _empty(side), _empty(side)
    inp[2, :a] = 3
    inp[5, :b] = 5
    out[3, :min(a + b, side)] = 4
    return inp, out


def gen_sub(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    a = rng.randint(2, side - 2)
    b = rng.randint(1, a - 1)
    inp, out = _empty(side), _empty(side)
    inp[2, :a] = 3
    inp[5, :b] = 5
    out[3, :a - b] = 4
    return inp, out


def gen_mul(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    a = rng.randint(2, min(6, side - 1))
    b = rng.randint(2, min(6, side - 1))
    inp, out = _empty(side), _empty(side)
    inp[2, :a] = 3
    inp[5, :b] = 5
    out[:a, :b] = 4
    return inp, out


def gen_div(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    b = rng.randint(2, 4)
    q = rng.randint(2, min(5, side - 2))
    a = b * q
    inp, out = _empty(side), _empty(side)
    inp[2, :a] = 3
    inp[5, :b] = 5
    out[3, :q] = 4
    return inp, out


def gen_mod(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    b = rng.randint(2, 5)
    q = rng.randint(1, 3)
    r = rng.randint(0, b - 1)
    a = b * q + r
    if a >= side - 1:
        a = min(a, side - 2)
    inp, out = _empty(side), _empty(side)
    inp[2, :a] = 3
    inp[5, :b] = 5
    if r > 0:
        out[3, :r] = 4
    return inp, out


def gen_double(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    n = rng.randint(1, side // 2 - 1)
    inp, out = _empty(side), _empty(side)
    inp[3, :n] = 5
    out[3, :2 * n] = 5
    return inp, out


def gen_halve(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    n = rng.randint(2, side - 2)
    if n % 2: n += 1
    inp, out = _empty(side), _empty(side)
    inp[3, :n] = 5
    out[3, :n // 2] = 5
    return inp, out


# ============ SEQUENCES ============

def gen_arith_seq(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    start = rng.randint(1, 3)
    step = rng.randint(1, 2)
    seq = [start + i * step for i in range(min(6, side - 2))]
    inp, out = _empty(side), _empty(side)
    hide = rng.randint(0, len(seq) - 1)
    for i, v in enumerate(seq):
        c = min(v, 9)
        if i == hide:
            inp[3, i] = 9  # missing marker
        else:
            inp[3, i] = c
        out[3, i] = c
    return inp, out


def gen_fib(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    a, b = 1, 1
    seq = [a, b]
    while len(seq) < min(7, side - 1) and b < 9:
        a, b = b, a + b
        if b > 9: break
        seq.append(b)
    inp, out = _empty(side), _empty(side)
    hide = rng.randint(0, len(seq) - 1)
    for i, v in enumerate(seq):
        if i == hide: inp[3, i] = 9
        else: inp[3, i] = v
        out[3, i] = v
    return inp, out


def gen_prime_mark(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Mark primes in a sequence."""
    n = min(side - 2, 9)
    primes = {2, 3, 5, 7}
    inp, out = _empty(side), _empty(side)
    for i in range(1, n + 1):
        inp[3, i - 1] = i
        out[3, i - 1] = 4 if i in primes else 1
    return inp, out


# ============ COMPARISON ============

def gen_max(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    a = rng.randint(2, side - 2)
    b = rng.randint(2, side - 2)
    while a == b: b = rng.randint(2, side - 2)
    inp, out = _empty(side), _empty(side)
    inp[2, :a] = 3
    inp[5, :b] = 5
    longer_row, longer_color, longer_len = (2, 3, a) if a > b else (5, 5, b)
    out[longer_row, :longer_len] = longer_color
    return inp, out


def gen_min(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    a = rng.randint(2, side - 2)
    b = rng.randint(2, side - 2)
    while a == b: b = rng.randint(2, side - 2)
    inp, out = _empty(side), _empty(side)
    inp[2, :a] = 3
    inp[5, :b] = 5
    shorter_row, shorter_color, shorter_len = (2, 3, a) if a < b else (5, 5, b)
    out[shorter_row, :shorter_len] = shorter_color
    return inp, out


def gen_equal_check(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    a = rng.randint(2, side - 2)
    equal = rng.random() < 0.5
    b = a if equal else (a + rng.choice([-1, 1]))
    b = max(1, min(side - 2, b))
    inp, out = _empty(side), _empty(side)
    inp[2, :a] = 3
    inp[5, :b] = 5
    out[3, 0] = 4 if a == b else 2
    return inp, out


def gen_sort_rows(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """3 rows of differing lengths -> sort by length ascending."""
    lens = rng.sample(range(2, side - 2), 3)
    rows_in = [(0, lens[0], 3), (3, lens[1], 5), (6, lens[2], 7)]
    inp = _empty(side); out = _empty(side)
    for r, ln, c in rows_in:
        inp[r, :ln] = c
    # sort by len ascending
    rows_in.sort(key=lambda t: t[1])
    for i, (_, ln, c) in enumerate(rows_in):
        out[i * 3, :ln] = c
    return inp, out


# ============ SPATIAL ============

def gen_gravity(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Random scattered cells fall to bottom row by row in each column."""
    inp = _empty(side)
    n_filled = rng.randint(side // 3, side)
    positions = rng.sample(range(side * side), k=min(n_filled, side * side))
    for p in positions[:n_filled]:
        r, c = p // side, p % side
        inp[r, c] = rng.randint(1, 6)
    out = _empty(side)
    for c in range(side):
        col = inp[:, c]
        nonzero = col[col != 0]
        if len(nonzero):
            out[side - len(nonzero):, c] = nonzero
    return inp, out


def gen_mirror_h(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    inp = _empty(side)
    for _ in range(rng.randint(3, 8)):
        r = rng.randint(0, side - 1); c = rng.randint(0, side // 2 - 1)
        inp[r, c] = rng.randint(1, 9)
    out = np.flip(inp, axis=1).copy()
    return inp, out


def gen_rotate(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    inp = _empty(side)
    for _ in range(rng.randint(4, 10)):
        r = rng.randint(0, side - 1); c = rng.randint(0, side - 1)
        inp[r, c] = rng.randint(1, 9)
    out = np.rot90(inp, k=rng.randint(1, 3)).copy()
    return inp, out


# ============ COUNTING ============

def gen_count_colored(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Count non-zero cells, output bar of length count in row 0."""
    inp = _empty(side)
    n = rng.randint(1, min(side - 1, 9))
    cells = rng.sample(range(side * side), k=n)
    color = rng.randint(1, 8)
    for p in cells:
        inp[p // side, p % side] = color
    out = _empty(side)
    out[0, :n] = color
    return inp, out


def gen_histogram(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """3 colors scattered, output histogram with column heights = counts."""
    inp = _empty(side)
    counts = [rng.randint(1, min(side - 1, 5)) for _ in range(3)]
    colors = rng.sample(range(1, 9), 3)
    cells = list(range(side * side)); rng.shuffle(cells)
    idx = 0
    for ct, col in zip(counts, colors):
        for _ in range(ct):
            if idx >= len(cells): break
            p = cells[idx]; idx += 1
            inp[p // side, p % side] = col
    out = _empty(side)
    for i, (ct, col) in enumerate(zip(counts, colors)):
        for r in range(side - ct, side):
            out[r, i * 2] = col
    return inp, out


def gen_parity_row(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Row of dots -> output marks parity (even=red, odd=green)."""
    inp = _empty(side); out = _empty(side)
    n_rows = rng.randint(3, min(6, side - 1))
    for r in range(n_rows):
        ln = rng.randint(1, side - 2)
        inp[r, :ln] = 5
        out[r, 0] = 2 if ln % 2 else 3   # red=odd, green=even
    return inp, out


# ============ CONSTRAINT ============

def gen_arith_chain(side: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Chain a -> a+k -> a+2k, predict third term."""
    a = rng.randint(0, 3)
    k = rng.randint(1, 3)
    inp = _empty(side); out = _empty(side)
    inp[3, 0] = a
    inp[3, 2] = a + k
    inp[3, 4] = 9  # marker for unknown
    out[3, 0] = a
    out[3, 2] = a + k
    out[3, 4] = min(a + 2 * k, 9)
    return inp, out


# ============ DISPATCH ============

ALL_GENERATORS = [
    gen_add, gen_sub, gen_mul, gen_div, gen_mod, gen_double, gen_halve,
    gen_arith_seq, gen_fib, gen_prime_mark,
    gen_max, gen_min, gen_equal_check, gen_sort_rows,
    gen_gravity, gen_mirror_h, gen_rotate,
    gen_count_colored, gen_histogram, gen_parity_row,
    gen_arith_chain,
]


def gen_math_pair_v2(side: int = 16, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    fn = rng.choice(ALL_GENERATORS)
    return fn(side, rng)
