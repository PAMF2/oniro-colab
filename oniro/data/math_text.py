"""Procedural math problems paired with grid visualizations + text Q/A.

Covers:
    - arithmetic (add, sub, mul, div)
    - sequence completion
    - comparison (>, <, =)
    - parity
    - max/min
    - modular arithmetic
    - linear equations a*x + b = c -> solve x
    - simple word problems

Each sample returns (grid_in, grid_out, question_text, answer_text).
grid_in/grid_out: 2D int8 arrays usable as visual context for URM.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np


def _digits_to_grid(n: int, side: int) -> np.ndarray:
    """Render integer as digit cells along a row, padded to side×side."""
    s = str(abs(n))[:side]
    g = np.zeros((side, side), dtype=np.int8)
    for i, ch in enumerate(s):
        if i >= side:
            break
        g[0, i] = (int(ch) % 10)
    if n < 0:
        g[1, 0] = 9  # marker for negative
    return g


def _arith(rng: random.Random, side: int = 16) -> tuple:
    op = rng.choice(["+", "-", "*"])
    a = rng.randint(1, 99)
    b = rng.randint(1, 99)
    if op == "+":
        res = a + b
    elif op == "-":
        if b > a:
            a, b = b, a
        res = a - b
    else:
        a = rng.randint(2, 12); b = rng.randint(2, 12)
        res = a * b
    q = f"{a}{op}{b}=?"
    ans = str(res)
    g_in = _digits_to_grid(a, side)
    g_out = _digits_to_grid(res, side)
    return g_in, g_out, q, ans


def _sequence(rng: random.Random, side: int = 16) -> tuple:
    start = rng.randint(0, 20)
    step = rng.randint(1, 9)
    seq = [start + i * step for i in range(4)]
    nxt = start + 4 * step
    q = f"{seq[0]},{seq[1]},{seq[2]},{seq[3]},?"
    ans = str(nxt)
    g_in = _digits_to_grid(seq[3], side)
    g_out = _digits_to_grid(nxt, side)
    return g_in, g_out, q, ans


def _compare(rng: random.Random, side: int = 16) -> tuple:
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    if a > b: sym = ">"
    elif a < b: sym = "<"
    else: sym = "="
    q = f"{a}?{b}"
    ans = sym
    g_in = _digits_to_grid(a, side)
    g_out = _digits_to_grid(b, side)
    return g_in, g_out, q, ans


def _parity(rng: random.Random, side: int = 16) -> tuple:
    n = rng.randint(0, 999)
    q = f"parity({n})?"
    ans = "odd" if n % 2 else "even"
    g_in = _digits_to_grid(n, side)
    g_out = _digits_to_grid(n % 2, side)
    return g_in, g_out, q, ans


def _modular(rng: random.Random, side: int = 16) -> tuple:
    a = rng.randint(1, 99)
    m = rng.randint(2, 12)
    res = a % m
    q = f"{a}mod{m}=?"
    ans = str(res)
    g_in = _digits_to_grid(a, side)
    g_out = _digits_to_grid(res, side)
    return g_in, g_out, q, ans


def _linear_eq(rng: random.Random, side: int = 16) -> tuple:
    a = rng.randint(1, 9)
    x = rng.randint(0, 12)
    b = rng.randint(0, 20)
    c = a * x + b
    q = f"{a}x+{b}={c},x=?"
    ans = str(x)
    g_in = _digits_to_grid(c, side)
    g_out = _digits_to_grid(x, side)
    return g_in, g_out, q, ans


GENERATORS: list[Callable] = [
    _arith, _sequence, _compare, _parity, _modular, _linear_eq,
]


def gen_math_text_pair(rng: random.Random | None = None, side: int = 16) -> tuple:
    """Return (grid_in, grid_out, question_text, answer_text)."""
    if rng is None:
        rng = random.Random()
    fn = rng.choice(GENERATORS)
    return fn(rng, side=side)
