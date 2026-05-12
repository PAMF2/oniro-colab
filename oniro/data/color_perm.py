"""Color permutation augmentation - NVARC/ARC SOTA pattern.

10! permutations possible. Standard practice: random permutation per sample,
preserving 0 (background) optionally.
"""

from __future__ import annotations

import random
import numpy as np


def random_color_perm(rng: random.Random, n_colors: int = 10,
                      keep_bg: bool = False) -> np.ndarray:
    """Returns permutation array of length n_colors."""
    if keep_bg:
        rest = list(range(1, n_colors))
        rng.shuffle(rest)
        return np.array([0] + rest, dtype=np.int64)
    p = list(range(n_colors))
    rng.shuffle(p)
    return np.array(p, dtype=np.int64)


def apply_color_perm(g: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """g: int grid, perm: length-n_colors permutation. Returns permuted grid."""
    out = perm[np.clip(g, 0, len(perm) - 1)]
    return out.astype(g.dtype, copy=False)
