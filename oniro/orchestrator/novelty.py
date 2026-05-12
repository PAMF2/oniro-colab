"""Novelty bonus via k-NN distance in descriptor space.

novelty(v) = mean L2 distance from v.descriptor to its k nearest neighbors in the
descriptor cloud of all canonical + provisional variants.
"""

from __future__ import annotations

import math
from oniro.orchestrator.qd_archive import QDArchive, Descriptor


def _desc_vec(d: Descriptor) -> tuple[float, float, float]:
    return (d.slot_purity, d.jepa_loss, d.action_acc)


def _l2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def novelty(archive: QDArchive, descriptor: Descriptor, k: int = 5) -> float:
    cloud = [
        _desc_vec(v.descriptor)
        for v in list(archive.all_canonical()) + list(archive.all_provisional())
    ]
    if not cloud:
        return 1.0
    target = _desc_vec(descriptor)
    dists = sorted(_l2(target, c) for c in cloud)
    take = dists[: min(k, len(dists))]
    return sum(take) / len(take)
