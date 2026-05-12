"""MAP-Elites archive (Quality-Diversity).

Variants are placed into cells of a 3D descriptor grid:
    (slot_purity_bucket, jepa_loss_bucket, action_acc_bucket)

Each cell holds at most one "elite" — the highest-fitness variant whose descriptor
falls in that cell. Plus a "provisional" tier: variants that failed the Gödel gate
but landed in an empty cell with novelty > 0.7. Provisionals cannot supersede
parents but may seed mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator
import json


def _bucket(value: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


@dataclass
class Descriptor:
    slot_purity: float
    jepa_loss: float
    action_acc: float

    def cell(
        self,
        purity_edges: list[float],
        jepa_edges: list[float],
        action_edges: list[float],
    ) -> tuple[int, int, int]:
        return (
            _bucket(self.slot_purity, purity_edges),
            _bucket(self.jepa_loss, jepa_edges),
            _bucket(self.action_acc, action_edges),
        )


@dataclass
class Variant:
    id: str
    parents: list[str]
    descriptor: Descriptor
    fitness: float
    gate_passed: bool
    audit_stage_passed: int
    checkpoint_path: str
    tier: str = "canonical"  # canonical | provisional
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class QDArchiveConfig:
    purity_edges: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.85])
    jepa_edges: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    action_edges: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.85])


class QDArchive:
    def __init__(self, root: str | Path, cfg: QDArchiveConfig | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or QDArchiveConfig()
        self._cells: dict[tuple[int, int, int], Variant] = {}
        self._provisional: dict[tuple[int, int, int], Variant] = {}
        self.index_path = self.root / "lineage.jsonl"

    def cell_of(self, descriptor: Descriptor) -> tuple[int, int, int]:
        return descriptor.cell(
            self.cfg.purity_edges, self.cfg.jepa_edges, self.cfg.action_edges,
        )

    def insert(self, v: Variant) -> tuple[bool, str]:
        """Insert if cell empty or new fitness > elite fitness. Returns (placed, reason)."""
        cell = self.cell_of(v.descriptor)
        if v.tier == "provisional":
            if cell not in self._cells and cell not in self._provisional:
                self._provisional[cell] = v
                self._log(v, cell, "placed_provisional")
                return True, "placed_provisional"
            return False, "cell_occupied"

        elite = self._cells.get(cell)
        if elite is None or v.fitness > elite.fitness:
            self._cells[cell] = v
            self._provisional.pop(cell, None)
            self._log(v, cell, "placed_canonical")
            return True, "placed_canonical"
        return False, "lower_fitness_than_elite"

    def _log(self, v: Variant, cell: tuple[int, int, int], event: str) -> None:
        line = {"event": event, "cell": list(cell), "variant": v.to_dict()}
        with self.index_path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def all_canonical(self) -> Iterator[Variant]:
        return iter(self._cells.values())

    def all_provisional(self) -> Iterator[Variant]:
        return iter(self._provisional.values())

    def pareto_front(self) -> list[Variant]:
        items = list(self._cells.values())
        front = []
        for v in items:
            dominated = False
            for u in items:
                if u is v:
                    continue
                if (
                    u.fitness >= v.fitness
                    and u.descriptor.slot_purity >= v.descriptor.slot_purity
                    and u.descriptor.action_acc >= v.descriptor.action_acc
                    and (
                        u.fitness > v.fitness
                        or u.descriptor.slot_purity > v.descriptor.slot_purity
                        or u.descriptor.action_acc > v.descriptor.action_acc
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(v)
        return front

    def sample_parent(self, rng) -> Variant | None:
        items = list(self._cells.values())
        if not items:
            return None
        weights = [max(v.fitness, 1e-3) for v in items]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for v, w in zip(items, weights):
            acc += w
            if acc >= r:
                return v
        return items[-1]
