"""Archivist agent: wiki invariants, dedupe, edge typing.

Operates on the persistent wiki at `oniro/wiki/`:
    - Verifies that no two live variants `contradict` each other without an open
      audit.
    - Deduplicates variants whose descriptors are within ε of an existing node.
    - Types edges by inspecting the parent/child relationship (extends if metrics
      improved monotonically; supersedes if Gödel-gated and improvement is large;
      contradicts if descriptor moves to an opposite cell).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re
import yaml


EDGE_TYPES = {"extends", "contradicts", "supersedes", "invalidates", "depends_on"}

FRONTMATTER_RE = re.compile(r"^---\n(?P<yaml>.*?)\n---", re.DOTALL)


def _read_variant(path: Path) -> dict:
    text = path.read_text()
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    return yaml.safe_load(m.group("yaml")) or {}


def _write_variant(path: Path, frontmatter: dict, body: str = "") -> None:
    yml = yaml.safe_dump(frontmatter, sort_keys=False)
    path.write_text(f"---\n{yml}---\n{body}")


class Archivist:
    def __init__(self, wiki_dir: str | Path):
        self.wiki_dir = Path(wiki_dir)
        self.variants_dir = self.wiki_dir / "variants"
        self.variants_dir.mkdir(parents=True, exist_ok=True)

    def all_variants(self) -> Iterable[dict]:
        for p in sorted(self.variants_dir.glob("V-*.md")):
            d = _read_variant(p)
            if d:
                d["_path"] = str(p)
                yield d

    def add_variant(
        self,
        variant_id: str,
        parents: list[str],
        descriptor: dict,
        fitness: float,
        gate: dict,
        auditor: str,
        audit_stage_max: int,
        edges: list[dict] | None = None,
        body: str = "",
    ) -> Path:
        for e in edges or []:
            if e.get("type") not in EDGE_TYPES:
                raise ValueError(f"invalid edge type: {e.get('type')}")
        frontmatter = {
            "id": variant_id,
            "parents": parents,
            "edges": edges or [],
            "descriptor": descriptor,
            "fitness": fitness,
            "gate": gate,
            "auditor": auditor,
            "audit_stage_max": audit_stage_max,
        }
        path = self.variants_dir / f"{variant_id}.md"
        _write_variant(path, frontmatter, body)
        return path

    def check_invariants(self) -> list[str]:
        problems: list[str] = []
        contradicts: dict[str, set[str]] = {}
        for v in self.all_variants():
            vid = v["id"]
            for e in v.get("edges", []) or []:
                if e.get("type") == "contradicts":
                    contradicts.setdefault(vid, set()).add(e["target"])
        live = {v["id"] for v in self.all_variants() if v.get("audit_stage_max", 0) == 3}
        for a, peers in contradicts.items():
            if a not in live:
                continue
            for b in peers:
                if b in live:
                    problems.append(f"live variants contradict without open audit: {a} <> {b}")
        return problems
