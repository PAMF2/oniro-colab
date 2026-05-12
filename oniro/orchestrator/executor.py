"""Executor agent.

Reads the current wiki frontier + the failure ledger, then emits ONE typed mutation
as a unified diff against the `oniro/` package. Backed by Claude Opus 4.7 (1M ctx)
via the Anthropic API; for offline / smoke testing a `DryRunExecutor` returns a
canned diff so the rest of the loop can be exercised without API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json
import textwrap


MUTATION_OPS = [
    "loss-reweight",
    "new-loss-term",
    "layer-swap",
    "data-mix-shift",
    "hyperparam-jitter",
    "lora-rank-change",
    "slot-count-change",
]


@dataclass
class Mutation:
    op: str
    diff: str
    rationale: str
    target_files: list[str]
    closed_branches_consulted: list[str]


class ExecutorProtocol(Protocol):
    def propose(self, wiki_dir: Path, failures_path: Path) -> Mutation: ...


def _read_failures(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _frontier_summary(wiki_dir: Path, n: int = 3) -> str:
    variants_dir = wiki_dir / "variants"
    if not variants_dir.exists():
        return "<no variants>"
    files = sorted(variants_dir.glob("V-*.md"))[-n:]
    return "\n\n---\n\n".join(f.read_text() for f in files) or "<empty>"


class DryRunExecutor:
    """Offline canned executor: returns a no-op-style mutation for smoke tests."""

    def propose(self, wiki_dir: Path, failures_path: Path) -> Mutation:
        failures = _read_failures(failures_path)
        closed = [b for f in failures for b in f.get("closed_branches", [])]
        return Mutation(
            op="hyperparam-jitter",
            diff=textwrap.dedent(
                """
                # nominal no-op canned diff for DryRunExecutor
                """
            ).strip(),
            rationale="dry-run; loop smoke test",
            target_files=["configs/train/phase1_pretrain.yaml"],
            closed_branches_consulted=closed,
        )


class ClaudeExecutor:
    """Anthropic Claude-backed executor.

    Requires `ANTHROPIC_API_KEY` in env. Returns a Mutation parsed from a strict
    JSON response. The system prompt enforces the ARIS skill `propose-mutation`.
    """

    def __init__(self, model: str = "claude-opus-4-7", max_tokens: int = 8192):
        try:
            from anthropic import Anthropic
            self._client = Anthropic()
        except Exception as e:
            raise RuntimeError(
                "anthropic SDK unavailable; pip install anthropic and set ANTHROPIC_API_KEY"
            ) from e
        self.model = model
        self.max_tokens = max_tokens

    def propose(self, wiki_dir: Path, failures_path: Path) -> Mutation:
        skill_md = (wiki_dir.parent / "skills" / "propose-mutation" / "SKILL.md").read_text()
        frontier = _frontier_summary(wiki_dir)
        failures = _read_failures(failures_path)
        closed = [b for f in failures for b in f.get("closed_branches", [])]

        system = skill_md
        user = json.dumps(
            {"frontier": frontier, "closed_branches": closed, "ops": MUTATION_OPS}
        )

        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = msg.content[0].text
        parsed = json.loads(text)
        return Mutation(
            op=parsed["op"],
            diff=parsed["diff"],
            rationale=parsed["rationale"],
            target_files=parsed.get("target_files", []),
            closed_branches_consulted=closed,
        )
