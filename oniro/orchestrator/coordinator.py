"""Project coordinator (AI Co-Mathematician arXiv:2605.06651 §3).

Schedules parallel workstreams, allocates GPU budget, kills stalled runs, surfaces
explicit asks to the user when a workstream blocks. Workstreams live under
`oniro/workspace/<name>/` with three Markdown files:
    STATE.md  — live progress, updated by the running stream
    PLAN.md   — rolling local plan
    LEDGER.md — local failures (synced to global wiki/failures.jsonl)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json
import time


StreamState = Literal["idle", "running", "blocked", "done", "failed"]


@dataclass
class Workstream:
    name: str
    state: StreamState = "idle"
    started_at: float = 0.0
    heartbeat: float = 0.0
    blocked_reason: str = ""
    gpu_budget_hours: float = 24.0
    gpu_used_hours: float = 0.0


@dataclass
class CoordinatorConfig:
    stall_timeout_seconds: float = 60 * 60 * 6   # 6h heartbeat → blocked
    kill_timeout_seconds: float = 60 * 60 * 24   # 24h blocked → kill
    total_gpu_budget_hours: float = 24 * 14      # two weeks of GPU-hours


class Coordinator:
    def __init__(self, workspace_root: str | Path, cfg: CoordinatorConfig | None = None):
        self.root = Path(workspace_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or CoordinatorConfig()
        self.streams: dict[str, Workstream] = {}

    def register(self, name: str, gpu_budget_hours: float = 24.0) -> Workstream:
        if name in self.streams:
            return self.streams[name]
        sd = self.root / name
        sd.mkdir(parents=True, exist_ok=True)
        for f in ("STATE.md", "PLAN.md", "LEDGER.md"):
            p = sd / f
            if not p.exists():
                p.write_text(f"# {name} :: {f}\n")
        ws = Workstream(name=name, gpu_budget_hours=gpu_budget_hours)
        self.streams[name] = ws
        return ws

    def start(self, name: str) -> None:
        ws = self.streams[name]
        ws.state = "running"
        ws.started_at = time.time()
        ws.heartbeat = ws.started_at
        self._write_state(name, f"started at {ws.started_at}")

    def heartbeat(self, name: str, note: str = "") -> None:
        ws = self.streams[name]
        ws.heartbeat = time.time()
        if note:
            self._write_state(name, note)

    def block(self, name: str, reason: str) -> None:
        ws = self.streams[name]
        ws.state = "blocked"
        ws.blocked_reason = reason
        self._write_state(name, f"BLOCKED: {reason}")

    def complete(self, name: str) -> None:
        ws = self.streams[name]
        ws.state = "done"
        self._write_state(name, "DONE")

    def fail(self, name: str, reason: str) -> None:
        ws = self.streams[name]
        ws.state = "failed"
        self._write_state(name, f"FAILED: {reason}")

    def tick(self) -> list[str]:
        """Returns list of stream names that need user attention."""
        now = time.time()
        attention: list[str] = []
        for name, ws in self.streams.items():
            if ws.state == "running" and now - ws.heartbeat > self.cfg.stall_timeout_seconds:
                self.block(name, f"no heartbeat for {now - ws.heartbeat:.0f}s")
                attention.append(name)
            if ws.state == "blocked" and now - ws.heartbeat > self.cfg.kill_timeout_seconds:
                self.fail(name, "blocked too long")
                attention.append(name)
        return attention

    def _write_state(self, name: str, note: str) -> None:
        path = self.root / name / "STATE.md"
        with path.open("a") as f:
            f.write(f"- {time.strftime('%Y-%m-%dT%H:%M:%S')}  {note}\n")
