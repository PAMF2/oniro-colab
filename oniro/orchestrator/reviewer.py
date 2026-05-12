"""Reviewer agent: 3-stage claim audit (ARIS arXiv:2605.03042 §3.1).

Stage 1: experiment-integrity audit — convergence, seeds, leakage.
Stage 2: result-to-claim mapping — supported / partially / invalidated per claim.
Stage 3: paper-claim audit — fresh zero-context reviewer re-reads wiki narrative.

The reviewer rotates across model families (Anthropic / OpenAI / Google) to break
collusion. Two-of-three agreement required for canonical merge of any mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
import json


Verdict = Literal["supported", "partially_supported", "invalidated"]


@dataclass
class StageReport:
    stage: int
    verdict: Verdict
    items: list[str] = field(default_factory=list)
    auditor: str = "unknown"


@dataclass
class ReviewBundle:
    stage1: StageReport
    stage2: StageReport
    stage3: StageReport

    def passed(self) -> bool:
        return all(s.verdict == "supported" for s in (self.stage1, self.stage2, self.stage3))

    def max_stage_passed(self) -> int:
        for n, s in enumerate((self.stage1, self.stage2, self.stage3), start=1):
            if s.verdict != "supported":
                return n - 1
        return 3


class ReviewerProtocol(Protocol):
    def review(self, run_dir: Path, claim: str) -> ReviewBundle: ...


class DryRunReviewer:
    """Canned reviewer that always passes; used to exercise the loop."""

    def review(self, run_dir: Path, claim: str) -> ReviewBundle:
        return ReviewBundle(
            stage1=StageReport(1, "supported", auditor="dry-run"),
            stage2=StageReport(2, "supported", auditor="dry-run"),
            stage3=StageReport(3, "supported", auditor="dry-run"),
        )


class RotatingReviewer:
    """Rotates Anthropic / OpenAI / Google reviewers per stage.

    Requires API keys in env: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY.
    """

    def __init__(self) -> None:
        self._anthropic = None
        self._openai = None
        self._gemini = None

    def _call_anthropic(self, system: str, user: str) -> str:
        if self._anthropic is None:
            from anthropic import Anthropic
            self._anthropic = Anthropic()
        msg = self._anthropic.messages.create(
            model="claude-opus-4-7", max_tokens=4096,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    def _call_openai(self, system: str, user: str) -> str:
        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI()
        msg = self._openai.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return msg.choices[0].message.content or ""

    def _call_gemini(self, system: str, user: str) -> str:
        if self._gemini is None:
            from google import genai
            self._gemini = genai.Client()
        resp = self._gemini.models.generate_content(
            model="gemini-3.1-pro",
            contents=[{"role": "user", "parts": [{"text": system + "\n\n" + user}]}],
        )
        return resp.text or ""

    def _parse(self, text: str, stage: int, auditor: str) -> StageReport:
        try:
            data = json.loads(text)
            return StageReport(
                stage=stage, verdict=data["verdict"],
                items=data.get("items", []), auditor=auditor,
            )
        except Exception:
            return StageReport(stage=stage, verdict="invalidated", items=[text[:200]], auditor=auditor)

    def review(self, run_dir: Path, claim: str) -> ReviewBundle:
        skills_dir = run_dir.parent / "skills"
        s1_prompt = (skills_dir / "audit-experiment-integrity" / "SKILL.md").read_text()
        s2_prompt = (skills_dir / "audit-result-to-claim" / "SKILL.md").read_text()
        s3_prompt = (skills_dir / "audit-paper-claim" / "SKILL.md").read_text()

        ctx = json.dumps({"run_dir": str(run_dir), "claim": claim})

        s1 = self._parse(self._call_anthropic(s1_prompt, ctx), 1, "claude-opus-4-7")
        s2 = self._parse(self._call_openai(s2_prompt, ctx), 2, "gpt-5.4")
        s3 = self._parse(self._call_gemini(s3_prompt, ctx), 3, "gemini-3.1-pro")
        return ReviewBundle(s1, s2, s3)
