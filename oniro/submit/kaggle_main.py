"""Kaggle offline submission entrypoint.

ARC-AGI-3 Kaggle constraints:
    - No internet during scoring
    - ~$50 compute cap (≈ 9h L4 / 3h A100)
    - All inference + online adaptation must be self-contained

This script:
    1. Loads the int8-quantized ONIRO checkpoint from the Kaggle inputs dir.
    2. For each game in the eval set, runs `run_arc3_episode` with a smaller
       MPC horizon (H=6) and the Gödel gate using a *local* rolling OOD buffer.
    3. Writes per-game scores to `/kaggle/working/submission.csv`.

The orchestrator (executor / reviewer) is NOT loaded; this is inference-only.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch

from oniro.models.oniro import Oniro, OniroConfig
from oniro.planner.mpc import MPCPlanner, MPCConfig
from oniro.orchestrator.godel_gate import GodelGate
from oniro.eval.ood_splits import OODBuffer
from oniro.eval.arc3_runner import run_arc3_episode
from oniro.data.arc3_env import ARC3Env


CHECKPOINT_PATH = os.environ.get(
    "ONIRO_CKPT", "/kaggle/input/oniro-weights/oniro_int8.pt"
)
SUBMISSION_PATH = Path("/kaggle/working/submission.csv")


def load_model(device: str = "cuda") -> Oniro:
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    cfg = OniroConfig(**state.get("cfg", {}))
    model = Oniro(cfg).to(device).eval()
    model.load_state_dict(state["model"], strict=False)
    return model


def list_games() -> list[str]:
    env_root = Path(os.environ.get("ARC_GAMES_DIR", "/kaggle/input/arc-agi-3/games"))
    if not env_root.exists():
        return ["ls20"]
    return sorted(p.name for p in env_root.iterdir() if p.is_dir())


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    planner = MPCPlanner(model, MPCConfig(horizon=6, branching=8, temperature=0.7))
    gate = GodelGate(n_splits=10, min_splits_improved=7, sigma_multiplier=0.5)
    buf = OODBuffer(capacity_per_split=16, n_splits=10)

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSION_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "score"])

        for game_id in list_games():
            env = ARC3Env(game_id, backend="kaggle")
            try:
                result = run_arc3_episode(
                    env, model, planner, gate, buf, device=device,
                )
                total = sum(result.per_level.values()) / max(len(result.per_level), 1)
            except Exception as e:
                total = 0.0
            w.writerow([game_id, f"{total:.4f}"])


if __name__ == "__main__":
    main()
