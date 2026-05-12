"""Open-X-Embodiment loader (arXiv:2310.08864).

Only the Apache-licensed subset is used to keep release rights clean. Continuous
actions are bucketed into 8 VQ bins per dimension (offline-fitted) and stored as
discrete indices in the shard `action.json` field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

from oniro.data.shards import iter_shards


def openx_iter(
    shard_root: str | Path,
    frames_per_clip: int = 32,
    shuffle: int = 500,
) -> Iterator[dict[str, torch.Tensor]]:
    pattern = str(Path(shard_root) / "apache_subset" / "*.tar")
    for sample in iter_shards(pattern, shuffle=shuffle, frames_per_clip=frames_per_clip):
        actions = sample.pop("action") or []
        a_disc = torch.tensor(
            [int(a.get("discrete", 0)) if a else 0 for a in actions], dtype=torch.long
        )
        a_click = torch.tensor(
            [
                a.get("click", [0.0, 0.0]) if a else [0.0, 0.0]
                for a in actions
            ],
            dtype=torch.float32,
        )
        sample["frames"] = sample["frames"].float() / 255.0
        sample["action_disc"] = a_disc
        sample["action_click"] = a_click
        yield sample
