"""Ego4D loader wrapper.

Ego4D is research-licensed; checkpoints trained with it stay internal until license
review. This loader assumes a pre-processed shard tree at `data/shards/ego4d/*.tar`,
4 fps, 256x256, frame jpegs + narration text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

from oniro.data.shards import iter_shards


def ego4d_iter(
    shard_root: str | Path,
    frames_per_clip: int = 16,
    shuffle: int = 1000,
) -> Iterator[dict[str, torch.Tensor | str]]:
    pattern = str(Path(shard_root) / "*.tar")
    for sample in iter_shards(pattern, shuffle=shuffle, frames_per_clip=frames_per_clip):
        sample["frames"] = sample["frames"].float() / 255.0
        yield sample
