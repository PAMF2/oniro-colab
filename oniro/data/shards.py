"""WebDataset shard utilities for video corpora.

We store all video data as sharded `.tar` files with members:
    {key}.jpg            raw frame (or .png)
    {key}.action.json    action label (optional, for OXE)
    {key}.cap.txt        caption (optional, for Ego4D narrations)
    {key}.meta.json      metadata: clip_id, frame_idx, fps, license

This module exposes a thin loader that yields (frames, actions, caption) tuples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import json

try:
    import webdataset as wds
except ImportError:
    wds = None

import torch
import numpy as np


def _decode_frame(buf: bytes) -> np.ndarray:
    from PIL import Image
    from io import BytesIO
    img = Image.open(BytesIO(buf)).convert("RGB")
    return np.array(img)


def iter_shards(
    pattern: str | Path,
    shuffle: int = 1000,
    frames_per_clip: int = 16,
) -> Iterator[dict[str, torch.Tensor | str]]:
    """Yield dicts with `frames` (T, 3, H, W) uint8 and optional `action`, `caption`."""
    if wds is None:
        raise ImportError("install webdataset: `pip install webdataset`")

    ds = wds.WebDataset(str(pattern)).shuffle(shuffle)
    buffer: list[dict] = []

    for sample in ds:
        decoded: dict[str, object] = {}
        for key, val in sample.items():
            if key.endswith(".jpg") or key.endswith(".png"):
                decoded["frame"] = _decode_frame(val)
            elif key.endswith(".action.json"):
                decoded["action"] = json.loads(val.decode())
            elif key.endswith(".cap.txt"):
                decoded["caption"] = val.decode().strip()
            elif key.endswith(".meta.json"):
                decoded["meta"] = json.loads(val.decode())

        if "frame" not in decoded:
            continue

        buffer.append(decoded)
        if len(buffer) >= frames_per_clip:
            stacked = np.stack([b["frame"] for b in buffer])
            t = torch.from_numpy(stacked).permute(0, 3, 1, 2)
            yield {
                "frames": t,
                "action": [b.get("action") for b in buffer],
                "caption": buffer[0].get("caption", ""),
                "meta": buffer[0].get("meta", {}),
            }
            buffer = []
