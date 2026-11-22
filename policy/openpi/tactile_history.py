"""Client-side tactile history for OpenVTLA serving.

OpenVTLA is trained on tactile clips: for every sample the loader reads each
side's frames at offsets [-6, -4, -2, 0] and repeats the first frame of the
episode where the history would start before it. The policy server is queried
only once per action chunk, so it never sees the frames in between; the clip
therefore has to be assembled here, from the observation of every step.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

LEFT_IMAGE_KEY = "observation/left_tactile_image"
RIGHT_IMAGE_KEY = "observation/right_tactile_image"
LEFT_CLIP_KEY = "observation/left_tactile_clip"
RIGHT_CLIP_KEY = "observation/right_tactile_clip"


class TactileHistory:
    """Keeps the recent tactile frames of one episode and builds clips from them.

    Call `push(obs)` once per policy step, with the single-frame observation.
    It returns a copy in which the two single-frame tactile keys are replaced by
    clips of shape [len(offsets), H, W, 3], uint8, oldest frame first.
    Call `reset()` at the start of every episode.
    """

    def __init__(self, offsets: Sequence[int] = (-6, -4, -2, 0)):
        offsets = tuple(int(o) for o in offsets)
        if not offsets or any(o > 0 for o in offsets) or list(offsets) != sorted(offsets) or offsets[-1] != 0:
            raise ValueError(f"offsets must be non-positive, ascending and end with 0, got {offsets}")
        self.offsets = offsets
        self._frames: dict[str, deque[np.ndarray]] = {
            key: deque(maxlen=1 - offsets[0]) for key in (LEFT_IMAGE_KEY, RIGHT_IMAGE_KEY)
        }

    def reset(self) -> None:
        for frames in self._frames.values():
            frames.clear()

    def _clip(self, key: str) -> np.ndarray:
        frames = self._frames[key]
        last = len(frames) - 1
        # Index 0 is the oldest frame kept. Before the episode has enough frames the
        # index is clamped to it, which repeats the first frame like the training loader.
        return np.ascontiguousarray(np.stack([frames[max(last + offset, 0)] for offset in self.offsets]))

    def push(self, obs: dict) -> dict:
        missing = [key for key in self._frames if key not in obs]
        if missing:
            raise KeyError(f"Tactile history needs single-frame tactile images, missing: {missing}")
        out = dict(obs)
        for key, clip_key in ((LEFT_IMAGE_KEY, LEFT_CLIP_KEY), (RIGHT_IMAGE_KEY, RIGHT_CLIP_KEY)):
            frame = np.asarray(out.pop(key))
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"{key} must be uint8 [H, W, 3], got {frame.dtype} {frame.shape}")
            self._frames[key].append(frame.copy())
            out[clip_key] = self._clip(key)
        return out
