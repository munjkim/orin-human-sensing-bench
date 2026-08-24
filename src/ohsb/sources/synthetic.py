"""Deterministic synthetic frames.

Useful for smoke-testing the harness on a dev machine and for isolating
inference cost from decode cost. Detection rates on these frames are
meaningless — never quote accuracy from a synthetic run.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import Frame, FrameSource


class SyntheticSource(FrameSource):
    def load(self) -> List[Frame]:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        h, w = int(cfg.height), int(cfg.width)
        frames = []
        for i in range(int(cfg.count)):
            # Smooth gradient + light noise: compresses and preprocesses like a
            # real frame without pinning the CPU on random number generation.
            base = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
            img = np.repeat(np.repeat(base, h, axis=0), 3, axis=2)
            img = (img + rng.integers(0, 24, size=(h, w, 3))) % 256
            frames.append(Frame(index=i, image=img.astype(np.uint8), timestamp_ms=i * 33))
        return frames
