"""Frames decoded from a directory of still images."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .base import Frame, FrameSource, import_cv2, to_rgb_uint8

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageDirSource(FrameSource):
    def load(self) -> List[Frame]:
        root = Path(self.cfg.path).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"source.path is not a directory: {root}")
        paths = sorted(p for p in root.iterdir() if p.suffix.lower() in _EXTS)
        if not paths:
            raise FileNotFoundError(f"no images with {sorted(_EXTS)} under {root}")
        if self.cfg.count > 0:
            paths = paths[: self.cfg.count]

        cv2 = import_cv2()
        frames = []
        for i, path in enumerate(paths):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to decode image: {path}")
            rgb = to_rgb_uint8(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if self.cfg.resize:
                rgb = _resize(cv2, rgb, self.cfg.width, self.cfg.height)
            frames.append(Frame(index=i, image=rgb, timestamp_ms=i * 33))
        return frames


def _resize(cv2, image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
