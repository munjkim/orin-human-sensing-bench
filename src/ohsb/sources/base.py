"""Frame source protocol.

A source yields decoded RGB frames. Decoding is deliberately done *before*
the timed section (see :mod:`ohsb.runner`) so that measured latency is
inference latency, not I/O.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class Frame:
    index: int
    image: np.ndarray  # uint8, HxWx3, RGB
    timestamp_ms: int = 0

    @property
    def shape(self):
        return self.image.shape


class FrameSource(abc.ABC):
    """Loads a fixed set of frames into memory once, then replays them."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._frames: List[Frame] = []

    @abc.abstractmethod
    def load(self) -> List[Frame]:
        """Decode and return the frame set. Called once, outside timing."""

    def prepare(self) -> FrameSource:
        if not self._frames:
            self._frames = self.load()
        if not self._frames:
            raise RuntimeError(f"source {type(self).__name__} produced no frames")
        return self

    @property
    def frames(self) -> List[Frame]:
        return self._frames

    def __len__(self) -> int:
        return len(self._frames)

    def describe(self) -> Dict[str, Any]:
        h, w = (self._frames[0].image.shape[:2] if self._frames else (0, 0))
        return {"type": self.cfg.type, "frames": len(self._frames), "width": w, "height": h}


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Normalise an arbitrary decoded array to contiguous uint8 RGB."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def import_cv2():
    """Import OpenCV with an actionable message when it is missing.

    Shared by every path that touches real pixels: the image_dir and video
    sources, live webcam capture, and camera probing.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "OpenCV is required for camera and file sources. "
            "On a dev machine: pip install '.[video]'. "
            "On the Orin, use JetPack's build (sudo apt install python3-opencv) and "
            "create the venv with --system-site-packages — do not pip install "
            "opencv-python there."
        ) from exc
    return cv2


def decode_fourcc(value: float) -> str:
    """Decode OpenCV's packed FOURCC, or "" when the backend has no answer.

    Not every capture backend exposes the pixel format: macOS AVFoundation
    returns -1, which decodes to four high bytes and renders as mojibake.
    Returning empty is what lets callers say "unknown" instead of inventing
    a format name and then reporting a false mode substitution against it.
    """
    try:
        code = int(value)
    except (TypeError, ValueError):
        return ""
    if code <= 0:
        return ""
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    if not all(c.isascii() and c.isprintable() for c in chars):
        return ""
    return "".join(chars).strip()
