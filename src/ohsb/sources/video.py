"""Frames decoded from a video file.

The whole clip is decoded up front so that decode cost never lands inside
the timed inference loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .base import Frame, FrameSource, import_cv2, to_rgb_uint8
from .image_dir import _resize


class VideoSource(FrameSource):
    def load(self) -> List[Frame]:
        path = Path(self.cfg.path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"source.path is not a file: {path}")

        cv2 = import_cv2()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        limit = self.cfg.count if self.cfg.count > 0 else None

        frames: List[Frame] = []
        try:
            while limit is None or len(frames) < limit:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = to_rgb_uint8(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if self.cfg.resize:
                    rgb = _resize(cv2, rgb, self.cfg.width, self.cfg.height)
                idx = len(frames)
                frames.append(Frame(index=idx, image=rgb, timestamp_ms=int(idx * 1000 / fps)))
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(f"decoded 0 frames from {path}")
        return frames
