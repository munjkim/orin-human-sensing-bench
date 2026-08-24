"""Live USB camera capture.

Deliberately *not* a :class:`FrameSource`. A FrameSource materialises frames
up front so the offline benchmark can time inference in isolation; a camera
delivers frames in real time and its rate is part of what we are measuring.
Sharing a base class between the two would let one kind of number be
reported as the other.

Three things this class exists to get right:

* **Negotiated settings are recorded, not requested ones.** A UVC camera
  silently substitutes the nearest mode it supports, so asking for 1080p30
  and reporting 1080p30 while receiving 640x480 is a lie the harness would
  otherwise tell.
* **FOURCC is set before the resolution.** On V4L2 the pixel format
  constrains which sizes are available; setting size first and format second
  can leave the device in a mode neither call asked for.
* **Buffer depth is pinned to 1.** With the default queue a consumer slower
  than the camera reads progressively staler frames while appearing to keep
  up. Optional draining goes further and always processes the newest frame.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import decode_fourcc, import_cv2, to_rgb_uint8


class CameraError(RuntimeError):
    pass


class WebcamSource:
    def __init__(self, cfg):
        self.cfg = cfg
        self._cv2 = None
        self._cap = None
        self._negotiated: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> WebcamSource:
        cv2 = self._cv2 = import_cv2()
        device = _device_arg(self.cfg.device)

        # CAP_V4L2 explicitly: OpenCV may otherwise pick a backend that
        # ignores FOURCC requests entirely, making MJPG negotiation silently
        # fail and capping a USB 2.0 camera at raw-YUYV frame rates.
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2) if _is_linux() else cv2.VideoCapture(device)
        if not cap.isOpened():
            raise CameraError(
                f"could not open camera {self.cfg.device!r}. "
                f"Check `ohsb cameras` and that the device is not in use."
            )

        if self.cfg.fourcc and self.cfg.fourcc.lower() != "auto":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc.upper()))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.height))
        if self.cfg.fps:
            cap.set(cv2.CAP_PROP_FPS, float(self.cfg.fps))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.cfg.buffersize))
        except Exception:
            # Not supported by every backend; the drain option covers it.
            pass

        self._cap = cap
        self._negotiated = self._read_back()

        ok, _ = cap.read()
        if not ok:
            self.close()
            raise CameraError(
                f"camera {self.cfg.device!r} opened but returned no frame "
                f"in mode {self._negotiated.get('fourcc')} "
                f"{self._negotiated.get('width')}x{self._negotiated.get('height')}"
            )
        return self

    def _read_back(self) -> Dict[str, Any]:
        cv2, cap = self._cv2, self._cap
        return {
            "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "buffersize": _safe_get(cap, getattr(cv2, "CAP_PROP_BUFFERSIZE", None)),
        }

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    # -- capture -----------------------------------------------------------
    def read_bgr(self) -> Tuple[np.ndarray, int]:
        """Return ``(bgr_frame, frames_skipped)``.

        Deliberately does not time itself: the caller brackets this call with
        the same clock it uses for the other stages, so the stage breakdown
        sums to the frame time by construction rather than by agreement
        between two independent measurements.

        How long this blocks is the signal that separates the two regimes.
        When inference is slower than the camera a frame is always queued and
        the call returns immediately; when inference is faster, this call is
        the camera pacing the loop.
        """
        cap = self._cap
        if cap is None:
            raise CameraError("camera is not open")

        skipped = 0
        if self.cfg.drain:
            # grab() dequeues without decoding, so draining to the newest
            # frame costs far less than read()ing each one.
            skipped = self._drain(cap)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise CameraError("camera returned no frame (disconnected?)")
        return frame, skipped

    def _drain(self, cap) -> int:
        skipped = 0
        for _ in range(self.cfg.max_drain):
            t0 = time.perf_counter()
            if not cap.grab():
                break
            # A grab that blocks means the queue was empty and we are now
            # waiting on the camera — the newest frame is the one we hold.
            if (time.perf_counter() - t0) > self.cfg.drain_block_s:
                break
            skipped += 1
        return skipped

    def to_rgb(self, bgr: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        return to_rgb_uint8(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # -- reporting ---------------------------------------------------------
    @property
    def negotiated(self) -> Dict[str, Any]:
        return dict(self._negotiated)

    def describe(self) -> Dict[str, Any]:
        requested = {
            "device": self.cfg.device,
            "fourcc": self.cfg.fourcc,
            "width": self.cfg.width,
            "height": self.cfg.height,
            "fps": self.cfg.fps,
        }
        negotiated = self.negotiated
        return {
            "type": "webcam",
            "requested": requested,
            "negotiated": negotiated,
            "mode_substituted": _substituted(requested, negotiated),
            "drain": self.cfg.drain,
            "width": negotiated.get("width"),
            "height": negotiated.get("height"),
        }


def _substituted(requested: Dict[str, Any], negotiated: Dict[str, Any]) -> Optional[str]:
    """Describe any silent substitution the driver made, else None."""
    diffs = []
    for key in ("width", "height"):
        if requested.get(key) and negotiated.get(key) and requested[key] != negotiated[key]:
            diffs.append(f"{key}: asked {requested[key]}, got {negotiated[key]}")
    want = (requested.get("fourcc") or "").upper()
    got = (negotiated.get("fourcc") or "").upper()
    # An empty `got` means the backend does not report the format, not that
    # it refused ours — do not manufacture a substitution warning from it.
    if want and want != "AUTO" and got and want != got:
        diffs.append(f"fourcc: asked {want}, got {got}")
    return "; ".join(diffs) if diffs else None


def _device_arg(device):
    if isinstance(device, int):
        return device
    text = str(device)
    return int(text) if text.isdigit() else text


def _is_linux() -> bool:
    import sys

    return sys.platform.startswith("linux")


def _safe_get(cap, prop):
    if prop is None:
        return None
    try:
        return int(cap.get(prop))
    except Exception:
        return None
