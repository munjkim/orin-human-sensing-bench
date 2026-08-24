"""Discover what a USB camera actually supports.

Answering "how many fps can this webcam do" starts here, not at the model:
a cheap UVC camera on USB 2.0 is usually bandwidth-limited, and the same
device will do 720p30 in MJPG but 720p10 in raw YUYV. Guessing the envelope
produces benchmarks that measure the USB bus.

v4l2-ctl is authoritative when present. The OpenCV fallback can only tell
you what a device accepts, not what it advertises, so it probes a candidate
grid and reports what was actually negotiated.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Common UVC modes, smallest first. Used only by the OpenCV fallback.
PROBE_RESOLUTIONS = (
    (320, 240), (640, 480), (800, 600), (1280, 720), (1920, 1080),
)
PROBE_FOURCCS = ("MJPG", "YUYV")


def list_devices() -> List[str]:
    return [str(p) for p in sorted(Path("/dev").glob("video*"))]


def _v4l2_available() -> bool:
    return shutil.which("v4l2-ctl") is not None


def _run(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    try:
        out = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def device_name(device: str) -> Optional[str]:
    if not _v4l2_available():
        return None
    text = _run(["v4l2-ctl", "-d", device, "--info"])
    if not text:
        return None
    match = re.search(r"Card type\s*:\s*(.+)", text)
    return match.group(1).strip() if match else None


def parse_formats(text: str) -> List[Dict[str, Any]]:
    """Parse ``v4l2-ctl --list-formats-ext`` output.

    The format is a three-level indent: pixel format, then Size blocks, then
    Interval lines carrying the achievable fps for that exact size. fps is a
    property of (format, size) — which is precisely why a camera can do 30
    fps at 640x480 and 5 fps at 1080p on the same connection.
    """
    modes: List[Dict[str, Any]] = []
    fourcc: Optional[str] = None
    description = ""
    size: Optional[str] = None

    for line in text.splitlines():
        stripped = line.strip()

        fmt = re.match(r"\[\d+\]:\s*'(\w+)'\s*\((.+?)(?:,\s*compressed)?\)", stripped)
        if fmt:
            fourcc, description = fmt.group(1), fmt.group(2)
            size = None
            continue

        dims = re.match(r"Size:\s*Discrete\s*(\d+)x(\d+)", stripped)
        if dims:
            size = f"{dims.group(1)}x{dims.group(2)}"
            continue

        interval = re.search(r"Interval:\s*Discrete\s*[\d.]+s\s*\(([\d.]+)\s*fps\)", stripped)
        if interval and fourcc and size:
            width, height = (int(v) for v in size.split("x"))
            modes.append({
                "fourcc": fourcc,
                "description": description,
                "width": width,
                "height": height,
                "fps": float(interval.group(1)),
                "compressed": fourcc in ("MJPG", "H264", "MPEG"),
            })

    return modes


def probe_v4l2(device: str) -> Optional[List[Dict[str, Any]]]:
    if not _v4l2_available():
        return None
    text = _run(["v4l2-ctl", "-d", device, "--list-formats-ext"])
    return parse_formats(text) if text else None


def device_kind(device: str) -> str:
    """"video" | "metadata" | "unknown".

    A UVC webcam commonly claims two /dev/video nodes: one that captures and
    one that carries per-frame metadata. The metadata node enumerates zero
    formats while still reporting ``Type: Video Capture``, so the formats
    listing alone cannot tell them apart — only the Device Caps block can.
    Labelling it matters because "no modes reported" reads as a broken
    camera when it is a perfectly normal second node.
    """
    if not _v4l2_available():
        return "unknown"
    text = _run(["v4l2-ctl", "-d", device, "--all"])
    if not text:
        return "unknown"
    caps = text.split("Device Caps", 1)
    block = caps[1] if len(caps) > 1 else text
    block = block.split("Priority", 1)[0]
    if "Video Capture" in block:
        return "video"
    if "Metadata Capture" in block:
        return "metadata"
    return "unknown"


def probe_opencv(device: str) -> List[Dict[str, Any]]:
    """Fallback: ask the device to accept each candidate mode.

    Reports what came back, not what was asked for — a UVC camera silently
    substitutes the nearest mode it supports rather than failing.
    """
    from .sources.base import decode_fourcc, import_cv2

    cv2 = import_cv2()
    index = _device_index(device)
    modes: List[Dict[str, Any]] = []

    for fourcc in PROBE_FOURCCS:
        for width, height in PROBE_RESOLUTIONS:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual = {
                "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "requested": {"fourcc": fourcc, "width": width, "height": height},
            }
            cap.release()
            if actual["width"] and actual not in modes:
                modes.append(actual)
    return modes


def _device_index(device: str):
    match = re.search(r"(\d+)$", str(device))
    return int(match.group(1)) if match else device


def probe(device: str) -> Dict[str, Any]:
    modes = probe_v4l2(device)
    if modes is not None:
        info = {"device": device, "name": device_name(device), "source": "v4l2-ctl",
                "modes": modes}
        if not modes:
            info["kind"] = device_kind(device)
        return info
    return {
        "device": device,
        "name": None,
        "source": "opencv-probe",
        "note": "install v4l2-utils for the authoritative list: sudo apt install v4l-utils",
        "modes": probe_opencv(device),
    }


def collapse_modes(modes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (format, resolution), keeping the highest fps.

    A UVC camera advertises every resolution at every frame rate it can
    divide down to — a C920 enumerates 336 combinations. Listing them all
    buries the only number that matters, which is the *ceiling* at each
    resolution; nothing is learned from seeing 1920x1080 at 30, 24, 20, 15,
    10, 7.5 and 5 fps.
    """
    best: Dict[tuple, Dict[str, Any]] = {}
    for mode in modes:
        key = (mode["fourcc"], mode["width"], mode["height"])
        if key not in best or mode.get("fps", 0) > best[key].get("fps", 0):
            best[key] = mode
    return sorted(
        best.values(),
        key=lambda m: (m["fourcc"], -(m["width"] * m["height"])),
    )


def realtime_ceiling(modes: List[Dict[str, Any]], fps: float = 30.0) -> Dict[str, Any]:
    """Largest resolution reaching ``fps``, per format.

    This one line usually answers "what can this camera actually do" — and
    on a bandwidth-limited USB link it is where compressed and raw formats
    diverge hardest.
    """
    out: Dict[str, Any] = {}
    for mode in modes:
        if mode.get("fps", 0) < fps:
            continue
        key = mode["fourcc"]
        area = mode["width"] * mode["height"]
        if key not in out or area > out[key]["width"] * out[key]["height"]:
            out[key] = mode
    return out


def format_modes(info: Dict[str, Any], max_rows: int = 60) -> str:
    """Render a probe as a table, collapsed to the ceiling per resolution."""
    lines = [f"{info['device']}  {info.get('name') or ''}  (via {info['source']})"]
    if info.get("note"):
        lines.append(f"  note: {info['note']}")

    modes = info.get("modes") or []
    if not modes:
        kind = info.get("kind")
        if kind == "metadata":
            lines.append("  metadata node, not a capture device — this is normal")
        elif kind == "video":
            lines.append("  reports no formats (device busy, or opened by another process?)")
        else:
            lines.append("  no modes reported")
        return "\n".join(lines)

    ceiling = realtime_ceiling(modes)
    if ceiling:
        summary = "  |  ".join(
            f"{fourcc} {m['width']}x{m['height']}"
            for fourcc, m in sorted(
                ceiling.items(), key=lambda kv: -(kv[1]["width"] * kv[1]["height"])
            )
        )
        lines.append(f"  30 fps ceiling:  {summary}")

    collapsed = collapse_modes(modes)
    lines.append(f"  {len(modes)} modes -> {len(collapsed)} (max fps per resolution)")
    lines.append(f"  {'format':<8} {'resolution':>12} {'max fps':>8}   {'note'}")
    for mode in collapsed[:max_rows]:
        note = "compressed" if mode.get("compressed") else ""
        lines.append(
            f"  {mode['fourcc']:<8} {mode['width']:>5}x{mode['height']:<6} "
            f"{mode.get('fps', 0):>7.1f}   {note}"
        )
    if len(collapsed) > max_rows:
        lines.append(f"  ... {len(collapsed) - max_rows} more")
    return "\n".join(lines)


def best_mode(modes: List[Dict[str, Any]], width: int, height: int) -> Optional[Dict[str, Any]]:
    """Highest-fps mode matching a resolution, preferring compressed formats."""
    matches = [m for m in modes if m["width"] == width and m["height"] == height]
    if not matches:
        return None
    return max(matches, key=lambda m: (m.get("fps", 0), m.get("compressed", False)))
