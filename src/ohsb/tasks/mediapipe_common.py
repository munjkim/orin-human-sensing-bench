"""Shared MediaPipe Tasks plumbing.

MediaPipe is imported lazily inside functions so that the package imports,
``ohsb list`` runs, and the test suite passes on a machine without it — which
matters because the harness is authored off-board and only runs on the Orin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

_INSTALL_HINT = (
    "MediaPipe is not installed. On x86/macOS: pip install '.[mediapipe]'. "
    "On Jetson, install a JetPack-matched wheel first — see docs/orin-setup.md."
)


def import_mediapipe() -> Tuple[Any, Any, Any]:
    """Return ``(mp, mp_python, vision)`` or raise with an actionable message."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_INSTALL_HINT) from exc
    return mp, mp_python, vision


def resolve_model(path: str, task_type: str) -> Path:
    if not path:
        raise ValueError(f"task.model is required for {task_type}; run scripts/fetch_models.sh")
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"model bundle not found: {resolved}. Run scripts/fetch_models.sh to download it."
        )
    return resolved


def base_options(cfg, task_type: str):
    """Build ``BaseOptions`` with the configured delegate."""
    _, mp_python, _ = import_mediapipe()
    delegate = (
        mp_python.BaseOptions.Delegate.GPU
        if cfg.delegate == "gpu"
        else mp_python.BaseOptions.Delegate.CPU
    )
    return mp_python.BaseOptions(
        model_asset_path=str(resolve_model(cfg.model, task_type)),
        delegate=delegate,
    )


def running_mode(cfg):
    _, _, vision = import_mediapipe()
    # LIVE_STREAM is intentionally unsupported: its callback API measures
    # end-to-end pipeline latency, not per-call inference latency, and mixing
    # the two in one results table would be misleading.
    return (
        vision.RunningMode.VIDEO if cfg.running_mode == "video" else vision.RunningMode.IMAGE
    )


def to_mp_image(image: np.ndarray):
    mp, _, _ = import_mediapipe()
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image)


def take(options: Dict[str, Any], key: str, default):
    value = options.get(key, default)
    return type(default)(value) if value is not None else default
