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


def create_from_options(mp_class, options, cfg):
    """``<Class>.create_from_options`` with an actionable GPU-delegate error.

    MediaPipe's Linux/aarch64 PyPI wheel is compiled with GPU support removed
    entirely, not just CUDA — even the OpenGL ES delegate is unavailable. It
    fails with a generic ``NotImplementedError`` naming an internal
    calculator, which explains nothing to someone hitting it fresh. There is
    no runtime flag to enable it; see docs/orin-setup.md for what genuine
    GPU/CUDA acceleration on this board would actually require.
    """
    try:
        return mp_class.create_from_options(options)
    except NotImplementedError as exc:
        if cfg.delegate == "gpu" and "GPU processing is disabled" in str(exc):
            raise NotImplementedError(
                "task.delegate=gpu failed: this MediaPipe build has GPU support "
                "compiled out entirely (build flag MEDIAPIPE_DISABLE_GPU) — this is "
                "not a board or config problem, the official PyPI Linux/aarch64 wheel "
                "ships this way. See 'GPU delegate' in docs/orin-setup.md for what "
                "actually getting GPU acceleration would require."
            ) from exc
        raise
