"""Frame sources feeding the benchmark loop."""

from __future__ import annotations

from ..config import SourceConfig
from .base import Frame, FrameSource
from .image_dir import ImageDirSource
from .synthetic import SyntheticSource
from .video import VideoSource

_SOURCES = {
    "synthetic": SyntheticSource,
    "image_dir": ImageDirSource,
    "video": VideoSource,
}


def available_sources():
    return sorted(_SOURCES)


def build_source(cfg: SourceConfig) -> FrameSource:
    try:
        factory = _SOURCES[cfg.type]
    except KeyError:
        raise ValueError(
            f"unknown source type {cfg.type!r}; available: {available_sources()}"
        ) from None
    return factory(cfg)


__all__ = ["Frame", "FrameSource", "build_source", "available_sources"]
