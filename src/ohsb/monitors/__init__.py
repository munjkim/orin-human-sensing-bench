"""System monitors sampled alongside the timed loop."""

from __future__ import annotations

from typing import List

from ..config import MonitorConfig, PowerConfig
from .base import Monitor, NullMonitor
from .jtop_monitor import JtopMonitor
from .tegrastats import TegrastatsMonitor


def build_power_monitor(cfg: PowerConfig) -> Monitor:
    """Pick a power backend.

    ``auto`` prefers jtop (no sudo, structured payload) and falls back to
    tegrastats. An explicit backend is never silently substituted — if you
    asked for tegrastats and it is unusable, the result says so.
    """
    if cfg.backend == "none":
        return NullMonitor("monitor.power.backend = none")
    if cfg.backend == "jtop":
        return JtopMonitor(cfg)
    if cfg.backend == "tegrastats":
        return TegrastatsMonitor(cfg)
    # auto
    if JtopMonitor.is_available():
        return JtopMonitor(cfg)
    if TegrastatsMonitor.is_available(cfg):
        return TegrastatsMonitor(cfg)
    return NullMonitor("neither jtop nor tegrastats is available (not a Jetson?)")


def build_monitors(cfg: MonitorConfig) -> List[Monitor]:
    return [build_power_monitor(cfg.power)]


__all__ = [
    "Monitor",
    "NullMonitor",
    "JtopMonitor",
    "TegrastatsMonitor",
    "build_monitors",
    "build_power_monitor",
]
