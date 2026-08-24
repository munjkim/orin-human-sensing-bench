"""Harness calibration task — no model, no MediaPipe.

Measures what the benchmark loop itself costs. Any real result should be
read against this floor: if a task reports 1.2 ms and noop reports 0.05 ms,
the harness overhead is ~4%.
"""

from __future__ import annotations

import time

from ..sources.base import Frame
from . import register
from .base import InferResult, Task


@register("noop")
class NoopTask(Task):
    """Optionally busy-waits ``options.sleep_ms`` to emulate a known workload."""

    def setup(self) -> None:
        self._sleep_s = float(self.cfg.options.get("sleep_ms", 0.0)) / 1e3

    def infer(self, frame: Frame) -> InferResult:
        if self._sleep_s:
            # Busy-wait: time.sleep()'s scheduler granularity is coarse enough
            # to swamp sub-millisecond targets.
            deadline = time.perf_counter() + self._sleep_s
            while time.perf_counter() < deadline:
                pass
        return InferResult(payload=None, detections=0)

    def describe(self):
        info = super().describe()
        info["model"] = "(none)"
        return info
