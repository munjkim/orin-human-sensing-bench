"""Task protocol: the unit whose latency we measure."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import TaskConfig
from ..sources.base import Frame


@dataclass
class InferResult:
    """One inference call's output.

    ``stages`` optionally breaks the call into named sub-steps (milliseconds),
    which is what makes a composite pipeline like face recognition
    interpretable rather than a single opaque number.
    """

    payload: Any = None
    detections: int = 0
    stages: Optional[Dict[str, float]] = field(default=None)


class Task(abc.ABC):
    """A benchmarkable inference workload.

    Lifecycle: ``setup()`` -> N x ``infer()`` -> ``teardown()``. Everything
    expensive and one-off (model load, delegate init, memory allocation)
    belongs in ``setup``; the runner additionally burns warmup iterations
    because MediaPipe defers real GPU delegate initialisation to the first
    ``detect`` call.
    """

    #: Registry key, set by @register.
    name: str = ""

    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def setup(self) -> None: ...

    @abc.abstractmethod
    def infer(self, frame: Frame) -> InferResult: ...

    def teardown(self) -> None:
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.cfg.type,
            "model": self.cfg.model,
            "delegate": self.cfg.delegate,
            "running_mode": self.cfg.running_mode,
            "options": dict(self.cfg.options),
        }
