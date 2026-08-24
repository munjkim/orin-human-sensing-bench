"""Monitor protocol.

A monitor samples system state in the background for the duration of the
timed loop. Monitors must never raise into the benchmark: a missing
``tegrastats`` should degrade the result, not lose the run. Failures are
recorded in ``summary()["error"]`` instead.
"""

from __future__ import annotations

import abc
from typing import Any, Dict


class Monitor(abc.ABC):
    name: str = "monitor"

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def summary(self) -> Dict[str, Any]: ...

    def samples(self) -> Dict[str, Any]:
        """Raw per-sample data, written only when output.save_power_samples."""
        return {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


class NullMonitor(Monitor):
    name = "null"

    def __init__(self, reason: str = ""):
        self.reason = reason

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def summary(self) -> Dict[str, Any]:
        return {"available": False, "reason": self.reason}
