"""Task registry.

New workloads register themselves with ``@register("name")`` and become
addressable from a config's ``task.type``.
"""

from __future__ import annotations

from typing import Dict, Type

from ..config import TaskConfig
from .base import InferResult, Task

_REGISTRY: Dict[str, Type[Task]] = {}


def register(name: str):
    def decorator(cls: Type[Task]) -> Type[Task]:
        if name in _REGISTRY:
            raise ValueError(f"task {name!r} is already registered by {_REGISTRY[name]!r}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_tasks():
    return sorted(_REGISTRY)


def build_task(cfg: TaskConfig) -> Task:
    try:
        cls = _REGISTRY[cfg.type]
    except KeyError:
        raise ValueError(
            f"unknown task type {cfg.type!r}; available: {available_tasks()}"
        ) from None
    return cls(cfg)


# Imported for their side effect of registering. Kept at the bottom because
# each module imports `register` from this one.
from . import face, noop, pose  # noqa: E402,F401

__all__ = ["Task", "InferResult", "register", "build_task", "available_tasks"]
