"""Benchmark configuration: dataclasses + YAML loading.

A config file fully describes one benchmark run, so a result is always
reproducible from the config embedded in it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigError(ValueError):
    """Raised when a config file is malformed or missing required keys."""


def _build(cls, data: Optional[Dict[str, Any]], path: str):
    """Instantiate a dataclass from a dict, rejecting unknown keys loudly.

    Unknown keys are almost always typos (``delagate: gpu``) and silently
    ignoring them would produce a run that does not match its config.
    """
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) {unknown} under '{path}'; expected one of {sorted(known)}"
        )
    return cls(**data)


@dataclass
class RunConfig:
    name: str = "unnamed"
    iterations: int = 200
    warmup: int = 30
    repeat: int = 1
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.iterations < 1:
            raise ConfigError("run.iterations must be >= 1")
        if self.warmup < 0:
            raise ConfigError("run.warmup must be >= 0")
        if self.repeat < 1:
            raise ConfigError("run.repeat must be >= 1")


@dataclass
class TaskConfig:
    type: str = "pose_landmarker"
    model: str = ""
    delegate: str = "cpu"  # cpu | gpu
    running_mode: str = "image"  # image | video
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.delegate = self.delegate.lower()
        self.running_mode = self.running_mode.lower()
        if self.delegate not in ("cpu", "gpu"):
            raise ConfigError(f"task.delegate must be 'cpu' or 'gpu', got {self.delegate!r}")
        if self.running_mode not in ("image", "video"):
            raise ConfigError(
                f"task.running_mode must be 'image' or 'video', got {self.running_mode!r}"
            )


@dataclass
class SourceConfig:
    type: str = "synthetic"  # synthetic | image_dir | video
    path: str = ""
    width: int = 1280
    height: int = 720
    count: int = 60
    resize: bool = True
    seed: int = 0


@dataclass
class CameraConfig:
    """Live USB camera settings for ``ohsb live``.

    ``fourcc`` is the one to reach for first on a cheap webcam: USB 2.0 has
    nowhere near the bandwidth for raw YUYV at 720p30, so an uncompressed
    mode silently caps the whole pipeline at a few fps no matter how fast
    the model is. MJPG trades that for JPEG decode cost on the CPU.
    """

    device: Any = 0  # index (0) or path ("/dev/video0")
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"  # MJPG | YUYV | auto
    buffersize: int = 1
    #: Drain the V4L2 queue before each read so inference always sees the
    #: newest frame. Measures responsiveness rather than every-frame throughput.
    drain: bool = False
    max_drain: int = 4
    drain_block_s: float = 0.005
    #: Frames pulled and discarded before measuring — UVC auto-exposure and
    #: auto-gain take a second or two to settle, and the frames arriving
    #: during that ramp are both slower and darker than steady state.
    settle_frames: int = 30
    #: Capture-only pass used to establish what the camera alone can deliver.
    #: Without this baseline you cannot tell a camera-bound result from an
    #: inference-bound one.
    baseline_frames: int = 60


@dataclass
class PowerConfig:
    """Power / utilisation sampling.

    ``auto`` prefers jtop (jetson-stats), which reads the INA3221 rails
    through a root daemon and so needs no sudo from us, and falls back to
    parsing ``tegrastats``. Off-board, both are absent and the run simply
    records no power data.
    """

    backend: str = "auto"  # auto | jtop | tegrastats | none
    interval_ms: int = 100
    binary: str = "tegrastats"
    # tegrastats needs root on most JetPack images; jtop does not.
    sudo: bool = True

    BACKENDS = ("auto", "jtop", "tegrastats", "none")

    def __post_init__(self):
        self.backend = self.backend.lower()
        if self.backend not in self.BACKENDS:
            raise ConfigError(
                f"monitor.power.backend must be one of {list(self.BACKENDS)}, "
                f"got {self.backend!r}"
            )
        if self.interval_ms < 10:
            raise ConfigError("monitor.power.interval_ms must be >= 10")


@dataclass
class MonitorConfig:
    power: PowerConfig = field(default_factory=PowerConfig)


@dataclass
class OutputConfig:
    dir: str = "results"
    save_raw_latencies: bool = True
    save_power_samples: bool = False


@dataclass
class BenchmarkConfig:
    run: RunConfig = field(default_factory=RunConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Kept verbatim so the exact input is recorded alongside the result.
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkConfig:
        data = dict(data or {})
        known = {"run", "task", "source", "camera", "monitor", "output"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown top-level key(s) {unknown}; expected {sorted(known)}")
        monitor_data = dict(data.get("monitor") or {})
        monitor = MonitorConfig(
            power=_build(PowerConfig, monitor_data.pop("power", None), "monitor.power")
        )
        if monitor_data:
            raise ConfigError(f"unknown key(s) {sorted(monitor_data)} under 'monitor'")
        return cls(
            run=_build(RunConfig, data.get("run"), "run"),
            task=_build(TaskConfig, data.get("task"), "task"),
            source=_build(SourceConfig, data.get("source"), "source"),
            camera=_build(CameraConfig, data.get("camera"), "camera"),
            monitor=monitor,
            output=_build(OutputConfig, data.get("output"), "output"),
            raw=copy.deepcopy(data),
        )

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkConfig:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        cfg = cls.from_dict(data)
        cfg.run.tags.setdefault("config", str(path))
        return cfg

    def apply_overrides(self, overrides: Dict[str, Any]) -> BenchmarkConfig:
        """Apply CLI ``--set a.b=c`` overrides, returning a new config."""
        data = copy.deepcopy(self.raw)
        for dotted, value in overrides.items():
            keys = dotted.split(".")
            cursor = data
            for key in keys[:-1]:
                cursor = cursor.setdefault(key, {})
                if not isinstance(cursor, dict):
                    raise ConfigError(f"cannot override '{dotted}': '{key}' is not a mapping")
            cursor[keys[-1]] = value
        return BenchmarkConfig.from_dict(data)
