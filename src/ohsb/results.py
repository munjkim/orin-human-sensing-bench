"""Benchmark result schema, persistence, and rendering.

One run produces one JSON file that is self-contained: config, platform
snapshot, latency distribution, throughput, and power. Nothing about a
stored result should require the shell history that produced it.
"""

from __future__ import annotations

import json
import platform as _platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__

SCHEMA_VERSION = 1


@dataclass
class RunResult:
    """A single measured run (one repeat of one config)."""

    run_id: str
    name: str
    started_at: str
    duration_s: float
    iterations: int
    warmup: int

    latency_ms: Dict[str, Any] = field(default_factory=dict)
    stage_latency_ms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    throughput_fps: float = 0.0
    detections: Dict[str, Any] = field(default_factory=dict)
    power: Dict[str, Any] = field(default_factory=dict)
    energy: Dict[str, Any] = field(default_factory=dict)
    #: Whether GPU frequency actually varied during this run — see
    #: runner._dvfs_observed. The trustworthy source for "was jetson_clocks
    #: really active", derived from the run itself rather than guessed
    #: beforehand.
    dvfs: Dict[str, Any] = field(default_factory=dict)
    #: Live-capture findings: camera baseline, realtime ratio, bottleneck verdict.
    live: Dict[str, Any] = field(default_factory=dict)
    raw_latencies_ms: Optional[List[float]] = None
    raw_samples: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkReport:
    """Everything written to one result file."""

    schema_version: int
    ohsb_version: str
    created_at: str
    #: "offline" = pre-decoded frames, pure inference latency.
    #: "live"    = end-to-end USB camera pipeline, camera rate included.
    mode: str
    config: Dict[str, Any]
    task: Dict[str, Any]
    source: Dict[str, Any]
    platform: Dict[str, Any]
    warnings: List[str]
    runs: List[RunResult]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["runs"] = [_drop_none(r) for r in data["runs"]]
        return data

    def write(self, directory: str | Path, filename: Optional[str] = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if filename is None:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            slug = _slug(self.config.get("run", {}).get("name", "run"))
            filename = f"{stamp}_{slug}.json"
        path = directory / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=False, default=str)
            fh.write("\n")
        return path


def new_report(config: Dict[str, Any], task: Dict[str, Any], source: Dict[str, Any],
               platform_snapshot: Dict[str, Any], warnings: List[str],
               mode: str = "offline") -> BenchmarkReport:
    return BenchmarkReport(
        schema_version=SCHEMA_VERSION,
        ohsb_version=__version__,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        mode=mode,
        config=config,
        task=task,
        source=source,
        platform=platform_snapshot,
        warnings=warnings,
        runs=[],
    )


def new_run_id() -> str:
    return f"{_platform.node().split('.')[0]}-{uuid.uuid4().hex[:8]}"


def _drop_none(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in str(text)]
    return "".join(keep).strip("-").lower() or "run"


# -- rendering -------------------------------------------------------------

def format_report(report: BenchmarkReport, verbose: bool = False) -> str:
    """Human-readable summary printed at the end of a run."""
    lines: List[str] = []
    task, source = report.task, report.source
    lines.append(
        f"task={task.get('type')} delegate={task.get('delegate')} "
        f"mode={task.get('running_mode')} model={Path(str(task.get('model') or '-')).name}"
    )
    if report.mode == "live":
        lines.append(_format_camera(source))
    else:
        lines.append(
            f"source={source.get('type')} {source.get('width')}x{source.get('height')} "
            f"frames={source.get('frames')}"
        )

    for run in report.runs:
        lines.append("")
        lines.append(f"[{run.name}] {run.iterations} iters (+{run.warmup} warmup)")
        lat = run.latency_ms
        if lat.get("count"):
            lines.append(
                "  latency ms   "
                f"mean={lat['mean']:.2f}  p50={lat['p50']:.2f}  p90={lat['p90']:.2f}  "
                f"p95={lat['p95']:.2f}  p99={lat['p99']:.2f}  max={lat['max']:.2f}"
            )
        lines.append(f"  throughput   {run.throughput_fps:.1f} fps")

        for stage, stats in run.stage_latency_ms.items():
            share = stats.get("share_pct")
            suffix = f"  ({share:.0f}% of frame)" if share is not None else ""
            lines.append(f"    stage {stage:<10} mean={stats['mean']:.2f} ms{suffix}")

        # The verdict reads as a conclusion, so it comes after the evidence.
        if run.live:
            lines.extend(_format_live(run.live))

        if run.detections:
            lines.append(
                f"  detections   mean={run.detections.get('mean', 0):.2f} "
                f"per frame (sanity check only, not accuracy)"
            )

        dvfs = run.dvfs
        if dvfs.get("conclusive"):
            tag = "DVFS LIVE" if dvfs["scaled"] else "clock steady"
            lines.append(f"  gpu clock    {tag} — {dvfs['note']}")

        power = run.power
        if power.get("available"):
            rail = power.get("total_rail")
            if rail:
                lines.append(
                    f"  power        {rail}={power['avg_total_power_mw'] / 1000:.2f} W avg "
                    f"({power['backend']}, {power['samples']} samples)"
                )
            for name, stats in sorted(power.get("power_mw", {}).items()):
                if name != rail:
                    lines.append(f"    rail {name:<16} {stats['mean'] / 1000:.2f} W avg")
            if run.energy:
                lines.append(
                    f"  energy       {run.energy['mj_per_frame']:.1f} mJ/frame  "
                    f"({run.energy['fps_per_watt']:.2f} fps/W)"
                )
            if power.get("gpu_util_pct"):
                lines.append(f"  gpu util     {power['gpu_util_pct']['mean']:.1f}% avg")
        else:
            lines.append(f"  power        unavailable — {power.get('reason', 'unknown')}")

    if report.warnings:
        lines.append("")
        lines.append("warnings:")
        lines.extend(f"  ! {w}" for w in report.warnings)
    return "\n".join(lines)


def _format_camera(source: Dict[str, Any]) -> str:
    negotiated = source.get("negotiated", {})
    line = (
        f"camera={source.get('requested', {}).get('device')} "
        f"{negotiated.get('fourcc') or 'fourcc?'} "
        f"{negotiated.get('width')}x{negotiated.get('height')}"
        f"@{negotiated.get('fps', 0):.0f}"
    )
    if source.get("drain"):
        line += " drain=on"
    if source.get("mode_substituted"):
        # The driver gave us something other than what was asked for; saying
        # so loudly is the difference between a result and a fiction.
        line += f"\n  ! driver substituted mode — {source['mode_substituted']}"
    return line


def _format_live(live: Dict[str, Any]) -> List[str]:
    lines = [
        f"  camera max   {live['camera_baseline_fps']:.1f} fps "
        f"(capture only, no inference)",
        f"  pipeline cap {live['pipeline_capable_fps']:.1f} fps "
        f"(convert+infer, camera removed)",
        f"  realtime     {live['realtime_ratio'] * 100:.0f}% of camera rate",
    ]
    if live.get("frames_skipped"):
        lines.append(
            f"  skipped      {live['frames_skipped']} frames drained as stale"
        )
    lines.append(f"  >> {live['verdict']}")
    return lines


def load_report(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
