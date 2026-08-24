"""The benchmark loop.

Design rules that the numbers depend on:

* **Decode happens before timing.** Sources materialise all frames up front,
  so a measured interval contains inference and nothing else.
* **Warmup is not optional.** MediaPipe defers real delegate initialisation
  (GPU context, tensor arena) to the first ``detect`` call, so iteration 0 is
  routinely an order of magnitude slower than steady state.
* **Repeats reuse one setup.** Model load is paid once; repeating only the
  timed loop is what exposes thermal drift, which on a passively-clocked
  15 W Orin Nano is a real effect rather than a footnote.
* **Monitors bracket the timed loop only**, so average power is the power of
  the workload, not of the workload plus model loading.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from .config import BenchmarkConfig
from .metrics import summarize
from .monitors import build_monitors
from .platform import reproducibility_warnings, snapshot
from .results import BenchmarkReport, RunResult, format_report, new_report, new_run_id
from .sources import build_source
from .sources.base import Frame
from .tasks import build_task

ProgressFn = Callable[[str], None]


def _noop(_message: str) -> None:
    return None


def run_benchmark(cfg: BenchmarkConfig, progress: Optional[ProgressFn] = None) -> BenchmarkReport:
    say = progress or _noop

    snap = snapshot()
    warnings = reproducibility_warnings(snap)

    say(f"preparing source: {cfg.source.type}")
    source = build_source(cfg.source).prepare()
    frames = source.frames

    say(f"building task: {cfg.task.type} (delegate={cfg.task.delegate})")
    task = build_task(cfg.task)

    report = new_report(
        config=cfg.raw or {
            "run": vars(cfg.run), "task": vars(cfg.task), "source": vars(cfg.source)
        },
        task=task.describe(),
        source=source.describe(),
        platform_snapshot=snap,
        warnings=warnings,
    )

    task.setup()
    try:
        if cfg.run.warmup:
            say(f"warmup: {cfg.run.warmup} iterations")
            for i in range(cfg.run.warmup):
                task.infer(frames[i % len(frames)])

        for repeat in range(cfg.run.repeat):
            label = cfg.run.name if cfg.run.repeat == 1 else f"{cfg.run.name}#{repeat + 1}"
            say(f"measuring: {label} ({cfg.run.iterations} iterations)")
            # Warmup is paid once, before the first repeat; later repeats
            # inherit the warm state rather than re-warming.
            warmup = cfg.run.warmup if repeat == 0 else 0
            report.runs.append(_measure(cfg, task, frames, label, warmup))
    finally:
        task.teardown()

    return report


def _measure(cfg: BenchmarkConfig, task, frames, label: str, warmup: int) -> RunResult:
    monitors = build_monitors(cfg.monitor)
    latencies: List[float] = []
    stage_totals: Dict[str, List[float]] = defaultdict(list)
    detections: List[int] = []
    n_frames = len(frames)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    for monitor in monitors:
        monitor.start()

    wall_start = time.perf_counter()
    for i in range(cfg.run.iterations):
        frame = frames[i % n_frames]
        t0 = time.perf_counter_ns()
        result = task.infer(frame)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)
        detections.append(result.detections)
        if result.stages:
            for stage, value in result.stages.items():
                stage_totals[stage].append(value)
    wall_s = time.perf_counter() - wall_start

    for monitor in monitors:
        monitor.stop()

    latency = summarize(latencies, unit="ms")
    throughput = cfg.run.iterations / wall_s if wall_s > 0 else 0.0

    stage_summary: Dict[str, Dict[str, Any]] = {}
    for stage, values in stage_totals.items():
        stats = summarize(values, unit="ms")
        if latency.get("mean"):
            stats["share_pct"] = 100.0 * stats["mean"] / latency["mean"]
        stage_summary[stage] = stats

    power: Dict[str, Any] = {}
    raw_samples: Dict[str, Any] = {}
    for monitor in monitors:
        power = monitor.summary()
        if cfg.output.save_power_samples:
            raw_samples.update(monitor.samples())

    return RunResult(
        run_id=new_run_id(),
        name=label,
        started_at=started_at,
        duration_s=wall_s,
        iterations=cfg.run.iterations,
        warmup=warmup,
        latency_ms=latency,
        stage_latency_ms=stage_summary,
        throughput_fps=throughput,
        detections=summarize(detections),
        power=power,
        energy=_energy(power, wall_s, cfg.run.iterations, throughput),
        raw_latencies_ms=latencies if cfg.output.save_raw_latencies else None,
        raw_samples=raw_samples or None,
    )


def _dvfs_observed(power: Dict[str, Any]) -> Dict[str, Any]:
    """Did the GPU clock actually change during THIS measured run?

    This replaces every pre-run guess at whether jetson_clocks is active.
    Three different static probes (devfreq sysfs, governor name, a single
    jtop sample) were each falsified on the Orin Nano — the clearest case
    being a jtop sample taken on a fresh reboot, jetson_clocks never run,
    that read gpu.freq.min == gpu.freq.max simply because the GPU had load
    at that instant. A snapshot cannot tell "pinned" from "busy" apart.

    What can: sampling gpu_freq_khz for the whole duration of the actual
    benchmark loop and checking whether it varied. That is a fact about the
    run that produced these numbers, not an inference about the system
    state before it, so it needs no falsifiable heuristic.
    """
    gpu_freq = power.get("gpu_freq_khz")
    if not gpu_freq or gpu_freq.get("count", 0) < 2:
        return {"conclusive": False,
                "reason": "fewer than 2 GPU frequency samples during the run"}
    lo, hi = gpu_freq.get("min"), gpu_freq.get("max")
    scaled = lo != hi
    result = {"conclusive": True, "scaled": scaled, "min_khz": lo, "max_khz": hi,
              "samples": gpu_freq["count"]}
    result["note"] = (
        f"GPU frequency varied {lo:.0f}-{hi:.0f} kHz during this run — DVFS was live"
        if scaled else
        f"GPU frequency held steady at {hi:.0f} kHz for all {gpu_freq['count']} "
        "samples during this run"
    )
    return result


def _energy(power: Dict[str, Any], wall_s: float, iterations: int,
            throughput_fps: float) -> Dict[str, Any]:
    """Energy per frame and efficiency, from average total-rail power.

    mW x s = mJ, so total energy is simply ``avg_power_mw * wall_s``. This is
    module-input power, not the accelerator's share: it includes the CPU,
    display and everything else drawing from the rail, which is the honest
    number for an edge power budget but not an attribution to the model.
    """
    if not power.get("available") or "avg_total_power_mw" not in power or iterations <= 0:
        return {}
    avg_mw = power["avg_total_power_mw"]
    total_mj = avg_mw * wall_s
    watts = avg_mw / 1000.0
    return {
        "rail": power.get("total_rail"),
        "avg_power_w": watts,
        "total_mj": total_mj,
        "mj_per_frame": total_mj / iterations,
        "fps_per_watt": (throughput_fps / watts) if watts > 0 else 0.0,
    }


# -- live capture ----------------------------------------------------------

def run_live_benchmark(cfg: BenchmarkConfig,
                       progress: Optional[ProgressFn] = None) -> BenchmarkReport:
    """End-to-end USB camera pipeline: capture -> convert -> infer.

    This answers a different question from :func:`run_benchmark`. Offline
    mode reports what the model costs; live mode reports what you actually
    get, which on a cheap webcam is frequently limited by USB bandwidth
    rather than by the Orin.

    Because "23 fps" alone cannot distinguish a camera that only delivers 23
    from a model that can only process 23, every live run also measures a
    **capture-only baseline** and reports which side is the constraint.
    """
    from .sources.webcam import WebcamSource

    say = progress or _noop
    snap = snapshot()
    warnings = reproducibility_warnings(snap)

    say(f"opening camera {cfg.camera.device!r}")
    camera = WebcamSource(cfg.camera).open()

    try:
        substituted = camera.describe().get("mode_substituted")
        if substituted:
            warnings.append(f"camera did not honour the requested mode — {substituted}")

        if cfg.camera.settle_frames:
            say(f"settling auto-exposure: {cfg.camera.settle_frames} frames")
            for _ in range(cfg.camera.settle_frames):
                camera.read_bgr()

        say(f"capture-only baseline: {cfg.camera.baseline_frames} frames")
        baseline = _capture_baseline(camera, cfg.camera.baseline_frames)
        say(f"  camera delivers {baseline['fps']:.1f} fps on its own")

        say(f"building task: {cfg.task.type} (delegate={cfg.task.delegate})")
        task = build_task(cfg.task)

        report = new_report(
            config=cfg.raw or {"run": vars(cfg.run), "task": vars(cfg.task)},
            task=task.describe(),
            source=camera.describe(),
            platform_snapshot=snap,
            warnings=warnings,
            mode="live",
        )

        task.setup()
        try:
            if cfg.run.warmup:
                say(f"warmup: {cfg.run.warmup} frames")
                for _ in range(cfg.run.warmup):
                    bgr, _ = camera.read_bgr()
                    task.infer(Frame(index=0, image=camera.to_rgb(bgr)))

            for repeat in range(cfg.run.repeat):
                label = cfg.run.name if cfg.run.repeat == 1 else f"{cfg.run.name}#{repeat + 1}"
                say(f"measuring: {label} ({cfg.run.iterations} frames)")
                warmup = cfg.run.warmup if repeat == 0 else 0
                report.runs.append(_measure_live(cfg, task, camera, label, warmup, baseline))
        finally:
            task.teardown()
    finally:
        camera.close()

    return report


def _capture_baseline(camera, frames: int) -> Dict[str, Any]:
    """What the camera delivers with nothing downstream of it.

    Plain reads, no draining and no colour conversion: this is the ceiling
    the full pipeline is measured against.
    """
    if frames <= 0:
        return {"fps": float("nan"), "frames": 0, "read_ms": {}}
    waits: List[float] = []
    start = time.perf_counter()
    for _ in range(frames):
        t0 = time.perf_counter()
        camera.read_bgr()
        waits.append((time.perf_counter() - t0) * 1e3)
    elapsed = time.perf_counter() - start
    return {
        "fps": frames / elapsed if elapsed > 0 else float("nan"),
        "frames": frames,
        "read_ms": summarize(waits, unit="ms"),
    }


def _measure_live(cfg: BenchmarkConfig, task, camera, label: str, warmup: int,
                  baseline: Dict[str, Any]) -> RunResult:
    monitors = build_monitors(cfg.monitor)
    totals: List[float] = []
    stages: Dict[str, List[float]] = defaultdict(list)
    detections: List[int] = []
    skipped_total = 0

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    for monitor in monitors:
        monitor.start()

    wall_start = time.perf_counter()
    for i in range(cfg.run.iterations):
        loop_t0 = time.perf_counter()

        bgr, skipped = camera.read_bgr()
        t_convert0 = time.perf_counter()
        skipped_total += skipped

        rgb = camera.to_rgb(bgr)
        t_convert1 = time.perf_counter()

        result = task.infer(Frame(index=i, image=rgb, timestamp_ms=int(loop_t0 * 1e3)))
        t_infer1 = time.perf_counter()

        stages["capture"].append((t_convert0 - loop_t0) * 1e3)
        stages["convert"].append((t_convert1 - t_convert0) * 1e3)
        stages["infer"].append((t_infer1 - t_convert1) * 1e3)
        totals.append((t_infer1 - loop_t0) * 1e3)
        detections.append(result.detections)
    wall_s = time.perf_counter() - wall_start

    for monitor in monitors:
        monitor.stop()

    latency = summarize(totals, unit="ms")
    throughput = cfg.run.iterations / wall_s if wall_s > 0 else 0.0

    stage_summary: Dict[str, Dict[str, Any]] = {}
    for stage, values in stages.items():
        stats = summarize(values, unit="ms")
        if latency.get("mean"):
            stats["share_pct"] = 100.0 * stats["mean"] / latency["mean"]
        stage_summary[stage] = stats

    power: Dict[str, Any] = {}
    raw_samples: Dict[str, Any] = {}
    for monitor in monitors:
        power = monitor.summary()
        if cfg.output.save_power_samples:
            raw_samples.update(monitor.samples())

    return RunResult(
        run_id=new_run_id(),
        name=label,
        started_at=started_at,
        duration_s=wall_s,
        iterations=cfg.run.iterations,
        warmup=warmup,
        latency_ms=latency,
        stage_latency_ms=stage_summary,
        throughput_fps=throughput,
        detections=summarize(detections),
        power=power,
        energy=_energy(power, wall_s, cfg.run.iterations, throughput),
        dvfs=_dvfs_observed(power),
        live=_diagnose(baseline, stage_summary, throughput, skipped_total),
        raw_latencies_ms=totals if cfg.output.save_raw_latencies else None,
        raw_samples=raw_samples or None,
    )


def _diagnose(baseline: Dict[str, Any], stages: Dict[str, Dict[str, Any]],
              throughput_fps: float, skipped: int) -> Dict[str, Any]:
    """Decide whether the camera or the compute is the constraint.

    ``pipeline_capable_fps`` deliberately excludes the capture wait: it is
    what the Orin could sustain if frames arrived instantly. Comparing it to
    the camera's own baseline is what turns a bare fps number into an
    actionable answer — buy a better camera, or pick a smaller model.
    """
    camera_fps = baseline.get("fps", float("nan"))
    convert_ms = stages.get("convert", {}).get("mean", 0.0)
    infer_ms = stages.get("infer", {}).get("mean", 0.0)
    compute_ms = convert_ms + infer_ms
    pipeline_fps = (1000.0 / compute_ms) if compute_ms > 0 else float("inf")

    dominant = max(
        (name for name in ("capture", "convert", "infer") if name in stages),
        key=lambda name: stages[name]["mean"],
        default="unknown",
    )

    # camera_fps is NaN when the baseline pass was skipped.
    usable = math.isfinite(camera_fps) and camera_fps > 0
    ratio = throughput_fps / camera_fps if usable else float("nan")
    headroom = pipeline_fps / camera_fps if usable else float("nan")

    if not usable:
        verdict = "camera baseline unavailable — cannot attribute the bottleneck"
    elif headroom >= 1.1:
        verdict = (
            f"CAMERA-BOUND — compute could sustain {pipeline_fps:.0f} fps but the camera "
            f"only delivers {camera_fps:.0f} fps. A faster model will not help; "
            f"a better camera mode or sensor will."
        )
    elif headroom <= 0.9:
        verdict = (
            f"COMPUTE-BOUND — the camera offers {camera_fps:.0f} fps but the pipeline "
            f"sustains only {pipeline_fps:.0f} fps ('{dominant}' is the largest stage). "
            f"A lighter model, a smaller input, or the GPU delegate is where the gain is."
        )
    else:
        verdict = (
            f"BALANCED — camera {camera_fps:.0f} fps vs pipeline {pipeline_fps:.0f} fps; "
            f"neither side has meaningful headroom."
        )

    return {
        "camera_baseline_fps": camera_fps,
        "camera_baseline_frames": baseline.get("frames", 0),
        "camera_read_ms": baseline.get("read_ms", {}),
        "pipeline_capable_fps": pipeline_fps,
        "achieved_fps": throughput_fps,
        "realtime_ratio": ratio,
        "headroom": headroom,
        "dominant_stage": dominant,
        "frames_skipped": skipped,
        "verdict": verdict,
    }


__all__ = ["run_benchmark", "run_live_benchmark", "format_report"]
