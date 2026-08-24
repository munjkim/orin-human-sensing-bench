"""Command line interface: ``ohsb run | list | doctor | report``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import __version__
from .config import BenchmarkConfig, ConfigError
from .platform import reproducibility_warnings, snapshot
from .results import format_report, load_report
from .runner import run_benchmark, run_live_benchmark


def _parse_set(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Parse ``--set a.b=value`` into a dotted override map.

    Values go through the YAML scalar parser so ``true``, ``0.5`` and ``3``
    arrive as the types the config dataclasses expect.
    """
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


# -- commands --------------------------------------------------------------

def _load_config(args):
    cfg = BenchmarkConfig.load(args.config)
    overrides = _parse_set(args.set)
    if overrides:
        cfg = cfg.apply_overrides(overrides)
        cfg.run.tags.setdefault("config", str(args.config))
    return cfg


def cmd_run(args) -> int:
    try:
        cfg = _load_config(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    # Flag overrides are mirrored into cfg.raw so the stored result — and the
    # filename derived from it — describe the run that actually happened.
    for value, section, key in (
        (args.name, "run", "name"),
        (args.iterations, "run", "iterations"),
        (args.warmup, "run", "warmup"),
        (args.output, "output", "dir"),
    ):
        if value is None:
            continue
        setattr(getattr(cfg, section), key, value)
        cfg.raw.setdefault(section, {})[key] = value

    say = (lambda msg: print(f"... {msg}", file=sys.stderr)) if not args.quiet else None

    for warning in reproducibility_warnings():
        print(f"! {warning}", file=sys.stderr)

    try:
        report = run_benchmark(cfg, progress=say)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print()
    print(format_report(report))

    if not args.no_save:
        path = report.write(cfg.output.dir)
        print()
        print(f"saved: {path}")
    return 0


def cmd_live(args) -> int:
    """Benchmark the end-to-end USB camera pipeline."""
    try:
        cfg = _load_config(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    for value, section, key in (
        (args.name, "run", "name"),
        (args.iterations, "run", "iterations"),
        (args.warmup, "run", "warmup"),
        (args.device, "camera", "device"),
        (args.output, "output", "dir"),
    ):
        if value is None:
            continue
        setattr(getattr(cfg, section), key, value)
        cfg.raw.setdefault(section, {})[key] = value

    say = (lambda msg: print(f"... {msg}", file=sys.stderr)) if not args.quiet else None
    for warning in reproducibility_warnings():
        print(f"! {warning}", file=sys.stderr)

    try:
        report = run_live_benchmark(cfg, progress=say)
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print()
    print(format_report(report))

    if not args.no_save:
        path = report.write(cfg.output.dir)
        print()
        print(f"saved: {path}")
    return 0


def cmd_cameras(args) -> int:
    """Report what each connected camera actually supports."""
    from .camera_probe import format_modes, list_devices, probe

    devices = [args.device] if args.device else list_devices()
    if not devices:
        print("no /dev/video* devices found — is the webcam plugged in?", file=sys.stderr)
        return 1

    for device in devices:
        try:
            info = probe(device)
        except ImportError as exc:
            print(f"{device}: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(info, indent=2, default=str))
        else:
            print(format_modes(info))
            print()
    return 0


def cmd_list(args) -> int:
    from .sources import available_sources
    from .tasks import available_tasks

    print("tasks:")
    for name in available_tasks():
        print(f"  {name}")
    print("sources:")
    for name in available_sources():
        print(f"  {name}")
    # Not in the offline registry: a live camera cannot be pre-decoded, so it
    # has its own runner rather than pretending to be a replayable source.
    print("  webcam            (live capture — use `ohsb live`, not `ohsb run`)")
    print("power backends:")
    for name in ("auto", "jtop", "tegrastats", "none"):
        print(f"  {name}")

    configs = sorted(Path("configs").glob("*.yaml")) if Path("configs").is_dir() else []
    if configs:
        print("configs:")
        for path in configs:
            print(f"  {path}")
    return 0


def cmd_doctor(args) -> int:
    if args.dump_jtop:
        from .monitors.jtop_monitor import dump_schema

        print(json.dumps(dump_schema(), indent=2, default=str))
        return 0

    snap = snapshot()
    print(json.dumps(snap, indent=2, default=str))
    print()

    checks: List[tuple] = []

    try:
        import mediapipe  # noqa: F401

        checks.append(("mediapipe importable", True, getattr(mediapipe, "__version__", "?")))
    except Exception as exc:
        checks.append(("mediapipe importable", False, str(exc).splitlines()[0]))

    from .config import PowerConfig
    from .monitors.jtop_monitor import JtopMonitor
    from .monitors.tegrastats import TegrastatsMonitor

    checks.append(("jtop available", JtopMonitor.is_available(), "power backend (preferred)"))
    checks.append((
        "tegrastats available",
        TegrastatsMonitor.is_available(PowerConfig()),
        "power backend (fallback, needs sudo)",
    ))

    models = sorted(Path("models").glob("*")) if Path("models").is_dir() else []
    models = [m for m in models if m.suffix in (".task", ".tflite")]
    checks.append(("model bundles present", bool(models), f"{len(models)} found in models/"))

    print("checks:")
    for label, ok, detail in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label:<24} {detail}")
    if models:
        for model in models:
            print(f"        - {model.name}")

    warnings = reproducibility_warnings(snap)
    if warnings:
        print()
        print("reproducibility warnings:")
        for warning in warnings:
            print(f"  ! {warning}")

    return 0 if all(ok for _, ok, _ in checks) else 1


def cmd_report(args) -> int:
    """Compare stored result files as one table."""
    paths: List[Path] = []
    for target in args.paths:
        path = Path(target)
        paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    if not paths:
        print("no result files found", file=sys.stderr)
        return 1

    header = (
        f"{'run':<30} {'del':<4} {'res':>9} "
        f"{'p50':>7} {'p95':>7} {'fps':>7} {'cam':>7} {'W':>6} {'mJ/f':>8}  bottleneck"
    )
    print(header)
    print("-" * len(header))
    for path in paths:
        try:
            data = load_report(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{path}: unreadable ({exc})", file=sys.stderr)
            continue
        task = data.get("task", {})
        source = data.get("source", {})
        res = f"{source.get('width', '?')}x{source.get('height', '?')}"
        for run in data.get("runs", []):
            lat = run.get("latency_ms", {})
            energy = run.get("energy", {})
            live = run.get("live", {})
            # `cam` is the capture-only baseline; a live fps at or near it
            # means the camera is the ceiling, not the model.
            print(
                f"{run.get('name', '?')[:30]:<30} "
                f"{str(task.get('delegate'))[:4]:<4} "
                f"{res:>9} "
                f"{lat.get('p50', float('nan')):>7.2f} "
                f"{lat.get('p95', float('nan')):>7.2f} "
                f"{run.get('throughput_fps', float('nan')):>7.1f} "
                f"{live.get('camera_baseline_fps', float('nan')):>7.1f} "
                f"{energy.get('avg_power_w', float('nan')):>6.2f} "
                f"{energy.get('mj_per_frame', float('nan')):>8.1f}  "
                f"{live.get('verdict', '').split(' ')[0].strip('—') or '-'}"
            )
    return 0


# -- entrypoint ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ohsb",
        description="Human sensing inference benchmark for NVIDIA Jetson Orin",
    )
    parser.add_argument("--version", action="version", version=f"ohsb {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a benchmark from a config file")
    run.add_argument("-c", "--config", required=True, help="path to a YAML config")
    run.add_argument(
        "-s", "--set", action="append", metavar="KEY=VALUE",
        help="override a config value, e.g. --set task.delegate=gpu (repeatable)",
    )
    run.add_argument("-n", "--name", help="override run.name")
    run.add_argument("-i", "--iterations", type=int, help="override run.iterations")
    run.add_argument("-w", "--warmup", type=int, help="override run.warmup")
    run.add_argument("-o", "--output", help="override output.dir")
    run.add_argument("--no-save", action="store_true", help="print only, do not write a file")
    run.add_argument("-q", "--quiet", action="store_true", help="suppress progress lines")
    run.set_defaults(func=cmd_run)

    live = sub.add_parser(
        "live", help="benchmark the end-to-end USB camera pipeline (capture->convert->infer)"
    )
    live.add_argument("-c", "--config", required=True, help="path to a YAML config")
    live.add_argument(
        "-s", "--set", action="append", metavar="KEY=VALUE",
        help="override a config value, e.g. --set camera.fourcc=YUYV (repeatable)",
    )
    live.add_argument("-n", "--name", help="override run.name")
    live.add_argument("-i", "--iterations", type=int, help="frames to measure")
    live.add_argument("-w", "--warmup", type=int, help="override run.warmup")
    live.add_argument("-d", "--device", help="camera device index or path")
    live.add_argument("-o", "--output", help="override output.dir")
    live.add_argument("--no-save", action="store_true", help="print only, do not write a file")
    live.add_argument("-q", "--quiet", action="store_true", help="suppress progress lines")
    live.set_defaults(func=cmd_live)

    cameras = sub.add_parser("cameras", help="list connected cameras and their supported modes")
    cameras.add_argument("-d", "--device", help="probe only this device (e.g. /dev/video0)")
    cameras.add_argument("--json", action="store_true", help="emit raw JSON")
    cameras.set_defaults(func=cmd_cameras)

    listing = sub.add_parser("list", help="list registered tasks, sources and configs")
    listing.set_defaults(func=cmd_list)

    doctor = sub.add_parser("doctor", help="report platform state and check prerequisites")
    doctor.add_argument(
        "--dump-jtop", action="store_true",
        help="print one raw jtop payload (to pin the power extractors to this board)",
    )
    doctor.set_defaults(func=cmd_doctor)

    report = sub.add_parser("report", help="compare stored result files")
    report.add_argument("paths", nargs="+", help="result .json files or directories")
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
