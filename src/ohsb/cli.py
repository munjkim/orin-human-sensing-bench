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


def _doctor_checks() -> List[dict]:
    """Prerequisite checks, as data so both renderers share one source."""
    from .config import PowerConfig
    from .monitors.jtop_monitor import JtopMonitor
    from .monitors.tegrastats import TegrastatsMonitor

    checks = []

    try:
        import mediapipe

        checks.append({"name": "mediapipe importable", "ok": True,
                       "detail": getattr(mediapipe, "__version__", "?")})
    except Exception as exc:
        checks.append({"name": "mediapipe importable", "ok": False,
                       "detail": str(exc).splitlines()[0]})

    checks.append({"name": "jtop available", "ok": JtopMonitor.is_available(),
                   "detail": "power backend (preferred)"})
    checks.append({"name": "tegrastats available",
                   "ok": TegrastatsMonitor.is_available(PowerConfig()),
                   "detail": "power backend (fallback, needs sudo)"})

    models = sorted(Path("models").glob("*")) if Path("models").is_dir() else []
    models = [m for m in models if m.suffix in (".task", ".tflite")]
    checks.append({"name": "model bundles present", "ok": bool(models),
                   "detail": f"{len(models)} found in models/",
                   "items": [m.name for m in models]})
    return checks


def cmd_doctor(args) -> int:
    if args.dump_jtop:
        from .monitors.jtop_monitor import dump_schema

        print(json.dumps(dump_schema(), indent=2, default=str))
        return 0

    snap = snapshot()
    checks = _doctor_checks()
    warnings = reproducibility_warnings(snap)
    ok = all(c["ok"] for c in checks)

    if args.json:
        print(json.dumps(
            {"platform": snap, "checks": checks, "warnings": warnings, "ok": ok},
            indent=2, default=str,
        ))
        return 0 if ok else 1

    print(json.dumps(snap, indent=2, default=str))
    print()
    print("checks:")
    for check in checks:
        status = "ok" if check["ok"] else "FAIL"
        print(f"  [{status:<4}] {check['name']:<24} {check['detail']}")
        for item in check.get("items", []):
            print(f"          - {item}")

    if warnings:
        print()
        print("reproducibility warnings:")
        for warning in warnings:
            print(f"  ! {warning}")

    return 0 if ok else 1


def _report_rows(paths: List[Path]):
    """Flatten result files into one row per measured run."""
    rows = []
    for path in paths:
        try:
            data = load_report(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{path}: unreadable ({exc})", file=sys.stderr)
            continue
        task = data.get("task", {})
        source = data.get("source", {})
        for run in data.get("runs", []):
            lat = run.get("latency_ms", {})
            live = run.get("live", {})
            is_noop = task.get("type") == "noop"
            rows.append({
                "name": run.get("name", "?"),
                "mode": data.get("mode", "offline"),
                "task": task.get("type"),
                "delegate": task.get("delegate"),
                "model": Path(str(task.get("model") or "-")).name,
                # noop touches no real pixels (synthetic source, no model) —
                # its resolution is not comparable to a camera run's, so the
                # table marks it n/a; markdown output adds a footnote.
                "res": ("n/a" if is_noop
                       else f"{source.get('width', '?')}x{source.get('height', '?')}"),
                "is_noop": is_noop,
                "p50": lat.get("p50", float("nan")),
                "p95": lat.get("p95", float("nan")),
                "fps": run.get("throughput_fps", float("nan")),
                # The capture-only ceiling; blank for offline runs.
                "cam": live.get("camera_baseline_fps", float("nan")),
                "watts": run.get("energy", {}).get("avg_power_w", float("nan")),
                "mj": run.get("energy", {}).get("mj_per_frame", float("nan")),
                "bottleneck": ("harness floor" if is_noop else
                              (live.get("verdict", "").split(" ")[0].strip("—") or "-")),
                "dvfs": run.get("dvfs", {}),
                "platform": data.get("platform", {}),
                "warnings": data.get("warnings", []),
            })
    return rows


def _num(value, spec=".2f", dash="-"):
    return dash if value != value else format(value, spec)


def _render_text(rows) -> str:
    header = (
        f"{'run':<30} {'del':<4} {'res':>9} "
        f"{'p50':>7} {'p95':>7} {'fps':>7} {'cam':>7} {'W':>6} {'mJ/f':>8}  bottleneck"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['name'][:30]:<30} {str(r['delegate'])[:4]:<4} {r['res']:>9} "
            f"{_num(r['p50']):>7} {_num(r['p95']):>7} {_num(r['fps'], '.1f'):>7} "
            f"{_num(r['cam'], '.1f'):>7} {_num(r['watts']):>6} {_num(r['mj'], '.1f'):>8}  "
            f"{r['bottleneck']}"
        )
    return "\n".join(lines)


def _render_markdown(rows) -> str:
    """A committable summary: the table plus the board state behind it.

    A published benchmark without its platform snapshot is not reproducible,
    so the board, power mode and clock state travel with the numbers.
    """
    lines = ["| run | task | delegate | resolution | p50 ms | p95 ms | fps | camera fps "
             "| W | mJ/frame | bottleneck |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['task']} | {r['delegate']} | {r['res']} "
            f"| {_num(r['p50'])} | {_num(r['p95'])} | {_num(r['fps'], '.1f')} "
            f"| {_num(r['cam'], '.1f')} | {_num(r['watts'])} | {_num(r['mj'], '.1f')} "
            f"| {r['bottleneck']} |"
        )

    if any(r.get("is_noop") for r in rows):
        lines += [
            "",
            "*`n/a` rows are the `noop` harness-calibration task: no camera, no model, "
            "synthetic frames only. It measures what the benchmark loop itself costs "
            "so every other row can be read against that floor — its resolution and "
            "bottleneck verdict are not comparable to a camera run's.*",
        ]

    snap = rows[0]["platform"] if rows else {}
    board = snap.get("board", {})
    mode = snap.get("power_mode", {})
    gpu = snap.get("gpu", {})
    cpu = snap.get("cpu", {})
    libs = snap.get("libraries", {})
    clocks = mode.get("jetson_clocks_active")
    clock_state = {True: "pinned", False: "INACTIVE (DVFS live)"}.get(
        clocks, "unknown (no static probe on this board is reliable — see below)"
    )
    cpu_pinned = cpu.get("cpu_pinned")
    cpu_state = {True: "pinned", False: "not pinned"}.get(cpu_pinned, "-")
    if cpu.get("cores_with_cpufreq"):
        cpu_state += f" ({cpu.get('cores_pinned', 0)}/{cpu['cores_with_cpufreq']} cores)"

    lines += [
        "",
        "## Board state",
        "",
        "| | |",
        "|---|---|",
        f"| model | {board.get('model') or '-'} |",
        f"| L4T | {board.get('l4t') or '-'} |",
        f"| power mode | {mode.get('nv_power_mode') or '-'} |",
        f"| jetson_clocks --show (pre-run, best-effort) | {clock_state} |",
        f"| GPU clock during these runs | {_dvfs_summary(rows)} |",
        f"| CPU frequency | {cpu_state} |",
        f"| GPU freq (sysfs, informational only) | {_freq(gpu)} |",
        f"| python | {libs.get('python') or '-'} |",
        f"| mediapipe | {libs.get('mediapipe') or 'not installed'} |",
        f"| opencv | {libs.get('cv2') or 'not installed'} |",
    ]

    warnings = []
    for r in rows:
        for w in r["warnings"]:
            if w not in warnings:
                warnings.append(w)
    if warnings:
        lines += ["", "## Reproducibility warnings", ""]
        lines += [f"- {w}" for w in warnings]
    return "\n".join(lines)


def _dvfs_summary(rows) -> str:
    """The trustworthy pin signal: what the GPU clock actually did per run.

    Pre-run probes (devfreq sysfs, governor name, a single jtop sample) were
    each tried and falsified on this hardware — see docs/orin-setup.md. This
    instead reports what every run's own gpu_freq_khz samples showed.
    """
    concluded = [r["dvfs"] for r in rows if r.get("dvfs", {}).get("conclusive")]
    if not concluded:
        return "not measured (no run had >=2 GPU frequency samples)"
    scaled = sum(1 for d in concluded if d["scaled"])
    steady = len(concluded) - scaled
    if scaled and steady:
        return f"varied in {scaled}/{len(concluded)} runs — mixed, check each row"
    if scaled:
        return f"varied (DVFS live) in all {len(concluded)} measured run(s)"
    return f"steady in all {len(concluded)} measured run(s)"


def _freq(gpu) -> str:
    """Report the devfreq sysfs numbers labelled as exactly that: informational.

    They do not indicate whether the GPU is pinned. On this Orin Nano
    /sys/class/devfreq/.../{min_freq,max_freq} reads 624750000/624750000
    whether jetson_clocks has run or not — confirmed with a before/after
    capture — so this row must never be read as a pinned/not judgement.
    That call belongs to the "jetson_clocks (GPU, via jtop)" row instead,
    which uses a source that actually distinguishes the two states.
    """
    lo, hi = gpu.get("min_freq_hz"), gpu.get("max_freq_hz")
    if lo is None or hi is None:
        return gpu.get("governor") or "-"
    scale = 1e6
    span = f"{hi / scale:.0f} MHz" if lo == hi else f"{lo / scale:.0f}-{hi / scale:.0f} MHz"
    cur = gpu.get("cur_freq_hz")
    if cur:
        span += f", cur {cur / scale:.0f}"
    governor = gpu.get("governor")
    return f"{span} (devfreq governor {governor})" if governor else span


def cmd_report(args) -> int:
    """Compare stored result files as one table."""
    paths: List[Path] = []
    for target in args.paths:
        path = Path(target)
        paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    if not paths:
        print("no result files found", file=sys.stderr)
        return 1

    rows = _report_rows(paths)
    if not rows:
        print("no runs found in the given files", file=sys.stderr)
        return 1

    text = _render_markdown(rows) if args.markdown else _render_text(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
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
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.add_argument(
        "--dump-jtop", action="store_true",
        help="print one raw jtop payload (to pin the power extractors to this board)",
    )
    doctor.set_defaults(func=cmd_doctor)

    report = sub.add_parser("report", help="compare stored result files")
    report.add_argument("paths", nargs="+", help="result .json files or directories")
    report.add_argument(
        "--markdown", action="store_true",
        help="render as markdown with the board state, for committing under benchmarks/",
    )
    report.add_argument("-o", "--out", help="write to this file instead of stdout")
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
