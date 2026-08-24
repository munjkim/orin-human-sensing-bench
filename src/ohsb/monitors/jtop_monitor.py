"""jetson-stats (jtop) power / utilisation monitor.

Preferred over tegrastats on-board: jtop talks to a root daemon, so sampling
needs no sudo from us, and it exposes per-rail power as structured data
instead of a text line we have to regex.

The payload schema has shifted between jetson-stats major versions, and the
board in ``jetson-orin-nano-spec/`` runs a newer jtop than this code was
written against. Every field is therefore read through tolerant extractors
rather than a fixed path, and ``ohsb doctor --dump-jtop`` prints the raw
structure so the mapping can be pinned against the actual board.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from ..metrics import summarize
from .base import Monitor

_IMPORT_HINT = (
    "jetson-stats is not importable. On the Orin: sudo pip3 install -U jetson-stats "
    "&& sudo systemctl restart jtop.service (then re-login)."
)


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def as_mapping(value: Any) -> Dict[str, Any]:
    """Return a plain dict for anything dict-like, else {}.

    Not every jtop field is a real ``dict``. On the Orin Nano running
    jetson-stats 7.x, ``power``, ``cpu`` and ``temperature`` come back as
    dicts but ``gpu`` and ``memory`` are custom mapping types whose repr
    merely looks like one. An ``isinstance(x, dict)`` guard silently drops
    them, which cost us GPU utilisation, GPU frequency and RAM in every run
    until a `doctor --dump-jtop` from the board showed it.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    keys = getattr(value, "keys", None)
    if callable(keys) and hasattr(value, "__getitem__"):
        try:
            return {k: value[k] for k in keys()}
        except Exception:
            return {}
    return {}


def _dig(node: Any, *keys: str) -> Any:
    """Return the first present key from a mapping, else None."""
    node = as_mapping(node)
    for key in keys:
        if key in node:
            return node[key]
    return None


def extract_power_mw(power: Any) -> Dict[str, float]:
    """Pull ``{rail_name: milliwatts}`` out of jtop's power payload.

    Handles the shapes seen across jetson-stats versions:
      * ``{"rail": {"VDD_IN": {"power": 4900, ...}}, "tot": {"power": 4900}}``
      * ``{"VDD_IN": {"power": 4900}, ...}``
      * ``{"VDD_IN": 4900, ...}``
    """
    if not isinstance(power, dict):
        return {}

    power = as_mapping(power)
    if not power:
        return {}

    rails = as_mapping(_dig(power, "rail", "rails"))
    container = rails or power

    out: Dict[str, float] = {}
    for name, entry in container.items():
        if name in ("tot", "total", "rail", "rails"):
            continue
        value = _as_float(entry)
        if value is None:
            value = _as_float(_dig(entry, "power", "curr", "cur", "inst"))
        if value is not None:
            out[str(name)] = value

    # The module input rail lives under `tot`, not in `rail` — on the Orin
    # Nano `rail` holds only VDD_CPU_GPU_CV and VDD_SOC. `tot.name` carries
    # the real rail name (VDD_IN here), so use it rather than inventing one:
    # a result that says "TOTAL" cannot be compared against a tegrastats run
    # that says "VDD_IN".
    total = _dig(power, "tot", "total")
    total_map = as_mapping(total)
    total_mw = _as_float(total)
    if total_mw is None:
        total_mw = _as_float(_dig(total_map, "power", "curr", "cur", "inst"))
    if total_mw is not None:
        out.setdefault(str(total_map.get("name") or "TOTAL"), total_mw)
    return out


def extract_gpu(gpu: Any) -> Dict[str, float]:
    """Return ``{"load_pct": ..., "freq_khz": ...}`` from jtop's gpu payload."""
    out: Dict[str, float] = {}
    gpu = as_mapping(gpu)
    if not gpu:
        return out
    # Either {"ga10b": {...}} (device-keyed) or the inner dict directly.
    candidates = [as_mapping(v) for v in gpu.values()]
    for node in [gpu, *[c for c in candidates if c]]:
        status = as_mapping(_dig(node, "status")) or node
        load = _as_float(_dig(status, "load", "val", "gpu"))
        if load is not None and "load_pct" not in out:
            out["load_pct"] = load
        freq = _dig(node, "freq")
        freq_map = as_mapping(freq)
        cur = _as_float(_dig(freq_map, "cur", "current")) if freq_map else _as_float(freq)
        if cur is not None and "freq_khz" not in out:
            out["freq_khz"] = cur
        if freq_map:
            # This is the pinned/scaling signal that actually works on this
            # board: jetson-stats reads nvhost_podgov's own min/max, which
            # collapse to a point when jetson_clocks pins the GPU. The
            # generic /sys/class/devfreq/.../{min_freq,max_freq} sysfs files
            # do not — they read 624750000/624750000 whether or not the GPU
            # is pinned, which is why platform.py no longer trusts them for
            # this. See docs/orin-setup.md.
            lo = _as_float(_dig(freq_map, "min"))
            hi = _as_float(_dig(freq_map, "max"))
            if lo is not None and "freq_min_khz" not in out:
                out["freq_min_khz"] = lo
            if hi is not None and "freq_max_khz" not in out:
                out["freq_max_khz"] = hi
        # 3d_scaling stays True either way on this board — it does not track
        # jetson_clocks state, only whether autoscaling is compiled in.
        scaling = _dig(status, "3d_scaling")
        if isinstance(scaling, bool) and "scaling_3d" not in out:
            out["scaling_3d"] = float(scaling)
    return out


def extract_cpu_pct(cpu: Any) -> Optional[float]:
    cpu = as_mapping(cpu)
    if not cpu:
        return None
    total = _dig(cpu, "total")
    if as_mapping(total):
        total = as_mapping(total)
        idle = _as_float(_dig(total, "idle"))
        if idle is not None:
            return max(0.0, 100.0 - idle)
        user = _as_float(_dig(total, "user")) or 0.0
        system = _as_float(_dig(total, "system")) or 0.0
        return user + system
    direct = _as_float(total)
    return direct


def extract_ram_mb(memory: Any) -> Optional[float]:
    ram = as_mapping(_dig(memory, "RAM", "ram"))
    used = _as_float(_dig(ram, "used")) if ram else None
    if used is None:
        return None
    # jtop reports RAM in kB.
    return used / 1024.0


def extract_temps(temperature: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, entry in as_mapping(temperature).items():
        value = _as_float(entry)
        if value is None:
            # Offline sensors report -256.0; recording that as a temperature
            # would drag every summary's min into nonsense.
            if as_mapping(entry).get("online") is False:
                continue
            value = _as_float(_dig(entry, "temp", "value"))
        if value is not None:
            out[str(name)] = value
    return out


class JtopMonitor(Monitor):
    name = "jtop"

    def __init__(self, cfg):
        self.cfg = cfg
        self._thread = None
        self._stop = threading.Event()
        self._samples: List[Dict[str, Any]] = []
        self._error = ""
        self._t_start = 0.0
        self._t_stop = 0.0
        self._raw_first: Dict[str, Any] = {}

    @staticmethod
    def is_available() -> bool:
        try:
            import jtop  # noqa: F401
        except Exception:
            return False
        return True

    def start(self) -> None:
        if not self.is_available():
            self._error = _IMPORT_HINT
            return
        self._t_start = time.perf_counter()
        self._thread = threading.Thread(target=self._pump, name="jtop", daemon=True)
        self._thread.start()
        # Give the daemon a moment to hand over the first payload so a short
        # run does not report zero samples.
        deadline = time.perf_counter() + 2.0
        while not self._samples and not self._error and time.perf_counter() < deadline:
            time.sleep(0.02)

    def _pump(self) -> None:
        try:
            from jtop import jtop as JtopClient
        except Exception as exc:  # pragma: no cover - guarded by is_available
            self._error = f"{_IMPORT_HINT} ({exc})"
            return
        try:
            with JtopClient(interval=self.cfg.interval_ms / 1000.0) as board:
                while board.ok() and not self._stop.is_set():
                    self._samples.append(self._snapshot(board))
        except Exception as exc:
            # A dead jtop.service must degrade the result, never kill the run.
            if not self._error:
                self._error = f"jtop sampling failed: {type(exc).__name__}: {exc}"

    def _snapshot(self, board) -> Dict[str, Any]:
        power = getattr(board, "power", None)
        gpu = getattr(board, "gpu", None)
        if not self._raw_first:
            self._raw_first = {
                "power": _plain(power),
                "gpu": _plain(gpu),
                "cpu": _plain(getattr(board, "cpu", None)),
                "memory": _plain(getattr(board, "memory", None)),
                "temperature": _plain(getattr(board, "temperature", None)),
            }
        sample: Dict[str, Any] = {
            "t": time.perf_counter() - self._t_start,
            "power_mw": extract_power_mw(power),
            "temp_c": extract_temps(getattr(board, "temperature", None)),
        }
        gpu_info = extract_gpu(gpu)
        if "load_pct" in gpu_info:
            sample["gpu_util_pct"] = gpu_info["load_pct"]
        if "freq_khz" in gpu_info:
            sample["gpu_freq_khz"] = gpu_info["freq_khz"]
        cpu_pct = extract_cpu_pct(getattr(board, "cpu", None))
        if cpu_pct is not None:
            sample["cpu_util_pct"] = cpu_pct
        ram = extract_ram_mb(getattr(board, "memory", None))
        if ram is not None:
            sample["ram_used_mb"] = ram
        return sample

    def stop(self) -> None:
        self._t_stop = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self._t_stop - self._t_start)

    def summary(self) -> Dict[str, Any]:
        if not self._samples:
            return {"available": False, "backend": self.name,
                    "reason": self._error or "no samples collected"}

        rails: Dict[str, List[float]] = defaultdict(list)
        gpu, gpu_f, cpu, ram = [], [], [], []
        temps: Dict[str, List[float]] = defaultdict(list)
        for s in self._samples:
            for rail, mw in s["power_mw"].items():
                rails[rail].append(mw)
            for key, bucket in (
                ("gpu_util_pct", gpu),
                ("gpu_freq_khz", gpu_f),
                ("cpu_util_pct", cpu),
                ("ram_used_mb", ram),
            ):
                if key in s:
                    bucket.append(s[key])
            for name, value in s["temp_c"].items():
                temps[name].append(value)

        out: Dict[str, Any] = {
            "available": True,
            "backend": self.name,
            "samples": len(self._samples),
            "interval_ms": self.cfg.interval_ms,
            "duration_s": self.duration_s,
            "power_mw": {r: summarize(v, unit="mW") for r, v in sorted(rails.items())},
        }
        for key, bucket, unit in (
            ("gpu_util_pct", gpu, "%"),
            ("gpu_freq_khz", gpu_f, "kHz"),
            ("cpu_util_pct", cpu, "%"),
            ("ram_used_mb", ram, "MB"),
        ):
            if bucket:
                out[key] = summarize(bucket, unit=unit)
        if temps:
            out["temp_c"] = {k: summarize(v, unit="C") for k, v in sorted(temps.items())}

        rail = total_rail(rails)
        if rail:
            out["total_rail"] = rail
            out["avg_total_power_mw"] = out["power_mw"][rail]["mean"]
        return out

    def samples(self) -> Dict[str, Any]:
        return {"jtop": self._samples, "jtop_raw_first_sample": self._raw_first}


#: Rails representing total module input power, most specific first.
#: Orin Nano exposes VDD_IN / VDD_CPU_GPU_CV / VDD_SOC.
TOTAL_RAIL_CANDIDATES = ("VDD_IN", "VIN_SYS_5V0", "POM_5V_IN", "TOTAL")


def total_rail(rails) -> str:
    for candidate in TOTAL_RAIL_CANDIDATES:
        if candidate in rails:
            return candidate
    return ""


def _plain(value: Any) -> Any:
    """Best-effort conversion of a jtop payload to JSON-safe primitives."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    mapping = as_mapping(value)
    if mapping:
        return {str(k): _plain(v) for k, v in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def dump_schema() -> Dict[str, Any]:
    """One raw jtop payload, for pinning the extractors against a real board."""
    try:
        from jtop import jtop as JtopClient
    except Exception as exc:
        return {"available": False, "reason": f"{_IMPORT_HINT} ({exc})"}
    try:
        with JtopClient(interval=0.5) as board:
            if not board.ok():
                return {"available": False, "reason": "jtop client not ready"}
            return {
                "available": True,
                "power": _plain(getattr(board, "power", None)),
                "gpu": _plain(getattr(board, "gpu", None)),
                "cpu": _plain(getattr(board, "cpu", None)),
                "memory": _plain(getattr(board, "memory", None)),
                "temperature": _plain(getattr(board, "temperature", None)),
                "board": _plain(getattr(board, "board", None)),
            }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def quick_gpu_state() -> Optional[Dict[str, float]]:
    """One fast jtop sample of GPU frequency state.

    Used to decide whether jetson_clocks has pinned the GPU. Returns None on
    any failure — jtop not installed, the daemon not running, the read
    timing out — so the caller can report "unknown" rather than guessing.
    That distinction mattered here: this project previously inferred pinning
    from /sys/class/devfreq, which reads identically whether the GPU is
    pinned or scaling on this hardware, and the false "pinned" reading
    suppressed the single most important reproducibility warning.
    """
    try:
        from jtop import jtop as JtopClient
    except Exception:
        return None
    try:
        with JtopClient(interval=0.2) as board:
            if not board.ok():
                return None
            return extract_gpu(getattr(board, "gpu", None))
    except Exception:
        return None
