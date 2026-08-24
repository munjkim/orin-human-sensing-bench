"""Platform snapshot recorded with every result.

On this board the numbers mean nothing without the machine state that
produced them: the Orin Nano dev kit boots at 15W with ``jetson_clocks``
inactive and GPU DVFS active (306 MHz idle, 624 MHz boost), so the same
config can differ by well over 2x run to run. Everything here exists to make
a stored result self-describing — and to let ``ohsb doctor`` warn before you
collect numbers you cannot reproduce.
"""

from __future__ import annotations

import os
import platform as _platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read(path: str) -> Optional[str]:
    try:
        with open(path, errors="replace") as fh:
            return fh.read().strip("\x00").strip()
    except OSError:
        return None


def _run(cmd: List[str], timeout: float = 5.0) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout or out.stderr).strip() or None


def is_jetson() -> bool:
    model = _read("/proc/device-tree/model") or ""
    return "NVIDIA" in model and ("Jetson" in model or "Orin" in model or "Tegra" in model)


def board_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model": _read("/proc/device-tree/model"),
        "is_jetson": is_jetson(),
    }
    # /etc/nv_tegra_release: "# R35 (release), REVISION: 3.1, GCID: ..."
    release = _read("/etc/nv_tegra_release")
    if release:
        first = release.splitlines()[0]
        info["l4t_raw"] = first
        major = re.search(r"R(\d+)", first)
        minor = re.search(r"REVISION:\s*([\d.]+)", first)
        if major and minor:
            info["l4t"] = f"{major.group(1)}.{minor.group(1)}"
    return info


def power_mode() -> Dict[str, Any]:
    """nvpmodel mode plus jetson_clocks state — the two reproducibility knobs."""
    out: Dict[str, Any] = {}
    raw = _run(["nvpmodel", "-q"])
    if raw:
        out["nvpmodel_raw"] = raw
        name = re.search(r"NV Power Mode:\s*(.+)", raw)
        mode = re.search(r"^\s*(\d+)\s*$", raw, re.MULTILINE)
        if name:
            out["nv_power_mode"] = name.group(1).strip()
        if mode:
            out["nvpmodel_id"] = int(mode.group(1))

    clocks = _run(["jetson_clocks", "--show"])
    if clocks:
        # Reported rather than parsed field by field: the --show format varies
        # across L4T releases, and the raw text is what you would diff anyway.
        out["jetson_clocks_show"] = clocks
    out["jetson_clocks_active"] = _jetson_clocks_active(clocks)
    return out


def _jetson_clocks_active(show_output: Optional[str]) -> Optional[bool]:
    """True when the GPU is pinned at its max frequency.

    ``jetson_clocks`` has no status flag on L4T 35, so we infer it the way
    the tool itself does: min == max on the GPU devfreq rail.
    """
    gpu = gpu_freq()
    lo, hi = gpu.get("min_freq_hz"), gpu.get("max_freq_hz")
    if lo is not None and hi is not None:
        return lo == hi
    if show_output:
        match = re.search(r"GPU MinFreq=(\d+)\s+MaxFreq=(\d+)", show_output)
        if match:
            return match.group(1) == match.group(2)
    return None


def _devfreq_gpu_dir() -> Optional[Path]:
    root = Path("/sys/class/devfreq")
    if not root.is_dir():
        return None
    for entry in sorted(root.iterdir()):
        name = (_read(str(entry / "device/of_node/name")) or entry.name).lower()
        if "gpu" in name or "gv11b" in name or "ga10b" in name:
            return entry
    return None


def gpu_freq() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    node = _devfreq_gpu_dir()
    if node is None:
        return out
    out["devfreq"] = node.name
    for key, filename in (
        ("cur_freq_hz", "cur_freq"),
        ("min_freq_hz", "min_freq"),
        ("max_freq_hz", "max_freq"),
    ):
        value = _read(str(node / filename))
        if value and value.isdigit():
            out[key] = int(value)
    governor = _read(str(node / "governor"))
    if governor:
        out["governor"] = governor
    return out


def cpu_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "arch": _platform.machine(),
        "logical_cores": os.cpu_count(),
    }
    governors, max_freqs, online = set(), set(), 0
    root = Path("/sys/devices/system/cpu")
    if root.is_dir():
        for cpu_dir in sorted(root.glob("cpu[0-9]*")):
            policy = cpu_dir / "cpufreq"
            if not policy.is_dir():
                continue
            online += 1
            gov = _read(str(policy / "scaling_governor"))
            mx = _read(str(policy / "scaling_max_freq"))
            if gov:
                governors.add(gov)
            if mx and mx.isdigit():
                max_freqs.add(int(mx))
    if governors:
        out["scaling_governor"] = sorted(governors)
    if max_freqs:
        out["scaling_max_freq_khz"] = sorted(max_freqs)
    if online:
        out["cores_with_cpufreq"] = online
    return out


def memory_info() -> Dict[str, Any]:
    """Total/available RAM in MB.

    Worth recording on this board: the module has ~6.3 GB usable shared
    between CPU and iGPU, and a running desktop session can already hold
    ~3 GB of it — enough to change which model bundles fit.
    """
    text = _read("/proc/meminfo")
    if not text:
        return {}
    values = {}
    for line in text.splitlines():
        match = re.match(r"(\w+):\s+(\d+) kB", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    out = {}
    if "MemTotal" in values:
        out["total_mb"] = round(values["MemTotal"] / 1024)
    if "MemAvailable" in values:
        out["available_mb"] = round(values["MemAvailable"] / 1024)
    return out


def library_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python": _platform.python_version(),
        "python_executable": sys.executable,
    }
    for module in ("numpy", "mediapipe", "cv2", "jtop"):
        try:
            mod = __import__(module)
            out[module] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[module] = None
    cuda = _read("/usr/local/cuda/version.txt")
    if cuda:
        out["cuda"] = cuda
    else:
        nvcc = _run(["nvcc", "--version"])
        if nvcc:
            match = re.search(r"release ([\d.]+)", nvcc)
            out["cuda"] = match.group(1) if match else nvcc.splitlines()[-1]
    return out


def snapshot() -> Dict[str, Any]:
    """Full platform record embedded in every result file."""
    return {
        "os": {
            "system": _platform.system(),
            "release": _platform.release(),
            "distro": _os_release_name() or _platform.platform(),
        },
        "board": board_info(),
        "power_mode": power_mode(),
        "gpu": gpu_freq(),
        "cpu": cpu_info(),
        "memory": memory_info(),
        "libraries": library_versions(),
        "hostname": _platform.node(),
    }


def _os_release_name() -> Optional[str]:
    text = _read("/etc/os-release") or ""
    match = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', text, re.MULTILINE)
    return match.group(1) if match else None


def reproducibility_warnings(snap: Optional[Dict[str, Any]] = None) -> List[str]:
    """Conditions that make a measurement untrustworthy.

    These are warnings, not errors: a quick exploratory run with DVFS on is
    fine, as long as you know that is what you collected.
    """
    snap = snap or snapshot()
    warnings: List[str] = []

    if not snap["board"].get("is_jetson"):
        warnings.append("not running on a Jetson — power data will be unavailable")
        return warnings

    if snap["power_mode"].get("jetson_clocks_active") is False:
        warnings.append(
            "jetson_clocks is inactive: GPU/CPU frequencies will scale during the run. "
            "Run `sudo jetson_clocks` to pin them before collecting comparable numbers."
        )
    gpu = snap.get("gpu", {})
    if gpu.get("governor") and gpu.get("governor") != "userspace":
        warnings.append(
            f"GPU devfreq governor is {gpu['governor']!r} (DVFS active); "
            "frequency will vary with load"
        )
    govs = snap["cpu"].get("scaling_governor") or []
    if any(g != "performance" for g in govs):
        warnings.append(f"CPU scaling governor is {govs}, not 'performance'")

    mode = snap["power_mode"].get("nv_power_mode")
    if mode:
        warnings.append(f"nvpmodel mode is {mode!r} — record it; it caps achievable clocks")

    mem = snap.get("memory", {})
    if mem.get("available_mb") is not None and mem["available_mb"] < 2048:
        warnings.append(
            f"only {mem['available_mb']} MB RAM available; "
            "close the desktop session (`sudo systemctl isolate multi-user.target`) "
            "before benchmarking"
        )
    return warnings
