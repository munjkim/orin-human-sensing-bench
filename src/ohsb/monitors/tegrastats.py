"""tegrastats-backed power / utilisation monitor.

``tegrastats`` prints one line per interval containing per-rail power in
milliwatts as ``NAME cur/avg``. We keep only the *current* value: the avg
column is tegrastats' own running average since process start, which would
smear the warmup phase into the measured window.

Rail names differ across Orin modules (AGX exposes ``VDD_GPU_SOC`` /
``VDD_CPU_CV`` / ``VIN_SYS_5V0``; Orin Nano/NX expose ``VDD_IN`` /
``VDD_CPU_GPU_CV`` / ``VDD_SOC``), so rails are discovered from the output
rather than hardcoded.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List

from ..metrics import summarize
from .base import Monitor
from .jtop_monitor import total_rail  # shared rail selection

_RE_POWER = re.compile(r"\b([A-Z][A-Z0-9_]*)\s+(\d+)mW/(\d+)mW")
_RE_RAM = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
_RE_GPU = re.compile(r"\bGR3D_FREQ\s+\[?(\d+)%")
_RE_CPU_BLOCK = re.compile(r"\bCPU\s+\[([^\]]*)\]")
_RE_CPU_CORE = re.compile(r"(\d+)%@(\d+)")
_RE_TEMP = re.compile(r"\b([a-zA-Z0-9_]+)@(-?[\d.]+)C")



def parse_line(line: str) -> Dict[str, Any]:
    """Parse one tegrastats line into a flat sample dict.

    Tolerant by design — JetPack releases add and reorder fields, and a
    partially understood line is worth more than a dropped one.
    """
    sample: Dict[str, Any] = {"power_mw": {}, "temp_c": {}}

    for name, cur, _avg in _RE_POWER.findall(line):
        sample["power_mw"][name] = int(cur)

    ram = _RE_RAM.search(line)
    if ram:
        sample["ram_used_mb"] = int(ram.group(1))
        sample["ram_total_mb"] = int(ram.group(2))

    gpu = _RE_GPU.search(line)
    if gpu:
        sample["gpu_util_pct"] = int(gpu.group(1))

    cpu = _RE_CPU_BLOCK.search(line)
    if cpu:
        loads = [int(m.group(1)) for m in _RE_CPU_CORE.finditer(cpu.group(1))]
        if loads:
            sample["cpu_util_pct"] = sum(loads) / len(loads)
            sample["cpu_util_pct_per_core"] = loads

    for name, value in _RE_TEMP.findall(line):
        # Power rails are matched as NAME<space>...mW; temps are name@NN.NC.
        sample["temp_c"][name] = float(value)

    return sample


class TegrastatsMonitor(Monitor):
    name = "tegrastats"

    def __init__(self, cfg):
        self.cfg = cfg
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._samples: List[Dict[str, Any]] = []
        self._error: str = ""
        self._t_start = 0.0
        self._t_stop = 0.0

    # -- lifecycle ---------------------------------------------------------
    def _command(self) -> List[str]:
        cmd = [self.cfg.binary, "--interval", str(int(self.cfg.interval_ms))]
        if self.cfg.sudo:
            # -n: never prompt. A passwordless sudo rule is the supported setup;
            # see docs/orin-setup.md. Without it we degrade to no power data.
            cmd = ["sudo", "-n", *cmd]
        return cmd

    @staticmethod
    def is_available(cfg) -> bool:
        binary = cfg.binary
        return binary.startswith("/") or shutil.which(binary) is not None

    def start(self) -> None:
        binary = self.cfg.binary
        if shutil.which(binary) is None and not binary.startswith("/"):
            self._error = f"{binary} not found on PATH (not a Jetson?)"
            return
        try:
            self._proc = subprocess.Popen(  # noqa: S603
                self._command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._error = f"failed to start tegrastats: {exc}"
            return

        self._t_start = time.perf_counter()
        self._thread = threading.Thread(target=self._pump, name="tegrastats", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            sample = parse_line(line)
            if sample.get("power_mw") or "gpu_util_pct" in sample:
                sample["t"] = time.perf_counter() - self._t_start
                self._samples.append(sample)

    def stop(self) -> None:
        self._t_stop = time.perf_counter()
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                _, stderr = self._proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
                _, stderr = self._proc.communicate()
            if not self._samples and stderr and not self._error:
                self._error = stderr.strip().splitlines()[-1] if stderr.strip() else "no output"
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # -- reporting ---------------------------------------------------------
    @property
    def duration_s(self) -> float:
        return max(0.0, self._t_stop - self._t_start)

    def summary(self) -> Dict[str, Any]:
        if not self._samples:
            return {
                "available": False,
                "backend": self.name,
                "reason": self._error or "no samples collected",
            }

        rails: Dict[str, List[float]] = defaultdict(list)
        gpu, cpu, ram, temps = [], [], [], defaultdict(list)
        for s in self._samples:
            for rail, mw in s["power_mw"].items():
                rails[rail].append(mw)
            if "gpu_util_pct" in s:
                gpu.append(s["gpu_util_pct"])
            if "cpu_util_pct" in s:
                cpu.append(s["cpu_util_pct"])
            if "ram_used_mb" in s:
                ram.append(s["ram_used_mb"])
            for name, value in s["temp_c"].items():
                temps[name].append(value)

        out: Dict[str, Any] = {
            "available": True,
            "backend": self.name,
            "samples": len(self._samples),
            "interval_ms": self.cfg.interval_ms,
            "duration_s": self.duration_s,
            "power_mw": {rail: summarize(v, unit="mW") for rail, v in sorted(rails.items())},
        }
        if gpu:
            out["gpu_util_pct"] = summarize(gpu, unit="%")
        if cpu:
            out["cpu_util_pct"] = summarize(cpu, unit="%")
        if ram:
            out["ram_used_mb"] = summarize(ram, unit="MB")
        if temps:
            out["temp_c"] = {k: summarize(v, unit="C") for k, v in sorted(temps.items())}

        rail = total_rail(rails)
        if rail:
            out["total_rail"] = rail
            out["avg_total_power_mw"] = out["power_mw"][rail]["mean"]
        return out

    def samples(self) -> Dict[str, Any]:
        return {"tegrastats": self._samples}
