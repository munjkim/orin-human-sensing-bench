"""Regression tests against payloads recorded on the real target board.

Source: env/siblab-desktop-20260824-132626 (Orin Nano dev kit, JetPack 5.1.1,
jetson-stats 7.x, Logitech C920). Every case here corresponds to a bug that
synthetic fixtures did not catch, so these are locked to the real shapes
rather than to what the API documentation implies.
"""

from collections.abc import Mapping

import pytest

from ohsb.camera_probe import collapse_modes, parse_formats, realtime_ceiling
from ohsb.monitors.jtop_monitor import (
    as_mapping,
    extract_cpu_pct,
    extract_gpu,
    extract_power_mw,
    extract_ram_mb,
    extract_temps,
    total_rail,
)


class JtopMapping(Mapping):
    """A Mapping that is not a dict subclass.

    jetson-stats returns these for `gpu` and `memory`; its repr looks exactly
    like a dict's, which is what made the bug invisible until a raw dump from
    the board was compared field by field.
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(self._data)


# Verbatim from the board's `ohsb doctor --dump-jtop`.
POWER = {
    "rail": {
        "VDD_CPU_GPU_CV": {"volt": 5056, "curr": 112, "power": 566, "online": True, "avg": 566},
        "VDD_SOC": {"volt": 5064, "curr": 272, "power": 1377, "online": True, "avg": 1377},
    },
    "tot": {"volt": 5064, "curr": 824, "power": 4172, "online": True, "avg": 4172,
            "name": "VDD_IN"},
}

GPU = JtopMapping({
    "ga10b": {
        "type": "integrated",
        "status": {"railgate": False, "tpc_pg_mask": False, "3d_scaling": True, "load": 0.0},
        "freq": {"governor": "nvhost_podgov", "cur": 306000, "max": 624750,
                 "min": 306000, "GPC": [305965]},
        "power_control": "auto",
    }
})

MEMORY = JtopMapping({
    "RAM": {"tot": 6636128, "used": 3038940, "free": 564388, "buffers": 70496,
            "cached": 3084052, "shared": 401048, "lfb": 4},
})

CPU = {"total": {"user": 3.1691864202907567, "nice": 0.2867645533918165,
                 "system": 2.915603807627699, "idle": 92.40688317882993}}

TEMPERATURE = {
    "CPU": {"temp": 48.343, "online": True},
    "CV0": {"temp": -256.0, "online": False},
    "GPU": {"temp": 47.937, "online": True},
    "tj": {"temp": 48.343, "online": True},
}


# -- the mapping bug -------------------------------------------------------

def test_gpu_payload_is_not_a_dict():
    # Guards the premise of the bug: an isinstance(x, dict) check fails here.
    assert not isinstance(GPU, dict)
    assert isinstance(GPU, Mapping)


def test_as_mapping_sees_through_non_dict_mappings():
    assert as_mapping(GPU)["ga10b"]["status"]["load"] == 0.0
    assert as_mapping({"a": 1}) == {"a": 1}
    assert as_mapping(None) == {}
    assert as_mapping("not a mapping") == {}


def test_gpu_metrics_survive_the_custom_mapping():
    out = extract_gpu(GPU)
    assert out["load_pct"] == 0.0
    assert out["freq_khz"] == 306000
    # 3d_scaling True == DVFS still active; jetson_clocks has not pinned it.
    assert out["scaling_3d"] == 1.0


def test_ram_survives_the_custom_mapping():
    # 3038940 kB used -> ~2968 MB
    assert extract_ram_mb(MEMORY) == pytest.approx(2967.7, abs=0.5)


# -- the total-rail naming bug --------------------------------------------

def test_total_rail_keeps_its_real_name():
    out = extract_power_mw(POWER)
    # VDD_IN is only under `tot`, never in `rail` — but the result must still
    # call it VDD_IN so it lines up with a tegrastats run of the same board.
    assert out["VDD_IN"] == 4172
    assert out["VDD_CPU_GPU_CV"] == 566
    assert out["VDD_SOC"] == 1377
    assert "TOTAL" not in out
    assert "tot" not in out
    assert total_rail(out) == "VDD_IN"


def test_unnamed_total_still_falls_back():
    out = extract_power_mw({"rail": {"VDD_SOC": {"power": 100}}, "tot": {"power": 900}})
    assert out["TOTAL"] == 900
    assert total_rail(out) == "TOTAL"


# -- the rest of the payload ----------------------------------------------

def test_cpu_percent_from_idle():
    assert extract_cpu_pct(CPU) == pytest.approx(7.593, abs=0.01)


def test_offline_sensors_are_dropped():
    out = extract_temps(TEMPERATURE)
    assert out == {"CPU": 48.343, "GPU": 47.937, "tj": 48.343}


# -- the C920's mode explosion --------------------------------------------

C920 = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.100s (10.000 fps)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.100s (10.000 fps)
\t\t\tInterval: Discrete 0.200s (5.000 fps)
\t\tSize: Discrete 1920x1080
\t\t\tInterval: Discrete 0.200s (5.000 fps)
\t[1]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.200s (5.000 fps)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.100s (10.000 fps)
\t\tSize: Discrete 1920x1080
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.200s (5.000 fps)
"""


def test_collapse_keeps_only_the_fps_ceiling():
    modes = parse_formats(C920)
    collapsed = collapse_modes(modes)
    assert len(modes) == 11
    assert len(collapsed) == 6  # 2 formats x 3 resolutions
    mjpg_1080 = [m for m in collapsed if m["fourcc"] == "MJPG" and m["width"] == 1920]
    assert mjpg_1080[0]["fps"] == 30.0


def test_realtime_ceiling_separates_compressed_from_raw():
    ceiling = realtime_ceiling(parse_formats(C920), fps=30.0)
    # The USB bandwidth wall, stated as one fact per format.
    assert (ceiling["MJPG"]["width"], ceiling["MJPG"]["height"]) == (1920, 1080)
    assert (ceiling["YUYV"]["width"], ceiling["YUYV"]["height"]) == (640, 480)


def test_collapse_is_sorted_largest_first_within_a_format():
    collapsed = collapse_modes(parse_formats(C920))
    mjpg = [(m["width"], m["height"]) for m in collapsed if m["fourcc"] == "MJPG"]
    assert mjpg == [(1920, 1080), (1280, 720), (640, 480)]


# -- jetson_clocks pin detection: the third wrong inference ---------------
#
# Verified with `ohsb doctor --dump-jtop` before and after `sudo jetson_clocks`
# on the target board. /sys/class/devfreq/.../min_freq,max_freq read
# 624750000/624750000 in BOTH captures — unusable. The GPU devfreq governor
# stayed 'nvhost_podgov' in both — also unusable. jtop's own freq.min/max is
# the only field that actually changed.

GPU_BEFORE_PIN = {
    "ga10b": {
        "type": "integrated",
        "status": {"railgate": False, "tpc_pg_mask": False, "3d_scaling": True, "load": 0.0},
        "freq": {"governor": "nvhost_podgov", "cur": 306000, "max": 624750,
                 "min": 306000, "GPC": [305965]},
        "power_control": "auto",
    }
}

GPU_AFTER_PIN = {
    "ga10b": {
        "type": "integrated",
        "status": {"railgate": False, "tpc_pg_mask": False, "3d_scaling": True, "load": 0.0},
        "freq": {"governor": "nvhost_podgov", "cur": 624750, "max": 624750,
                 "min": 624750, "GPC": [624691]},
        "power_control": "auto",
    }
}


def test_gpu_freq_range_before_pin_shows_scaling():
    out = extract_gpu(GPU_BEFORE_PIN)
    assert out["freq_min_khz"] == 306000
    assert out["freq_max_khz"] == 624750
    assert out["freq_min_khz"] != out["freq_max_khz"]


def test_gpu_freq_range_after_pin_has_collapsed():
    out = extract_gpu(GPU_AFTER_PIN)
    assert out["freq_min_khz"] == out["freq_max_khz"] == 624750


def test_3d_scaling_flag_does_not_track_pin_state():
    # Both captures report 3d_scaling: True — it must not be used as the signal.
    assert extract_gpu(GPU_BEFORE_PIN)["scaling_3d"] == 1.0
    assert extract_gpu(GPU_AFTER_PIN)["scaling_3d"] == 1.0


def test_governor_name_does_not_track_pin_state():
    # Neither capture shows a governor rename to 'userspace'.
    for payload in (GPU_BEFORE_PIN, GPU_AFTER_PIN):
        assert payload["ga10b"]["freq"]["governor"] == "nvhost_podgov"


# -- the fourth wrong inference: a single jtop freq sample --------------
#
# Recorded on a fresh reboot, `jetson_clocks` never run this boot, GPU under
# real load (36%) from the diagnostic process itself. This is what falsified
# "jtop freq.min == freq.max means pinned": it reads identically to the
# after-pin capture above despite jetson_clocks never having executed.

GPU_UNPINNED_BUT_BUSY = {
    "ga10b": {
        "type": "integrated",
        "status": {"railgate": False, "tpc_pg_mask": False, "3d_scaling": True, "load": 36.0},
        "freq": {"governor": "nvhost_podgov", "cur": 624750, "max": 624750,
                 "min": 624750, "GPC": [624660]},
        "power_control": "auto",
    }
}


def test_busy_but_unpinned_gpu_collapses_min_max_just_like_pinned():
    # The whole point: this payload is indistinguishable from GPU_AFTER_PIN
    # by min==max alone, which is exactly why that heuristic was dropped.
    busy = extract_gpu(GPU_UNPINNED_BUT_BUSY)
    pinned = extract_gpu(GPU_AFTER_PIN)
    assert busy["freq_min_khz"] == busy["freq_max_khz"] == 624750
    assert pinned["freq_min_khz"] == pinned["freq_max_khz"] == 624750
    # Only the load differs, and load is not part of the pin decision.


# -- power sampling: thread starvation on the live path -------------------
#
# Confirmed on the board: a background *thread* running jtop's sampling
# loop collected exactly 1 sample over a ~10s live benchmark run at a
# 100ms interval, instead of the ~100 expected — every published power/
# energy number up to that point was effectively a single instantaneous
# reading, not a time average. JtopMonitor now samples in a separate OS
# process instead, which cannot be starved by the benchmarked process's
# GIL usage regardless of its exact cause.

def test_jtop_monitor_uses_a_process_not_a_thread():
    from ohsb.monitors.jtop_monitor import JtopMonitor

    assert not hasattr(JtopMonitor, "_pump"), (
        "sampling must not be a background thread method — that was the "
        "starved design; see _sampling_process for the replacement"
    )


def test_unavailable_jtop_reports_cleanly_without_spawning_anything():
    from ohsb.config import PowerConfig
    from ohsb.monitors.jtop_monitor import JtopMonitor

    monitor = JtopMonitor(PowerConfig())
    monitor.start()  # jtop is not installed in the test environment
    monitor.stop()
    summary = monitor.summary()
    assert summary["available"] is False
    assert summary["backend"] == "jtop"
