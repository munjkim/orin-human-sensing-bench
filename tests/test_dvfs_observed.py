"""Post-hoc DVFS detection: did the GPU clock vary during THIS run.

This replaced every pre-run static probe after three were falsified on the
Orin Nano (devfreq sysfs, governor name, a single jtop sample — see
tests/test_orin_nano_payloads.py). Sampling gpu_freq_khz across the whole
measured run and checking min vs max is not a heuristic: it is a direct
fact about the run that produced the numbers next to it.
"""

from ohsb.runner import _dvfs_observed


def _power(gpu_freq_stats):
    return {"available": True, "gpu_freq_khz": gpu_freq_stats} if gpu_freq_stats else \
           {"available": True}


def test_steady_clock_is_conclusive_and_not_scaled():
    out = _dvfs_observed(_power({"count": 300, "min": 624750, "max": 624750}))
    assert out["conclusive"] is True
    assert out["scaled"] is False
    assert "held steady" in out["note"]


def test_varying_clock_is_conclusive_and_scaled():
    out = _dvfs_observed(_power({"count": 300, "min": 306000, "max": 624750}))
    assert out["conclusive"] is True
    assert out["scaled"] is True
    assert "varied" in out["note"]


def test_too_few_samples_is_inconclusive_not_a_guess():
    out = _dvfs_observed(_power({"count": 1, "min": 624750, "max": 624750}))
    assert out["conclusive"] is False


def test_no_gpu_freq_data_is_inconclusive():
    assert _dvfs_observed(_power(None))["conclusive"] is False


def test_power_unavailable_is_inconclusive():
    assert _dvfs_observed({"available": False})["conclusive"] is False
