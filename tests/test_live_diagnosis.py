"""Bottleneck attribution.

A bare fps number cannot distinguish "the camera only delivers 30" from
"the model can only process 30". These tests pin the logic that separates
them, because that distinction is the whole point of the live mode.
"""

import math

from ohsb.runner import _diagnose


def _stages(capture, convert, infer):
    return {
        "capture": {"mean": capture},
        "convert": {"mean": convert},
        "infer": {"mean": infer},
    }


def test_fast_model_slow_camera_is_camera_bound():
    # Compute needs 10 ms/frame (100 fps) but the camera gives 30.
    out = _diagnose(
        baseline={"fps": 30.0, "frames": 60},
        stages=_stages(capture=23.0, convert=2.0, infer=8.0),
        throughput_fps=29.8,
        skipped=0,
    )
    assert out["verdict"].startswith("CAMERA-BOUND")
    assert out["pipeline_capable_fps"] == 100.0
    assert out["realtime_ratio"] > 0.98
    # The actionable half: a faster model buys nothing here.
    assert "faster model will not help" in out["verdict"]


def test_slow_model_fast_camera_is_compute_bound_and_names_the_stage():
    # Camera offers 30 fps; inference alone needs 80 ms (12.5 fps).
    out = _diagnose(
        baseline={"fps": 30.0, "frames": 60},
        stages=_stages(capture=0.5, convert=3.0, infer=80.0),
        throughput_fps=12.0,
        skipped=0,
    )
    assert out["verdict"].startswith("COMPUTE-BOUND")
    assert out["dominant_stage"] == "infer"
    assert out["realtime_ratio"] < 0.45


def test_conversion_can_be_the_dominant_stage():
    # A large raw frame where colour conversion outweighs a tiny model.
    out = _diagnose(
        baseline={"fps": 30.0, "frames": 60},
        stages=_stages(capture=0.5, convert=60.0, infer=10.0),
        throughput_fps=14.0,
        skipped=0,
    )
    assert out["dominant_stage"] == "convert"
    assert "'convert' is the largest stage" in out["verdict"]


def test_matched_camera_and_pipeline_is_balanced():
    out = _diagnose(
        baseline={"fps": 30.0, "frames": 60},
        stages=_stages(capture=2.0, convert=3.0, infer=30.0),
        throughput_fps=29.0,
        skipped=0,
    )
    assert out["verdict"].startswith("BALANCED")


def test_missing_baseline_refuses_to_guess():
    out = _diagnose(
        baseline={"fps": float("nan"), "frames": 0},
        stages=_stages(capture=1.0, convert=1.0, infer=1.0),
        throughput_fps=50.0,
        skipped=0,
    )
    assert "cannot attribute" in out["verdict"]
    assert math.isnan(out["realtime_ratio"])


def test_skipped_frames_are_carried_through():
    out = _diagnose({"fps": 30.0}, _stages(1.0, 1.0, 50.0), 18.0, skipped=42)
    assert out["frames_skipped"] == 42
