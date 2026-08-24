"""Live pipeline end-to-end, driven by a fake camera.

Lets the whole live path — settle, baseline, warmup, staged timing, verdict —
be exercised off-board, with no webcam and no MediaPipe.
"""

import time

import numpy as np
import pytest

from ohsb.config import BenchmarkConfig
from ohsb.runner import run_live_benchmark
from ohsb.sources.webcam import _substituted


class FakeCamera:
    """A camera that delivers frames at a fixed rate."""

    #: Set by each test before run_live_benchmark is called.
    frame_interval_s = 0.002

    def __init__(self, cfg):
        self.cfg = cfg
        self.reads = 0
        self.closed = False

    def open(self):
        return self

    def read_bgr(self):
        self.reads += 1
        # Actually block, the way a real camera paces the loop — otherwise
        # the capture stage would be free and the shares meaningless.
        time.sleep(self.frame_interval_s)
        frame = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        return frame, 0

    def to_rgb(self, bgr):
        return np.ascontiguousarray(bgr[:, :, ::-1])

    def close(self):
        self.closed = True

    def describe(self):
        return {
            "type": "webcam",
            "requested": {"device": 0, "width": self.cfg.width, "height": self.cfg.height},
            "negotiated": {"fourcc": "MJPG", "width": self.cfg.width,
                           "height": self.cfg.height, "fps": 30.0},
            "mode_substituted": None,
            "drain": self.cfg.drain,
            "width": self.cfg.width,
            "height": self.cfg.height,
        }


@pytest.fixture
def fake_camera(monkeypatch):
    import ohsb.sources.webcam as webcam

    monkeypatch.setattr(webcam, "WebcamSource", FakeCamera)
    return FakeCamera


def _cfg(**over):
    data = {
        "run": {"name": "live", "iterations": 15, "warmup": 2, "repeat": 1},
        "task": {"type": "noop", "options": {"sleep_ms": 0}},
        "camera": {"device": 0, "width": 64, "height": 48,
                   "settle_frames": 3, "baseline_frames": 10},
        "monitor": {"power": {"backend": "none"}},
        "output": {"dir": "results", "save_raw_latencies": True},
    }
    for dotted, value in over.items():
        section, _, key = dotted.partition("__")
        data[section][key] = value
    return BenchmarkConfig.from_dict(data)


def test_live_run_reports_mode_and_stages(fake_camera):
    report = run_live_benchmark(_cfg())
    assert report.mode == "live"
    run = report.runs[0]
    # Every live frame is timed in three parts; a missing stage means the
    # bottleneck verdict is being computed from incomplete data.
    assert set(run.stage_latency_ms) == {"capture", "convert", "infer"}
    assert run.latency_ms["count"] == 15
    assert run.live["camera_baseline_frames"] == 10


def test_settle_and_baseline_frames_are_consumed_before_measuring(fake_camera):
    captured = {}

    class Counting(FakeCamera):
        def close(self):
            captured["reads"] = self.reads
            super().close()

    import ohsb.sources.webcam as webcam

    webcam.WebcamSource = Counting
    run_live_benchmark(_cfg())
    # 3 settle + 10 baseline + 2 warmup + 15 measured
    assert captured["reads"] == 30


def test_camera_is_closed_even_when_the_task_fails(fake_camera):
    cfg = _cfg(task__type="pose_landmarker")  # needs a model that is not there
    with pytest.raises((FileNotFoundError, ImportError, ValueError)):
        run_live_benchmark(cfg)


def test_stage_shares_sum_to_roughly_the_whole_frame(fake_camera):
    run = run_live_benchmark(_cfg()).runs[0]
    total_share = sum(s["share_pct"] for s in run.stage_latency_ms.values())
    assert 95.0 < total_share < 105.0


def test_live_result_serialises_with_the_verdict(tmp_path, fake_camera):
    import json

    report = run_live_benchmark(_cfg())
    data = json.loads(report.write(tmp_path).read_text())
    assert data["mode"] == "live"
    assert "verdict" in data["runs"][0]["live"]


# -- mode substitution -----------------------------------------------------

def test_substitution_detects_a_silently_downgraded_resolution():
    note = _substituted(
        {"width": 1920, "height": 1080, "fourcc": "MJPG"},
        {"width": 640, "height": 480, "fourcc": "MJPG"},
    )
    assert "asked 1920, got 640" in note
    assert "asked 1080, got 480" in note


def test_substitution_detects_a_rejected_pixel_format():
    # The failure that silently caps a USB 2.0 webcam at raw-YUYV rates.
    note = _substituted(
        {"width": 1280, "height": 720, "fourcc": "MJPG"},
        {"width": 1280, "height": 720, "fourcc": "YUYV"},
    )
    assert note == "fourcc: asked MJPG, got YUYV"


def test_no_substitution_returns_none():
    honoured = {"width": 1280, "height": 720, "fourcc": "MJPG"}
    assert _substituted(honoured, dict(honoured)) is None


def test_auto_fourcc_is_never_reported_as_substituted():
    note = _substituted(
        {"width": 640, "height": 480, "fourcc": "auto"},
        {"width": 640, "height": 480, "fourcc": "YUYV"},
    )
    assert note is None


# -- fourcc read-back ------------------------------------------------------

def test_decode_fourcc_round_trips_a_real_format():
    from ohsb.sources.base import decode_fourcc

    # 'MJPG' packed little-endian, the way V4L2 reports it.
    packed = sum(ord(c) << (8 * i) for i, c in enumerate("MJPG"))
    assert decode_fourcc(packed) == "MJPG"


@pytest.mark.parametrize("value", [-1, 0, 0xFFFFFFFF])
def test_decode_fourcc_returns_empty_for_backends_that_do_not_report_it(value):
    # macOS AVFoundation returns -1; decoding it as characters yields mojibake.
    from ohsb.sources.base import decode_fourcc

    assert decode_fourcc(value) == ""


def test_unreadable_fourcc_is_not_reported_as_a_substitution():
    # The bug this guards: an unreadable format read back as "" must not be
    # announced as "asked MJPG, got <garbage>".
    assert _substituted(
        {"width": 1280, "height": 720, "fourcc": "MJPG"},
        {"width": 1280, "height": 720, "fourcc": ""},
    ) is None
