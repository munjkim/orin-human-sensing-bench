"""End-to-end harness behaviour, using the model-free `noop` task."""

import json

from ohsb.config import BenchmarkConfig
from ohsb.results import format_report
from ohsb.runner import run_benchmark


def _cfg(**over):
    data = {
        "run": {"name": "t", "iterations": 20, "warmup": 3, "repeat": 1},
        "task": {"type": "noop", "options": {"sleep_ms": 0}},
        "source": {"type": "synthetic", "width": 64, "height": 48, "count": 4},
        "monitor": {"power": {"backend": "none"}},
        "output": {"dir": "results", "save_raw_latencies": True},
    }
    for dotted, value in over.items():
        section, _, key = dotted.partition("__")
        data[section][key] = value
    return BenchmarkConfig.from_dict(data)


def test_single_run_shape():
    report = run_benchmark(_cfg())
    assert len(report.runs) == 1
    run = report.runs[0]
    assert run.latency_ms["count"] == 20
    assert len(run.raw_latencies_ms) == 20
    assert run.throughput_fps > 0
    assert run.warmup == 3


def test_repeats_warm_up_only_once():
    report = run_benchmark(_cfg(run__repeat=3))
    assert [r.warmup for r in report.runs] == [3, 0, 0]
    assert [r.name for r in report.runs] == ["t#1", "t#2", "t#3"]


def test_measured_latency_tracks_a_known_workload():
    # The busy-wait gives the loop a known floor; anything wildly off means
    # the timing path itself is broken.
    report = run_benchmark(_cfg(task__options={"sleep_ms": 2.0}))
    mean = report.runs[0].latency_ms["mean"]
    assert 1.8 < mean < 3.0


def test_power_backend_none_is_reported_not_silently_dropped():
    report = run_benchmark(_cfg())
    power = report.runs[0].power
    assert power["available"] is False
    assert "none" in power["reason"]
    # No power means no energy figures rather than zeros pretending to be data.
    assert report.runs[0].energy == {}


def test_report_is_json_serialisable_and_self_contained(tmp_path):
    report = run_benchmark(_cfg())
    path = report.write(tmp_path)
    data = json.loads(path.read_text())
    assert data["schema_version"] == 1
    assert data["task"]["type"] == "noop"
    assert data["platform"]["libraries"]["python"]
    assert data["config"]["run"]["name"] == "t"


def test_format_report_runs_without_power():
    text = format_report(run_benchmark(_cfg()))
    assert "latency ms" in text
    assert "power" in text
