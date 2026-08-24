import pytest

from ohsb.config import BenchmarkConfig, ConfigError


def test_defaults():
    cfg = BenchmarkConfig.from_dict({})
    assert cfg.run.iterations == 200
    assert cfg.task.delegate == "cpu"
    assert cfg.monitor.power.backend == "auto"


def test_unknown_key_is_rejected():
    # Typos must fail loudly: silently ignoring them produces a run that does
    # not match its config.
    with pytest.raises(ConfigError, match="delagate"):
        BenchmarkConfig.from_dict({"task": {"delagate": "gpu"}})


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ConfigError, match="montior"):
        BenchmarkConfig.from_dict({"montior": {}})


@pytest.mark.parametrize(
    "data,message",
    [
        ({"task": {"delegate": "tensorrt"}}, "delegate"),
        ({"task": {"running_mode": "live_stream"}}, "running_mode"),
        ({"run": {"iterations": 0}}, "iterations"),
        ({"monitor": {"power": {"backend": "nvidia-smi"}}}, "backend"),
    ],
)
def test_invalid_values(data, message):
    with pytest.raises(ConfigError, match=message):
        BenchmarkConfig.from_dict(data)


def test_overrides_are_typed_and_nested():
    cfg = BenchmarkConfig.from_dict({"task": {"type": "noop"}})
    out = cfg.apply_overrides({"task.delegate": "gpu", "run.iterations": 10})
    assert out.task.delegate == "gpu"
    assert out.run.iterations == 10
    # The original is untouched.
    assert cfg.task.delegate == "cpu"
