import math

from ohsb.metrics import summarize


def test_summarize_empty():
    assert summarize([]) == {"count": 0}


def test_summarize_percentiles():
    out = summarize(list(range(1, 101)), unit="ms")
    assert out["count"] == 100
    assert out["min"] == 1
    assert out["max"] == 100
    assert math.isclose(out["p50"], 50.5)
    assert out["unit"] == "ms"


def test_single_sample_has_zero_std():
    out = summarize([4.2])
    assert out["std"] == 0.0
    assert out["mean"] == 4.2
