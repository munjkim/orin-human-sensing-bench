"""Energy arithmetic — mW x s = mJ is easy to get wrong by 1000x."""

from ohsb.runner import _energy


def test_energy_per_frame_and_efficiency():
    power = {"available": True, "avg_total_power_mw": 5000.0, "total_rail": "VDD_IN"}
    # 5 W for 10 s = 50 J total; over 300 frames that is 166.7 mJ/frame.
    out = _energy(power, wall_s=10.0, iterations=300, throughput_fps=30.0)
    assert out["avg_power_w"] == 5.0
    assert out["total_mj"] == 50000.0
    assert abs(out["mj_per_frame"] - 166.667) < 0.01
    assert out["fps_per_watt"] == 6.0
    assert out["rail"] == "VDD_IN"


def test_no_power_means_no_energy_rather_than_zeros():
    assert _energy({"available": False}, 10.0, 300, 30.0) == {}
    assert _energy({"available": True}, 10.0, 300, 30.0) == {}
