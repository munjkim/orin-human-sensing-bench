"""jtop payload extractors.

The target board runs a newer jetson-stats than this code was written
against, so the extractors must survive schema drift. These cases pin the
shapes seen across versions; `ohsb doctor --dump-jtop` on the board is what
confirms which one it actually emits.
"""

from ohsb.monitors.jtop_monitor import (
    extract_cpu_pct,
    extract_gpu,
    extract_power_mw,
    extract_ram_mb,
    extract_temps,
)


def test_power_nested_rail_shape():
    payload = {
        "rail": {
            "VDD_IN": {"volt": 5000, "curr": 980, "power": 4900, "avg": 5000},
            "VDD_CPU_GPU_CV": {"power": 924, "avg": 939},
            "VDD_SOC": {"power": 1500, "avg": 1600},
        },
        "tot": {"power": 4900, "avg": 5000},
    }
    out = extract_power_mw(payload)
    assert out["VDD_IN"] == 4900
    assert out["VDD_CPU_GPU_CV"] == 924
    # 'tot' must not leak in as a rail named 'tot'.
    assert "tot" not in out


def test_power_flat_shape():
    assert extract_power_mw({"VDD_IN": 4900, "VDD_SOC": 1500}) == {
        "VDD_IN": 4900.0,
        "VDD_SOC": 1500.0,
    }


def test_power_total_only_shape_synthesises_total_rail():
    out = extract_power_mw({"tot": {"power": 4900}})
    assert out == {"TOTAL": 4900.0}


def test_power_garbage_is_empty_not_an_exception():
    assert extract_power_mw(None) == {}
    assert extract_power_mw({"rail": "unexpected"}) == {}


def test_gpu_named_device_shape():
    out = extract_gpu({"ga10b": {"status": {"load": 26.9}, "freq": {"cur": 306000}}})
    assert out["load_pct"] == 26.9
    assert out["freq_khz"] == 306000


def test_cpu_uses_idle_when_present():
    assert extract_cpu_pct({"total": {"user": 12.0, "system": 5.0, "idle": 80.0}}) == 20.0


def test_cpu_falls_back_to_user_plus_system():
    assert extract_cpu_pct({"total": {"user": 12.0, "system": 5.0}}) == 17.0


def test_ram_converts_kb_to_mb():
    assert extract_ram_mb({"RAM": {"used": 3244032, "tot": 6434816}}) == 3168.0


def test_temps_skip_offline_sensors():
    out = extract_temps({
        "GPU": {"temp": 46.03, "online": True},
        "CV0": {"temp": -256.0, "online": False},
        "tj": 46.41,
    })
    assert out["GPU"] == 46.03
    assert out["tj"] == 46.41
    assert "CV0" not in out
