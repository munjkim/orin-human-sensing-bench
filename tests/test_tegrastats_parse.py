"""Parser locked against real tegrastats output.

The Orin Nano line below is the shape the target board emits (rails
VDD_IN / VDD_CPU_GPU_CV / VDD_SOC); the AGX line uses entirely different rail
names, which is why rails are discovered rather than hardcoded.
"""

from ohsb.monitors.jtop_monitor import total_rail
from ohsb.monitors.tegrastats import parse_line

ORIN_NANO = (
    "08-24-2026 11:19:00 RAM 3172/6284MB (lfb 44x4MB) SWAP 0/3142MB (cached 0MB) "
    "CPU [6%@1510,6%@1510,10%@729,7%@729,29%@729,18%@729] EMC_FREQ 0%@2133 "
    "GR3D_FREQ 26%@[306] NVDEC off NVJPG off VIC_FREQ off "
    "APE 200 cpu@46.41C soc2@45.34C soc0@43.62C gpu@46.03C tj@46.41C soc1@44.38C "
    "VDD_IN 4900mW/5000mW VDD_CPU_GPU_CV 924mW/939mW VDD_SOC 1500mW/1600mW"
)

AGX_ORIN = (
    "10-13-2026 12:00:00 RAM 4023/30536MB (lfb 5555x4MB) SWAP 0/15268MB (cached 0MB) "
    "CPU [0%@1190,0%@1190,off,off] EMC_FREQ 0%@2133 GR3D_FREQ 0%@[0,0] "
    "cpu@45.5C tj@45.5C VDD_GPU_SOC 3188mW/3188mW VDD_CPU_CV 1195mW/1195mW "
    "VIN_SYS_5V0 3229mW/3229mW"
)


def test_orin_nano_rails_and_utilisation():
    s = parse_line(ORIN_NANO)
    # The current value is kept, not tegrastats' own since-start average.
    assert s["power_mw"] == {"VDD_IN": 4900, "VDD_CPU_GPU_CV": 924, "VDD_SOC": 1500}
    assert s["gpu_util_pct"] == 26
    assert s["ram_used_mb"] == 3172
    assert s["ram_total_mb"] == 6284
    assert s["cpu_util_pct_per_core"] == [6, 6, 10, 7, 29, 18]
    assert s["temp_c"]["gpu"] == 46.03
    assert total_rail(s["power_mw"]) == "VDD_IN"


def test_agx_rail_names_differ():
    s = parse_line(AGX_ORIN)
    assert set(s["power_mw"]) == {"VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"}
    # No VDD_IN on AGX — the total falls through to the 5V input rail.
    assert total_rail(s["power_mw"]) == "VIN_SYS_5V0"


def test_offline_cores_are_skipped_not_counted_as_zero():
    s = parse_line(AGX_ORIN)
    assert s["cpu_util_pct_per_core"] == [0, 0]


def test_unparseable_line_does_not_raise():
    s = parse_line("garbage without any known fields")
    assert s["power_mw"] == {}
    assert "gpu_util_pct" not in s
