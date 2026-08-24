#!/usr/bin/env bash
# Dump every signal that could indicate whether clocks are pinned.
#
# Exists because inferring "is jetson_clocks active" has already been wrong
# twice on this hardware: /sys/class/devfreq reported min == max == 624 MHz
# while jtop reported the GPU idling at 306 MHz with 3d_scaling on. Rather
# than guess a third time, print every candidate signal side by side.
#
#   ./scripts/diagnose_clocks.sh            # current state
#   sudo ./scripts/diagnose_clocks.sh       # also gets `jetson_clocks --show`
set -uo pipefail

hr() { printf '\n== %s\n' "$*"; }

hr "jetson_clocks --show  (authoritative; needs root)"
if [[ $EUID -eq 0 ]]; then
  jetson_clocks --show 2>&1
else
  sudo -n jetson_clocks --show 2>&1 || \
    echo "(needs root — re-run as: sudo $0)"
fi

hr "nvpmodel"
nvpmodel -q 2>&1 | head -3

hr "GPU devfreq node (/sys/class/devfreq)"
for node in /sys/class/devfreq/*; do
  name="$(basename "$node")"
  case "$name" in
    *gpu*|*ga10b*|*gv11b*) ;;
    *) continue ;;
  esac
  echo "  node: $name"
  for f in governor cur_freq min_freq max_freq available_frequencies \
           available_governors target_freq polling_interval; do
    [[ -r "$node/$f" ]] && printf '    %-24s %s\n' "$f" "$(cat "$node/$f" 2>&1)"
  done
done

hr "GPU railgate / scaling (platform node)"
for f in /sys/devices/platform/bus@0/17000000.ga10b/railgate_enable \
         /sys/devices/17000000.ga10b/railgate_enable \
         /sys/devices/platform/bus@0/17000000.ga10b/enable_3d_scaling \
         /sys/devices/17000000.ga10b/enable_3d_scaling; do
  [[ -r "$f" ]] && printf '  %-64s %s\n' "$f" "$(cat "$f" 2>&1)"
done

hr "CPU cpufreq (cpu0 and cpu5)"
for c in 0 5; do
  d="/sys/devices/system/cpu/cpu$c/cpufreq"
  [[ -d "$d" ]] || continue
  echo "  cpu$c"
  for f in scaling_governor scaling_cur_freq scaling_min_freq scaling_max_freq \
           cpuinfo_min_freq cpuinfo_max_freq; do
    [[ -r "$d/$f" ]] && printf '    %-24s %s\n' "$f" "$(cat "$d/$f" 2>&1)"
  done
done

hr "EMC"
for f in /sys/kernel/debug/bpmp/debug/clk/emc/rate \
         /sys/kernel/nvpmodel_emc_cap/emc_iso_cap; do
  [[ -r "$f" ]] && printf '  %-56s %s\n' "$f" "$(cat "$f" 2>&1)"
done

hr "jtop's view of the GPU"
python3 - <<'PY' 2>&1
try:
    from jtop import jtop
except Exception as exc:
    print(f"  jtop unavailable: {exc}")
else:
    try:
        with jtop(interval=0.5) as j:
            if j.ok():
                print(f"  type: {type(j.gpu).__name__}")
                print(f"  {dict(j.gpu)}")
            else:
                print("  jtop client not ready")
    except Exception as exc:
        print(f"  jtop error: {type(exc).__name__}: {exc}")
PY

hr "what ohsb currently concludes"
python3 - <<'PY' 2>&1
try:
    from ohsb.platform import gpu_freq, power_mode, reproducibility_warnings
except Exception as exc:
    print(f"  ohsb not importable: {exc}")
else:
    print(f"  gpu_freq():            {gpu_freq()}")
    pm = power_mode()
    print(f"  jetson_clocks_active:  {pm.get('jetson_clocks_active')}")
    for w in reproducibility_warnings():
        print(f"  ! {w}")
PY
