#!/usr/bin/env bash
# Put the Orin into a repeatable benchmarking state.
#
# DRY RUN BY DEFAULT — it prints the commands and changes nothing. Pass
# --apply to actually run them. Every one of these changes global machine
# state, and two of them (MAXN, isolate multi-user) will visibly disrupt a
# desktop session, so the confirmation is deliberate.
#
#   ./scripts/setup_orin.sh            # show what would change
#   sudo ./scripts/setup_orin.sh --apply
set -euo pipefail

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

run() {
  if [[ $APPLY -eq 1 ]]; then
    printf '+ %s\n' "$*"
    "$@"
  else
    printf '  would run: %s\n' "$*"
  fi
}

if [[ $APPLY -eq 1 && $EUID -ne 0 ]]; then
  echo "error: --apply needs root (re-run with sudo)" >&2
  exit 1
fi

echo "== current state =="
nvpmodel -q || true
echo
echo "== changes =="

# 1. Unrestricted power mode. On the Orin Nano dev kit mode 0 is 15W and
#    mode 1 is 7W; there is no higher mode, so this mainly guards against a
#    board left in 7W. Record whichever you use — it caps achievable clocks.
run nvpmodel -m 0

# 2. Pin CPU/GPU/EMC clocks at max. Without this the GPU scales between
#    306 MHz and 624 MHz during a run and latency percentiles are noise.
run jetson_clocks

# 3. Fan to full. jetson_clocks disables the thermal-driven fan curve, so a
#    long sweep can throttle silently without this.
run jetson_clocks --fan

echo
echo "== optional: free the ~3 GB the desktop session holds =="
echo "  sudo systemctl isolate multi-user.target   # drop to console"
echo "  sudo systemctl start graphical.target      # restore afterwards"
echo
echo "== optional: let tegrastats run without a password prompt =="
echo "  (only needed if you set monitor.power.backend=tegrastats; jtop needs nothing)"
echo "  echo \"\$USER ALL=(root) NOPASSWD: /usr/bin/tegrastats\" | sudo tee /etc/sudoers.d/ohsb-tegrastats"

if [[ $APPLY -eq 0 ]]; then
  echo
  echo "dry run — nothing changed. re-run with: sudo $0 --apply"
fi
