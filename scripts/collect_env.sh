#!/usr/bin/env bash
# Capture everything needed to diagnose this board, in one shot.
#
# Run it, commit, push. That is the whole loop — no copying terminal output
# by hand, and every command's failure is recorded rather than aborting the
# rest, so a missing tool still produces a usable bundle.
#
#   ./scripts/collect_env.sh
#   git add env && git commit -m "Add board environment snapshot" && git push
set -uo pipefail   # deliberately not -e: a failing probe must not stop the sweep

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-$(hostname -s 2>/dev/null || echo board)-$(date +%Y%m%d-%H%M%S)}"
DEST="$ROOT/env/$NAME"
mkdir -p "$DEST"

capture() {
  local out="$1"; shift
  local rc
  printf '  %-24s %s\n' "$out" "$*"
  {
    printf '$ %s\n\n' "$*"
    if ! command -v "$1" >/dev/null 2>&1; then
      # Distinguish "tool absent" from "tool ran and failed" — on a board
      # missing v4l-utils or jetson-stats these are very different problems.
      printf '[not installed: %s]\n' "$1"
    else
      "$@" 2>&1
      rc=$?
      # $? must be read before any other command, printf included.
      if [[ $rc -ne 0 ]]; then
        printf '\n[exit %d]\n' "$rc"
      fi
    fi
  } > "$DEST/$out"
}

echo "collecting into env/$NAME"

# -- the harness's own view -------------------------------------------------
capture doctor.txt          ohsb doctor
capture doctor.json         ohsb doctor --json
capture jtop-schema.json    ohsb doctor --dump-jtop
capture cameras.txt         ohsb cameras
capture cameras.json        ohsb cameras --json
capture ohsb-list.txt       ohsb list

# -- raw camera capabilities ------------------------------------------------
# The authoritative source; ohsb cameras parses this, so keeping the raw text
# is what lets a parsing bug be told apart from a hardware limitation.
for dev in /dev/video*; do
  [[ -e "$dev" ]] || continue
  n="$(basename "$dev")"
  capture "v4l2-$n-formats.txt" v4l2-ctl -d "$dev" --list-formats-ext
  capture "v4l2-$n-info.txt"    v4l2-ctl -d "$dev" --all
done

# -- board and toolchain ----------------------------------------------------
capture uname.txt           uname -a
capture nvpmodel.txt        nvpmodel -q
capture jetson-clocks.txt   jetson_clocks --show
capture meminfo.txt         free -h
capture lsusb.txt           lsusb
capture pip-freeze.txt      python3 -m pip freeze
capture python-version.txt  python3 --version

{
  echo "collected: $(date -Iseconds)"
  echo "host: $(hostname 2>/dev/null || echo unknown)"
  echo "cwd: $ROOT"
  echo "git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > "$DEST/META.txt"

echo
echo "done: env/$NAME"
echo
echo "  git add env/$NAME && git commit -m 'Add env snapshot $NAME' && git push"
