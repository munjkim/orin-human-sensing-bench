#!/usr/bin/env bash
# Sweep one config across the axes that move the number, then print the table.
#
#   ./scripts/sweep.sh configs/pose_landmarker.yaml
set -euo pipefail

CONFIG="${1:?usage: sweep.sh <config.yaml> [out_dir]}"
OUT="${2:-results/sweep-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

for delegate in cpu gpu; do
  for res in "640 480" "1280 720"; do
    read -r w h <<<"$res"
    name="$(basename "$CONFIG" .yaml)-${delegate}-${w}x${h}"
    echo "== $name"
    # Keep going on failure: a GPU delegate that will not initialise should
    # cost you one row, not the whole sweep.
    ohsb run -c "$CONFIG" -o "$OUT" -n "$name" -q \
      --set "task.delegate=$delegate" \
      --set "source.width=$w" \
      --set "source.height=$h" || echo "  FAILED: $name"
  done
done

echo
ohsb report "$OUT"
