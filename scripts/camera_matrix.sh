#!/usr/bin/env bash
# Sweep the live camera pipeline across resolution x task x delegate.
#
# This is the "what does my webcam actually give me" run. Start it after
# `ohsb cameras` has told you which modes the device really supports —
# sweeping resolutions it does not offer just measures the driver
# substituting something else.
#
#   ./scripts/camera_matrix.sh                      # defaults below
#   ./scripts/camera_matrix.sh "1280x720 1920x1080" "cpu gpu"
set -euo pipefail

# Defaults chosen from a real C920: MJPG reaches 30 fps at every one of
# these, so the camera is not the variable — the Orin is.
RESOLUTIONS="${1:-640x480 1280x720 1920x1080}"
DELEGATES="${2:-cpu gpu}"
CONFIGS="${3:-configs/webcam_face_detect.yaml configs/webcam_face.yaml configs/webcam_pose.yaml}"
OUT="results/camera-matrix-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

for config in $CONFIGS; do
  for res in $RESOLUTIONS; do
    w="${res%x*}"; h="${res#*x}"
    for delegate in $DELEGATES; do
      name="$(basename "$config" .yaml)-${delegate}-${res}"
      echo "== $name"
      # Keep going on failure: a GPU delegate that will not initialise, or a
      # resolution the camera rejects, should cost one row, not the sweep.
      ohsb live -c "$config" -o "$OUT" -n "$name" -q \
        --set "task.delegate=$delegate" \
        --set "camera.width=$w" \
        --set "camera.height=$h" || echo "  FAILED: $name"
    done
  done
done

echo
ohsb report "$OUT"
