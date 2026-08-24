#!/usr/bin/env bash
# Download the MediaPipe model bundles the configs reference.
#
# Safe to re-run: existing files are skipped. Google occasionally rotates the
# version segment in these paths, so a 404 is a real failure, not noise —
# the script stops rather than leaving a truncated bundle behind.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models}"
BASE="https://storage.googleapis.com/mediapipe-models"
mkdir -p "$DEST"

fetch() {
  local name="$1" url="$2"
  local out="$DEST/$name"
  if [[ -s "$out" ]]; then
    printf '  skip  %s (already present)\n' "$name"
    return 0
  fi
  printf '  get   %s\n' "$name"
  if ! curl -fsSL --retry 3 -o "$out.part" "$url"; then
    rm -f "$out.part"
    printf '  FAIL  %s\n        %s\n' "$name" "$url" >&2
    return 1
  fi
  mv "$out.part" "$out"
}

echo "downloading MediaPipe model bundles into $DEST"

# Pose — three size variants; benchmarking all three is the point.
fetch pose_landmarker_lite.task \
  "$BASE/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
fetch pose_landmarker_full.task \
  "$BASE/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
fetch pose_landmarker_heavy.task \
  "$BASE/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

# Face detection + mesh.
fetch blaze_face_short_range.tflite \
  "$BASE/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
fetch face_landmarker.task \
  "$BASE/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Embedder for the face_recognition pipeline's identity stage.
fetch mobilenet_v3_small_embedder.tflite \
  "$BASE/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"

echo
echo "done. verify with: ohsb doctor"
