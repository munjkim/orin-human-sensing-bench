#!/usr/bin/env bash
# Download the MediaPipe model bundles the configs reference.
#
# Safe to re-run: existing files are skipped. Google occasionally rotates the
# version segment in these paths, so a 404 is a real failure, not noise —
# the script stops rather than leaving a truncated bundle behind.
#
#   ./scripts/fetch_models.sh              # download into ./models
#   ./scripts/fetch_models.sh --print-urls # just list them, to fetch by hand
set -euo pipefail

BASE="https://storage.googleapis.com/mediapipe-models"

# name <TAB> url-suffix. Defined once so --print-urls and the downloader
# cannot drift apart.
MODELS="\
pose_landmarker_lite.task	pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
pose_landmarker_full.task	pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
pose_landmarker_heavy.task	pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
blaze_face_short_range.tflite	face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
face_landmarker.task	face_landmarker/face_landmarker/float16/1/face_landmarker.task
mobilenet_v3_small_embedder.tflite	image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"

if [[ "${1:-}" == "--print-urls" ]]; then
  # Emitted as `curl -o name url` lines so the output is directly runnable
  # or pasteable into a browser, whichever is easier.
  while IFS=$'\t' read -r name suffix; do
    printf '%s\t%s/%s\n' "$name" "$BASE" "$suffix"
  done <<< "$MODELS"
  exit 0
fi

DEST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models}"
mkdir -p "$DEST"

# JetPack images do not always ship curl. Pick whichever downloader exists
# rather than failing with a bare "curl: command not found" from inside a
# loop, which tells you nothing about what to install.
if command -v curl >/dev/null 2>&1; then
  DOWNLOADER=curl
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER=wget
else
  cat >&2 <<'MSG'
error: neither curl nor wget is available.

  sudo apt install curl

Or list the URLs and fetch them by hand:

  ./scripts/fetch_models.sh --print-urls
MSG
  exit 1
fi

download() {
  local url="$1" out="$2"
  case "$DOWNLOADER" in
    curl) curl -fsSL --retry 3 -o "$out" "$url" ;;
    wget) wget -q --tries=3 -O "$out" "$url" ;;
  esac
}

fetch() {
  local name="$1" url="$2"
  local out="$DEST/$name"

  if [[ -s "$out" ]]; then
    printf '  skip  %s (already present)\n' "$name"
    return 0
  fi

  printf '  get   %s\n' "$name"
  if ! download "$url" "$out.part"; then
    rm -f "$out.part"
    printf '  FAIL  %s\n        %s\n' "$name" "$url" >&2
    return 1
  fi

  # A rotated URL can return an HTML error page with a 200. Every real bundle
  # here is well over 100 kB, so anything smaller is not a model.
  local size
  size=$(wc -c < "$out.part")
  if [[ "$size" -lt 100000 ]]; then
    rm -f "$out.part"
    printf '  FAIL  %s — got %s bytes, not a model bundle\n        %s\n' \
      "$name" "$size" "$url" >&2
    return 1
  fi

  mv "$out.part" "$out"
}

echo "downloading MediaPipe model bundles into $DEST (using $DOWNLOADER)"

failed=0
while IFS=$'\t' read -r name suffix; do
  fetch "$name" "$BASE/$suffix" || failed=$((failed + 1))
done <<< "$MODELS"

echo
if [[ $failed -gt 0 ]]; then
  echo "$failed download(s) failed — see the URLs above" >&2
  exit 1
fi
echo "done. verify with: ohsb doctor"
