#!/usr/bin/env bash
# Promote a set of scratch runs into a committed benchmark record.
#
# `results/` is gitignored: every run lands there, most are throwaway. This
# copies the ones worth keeping into `benchmarks/<name>/` together with a
# generated summary, so a published number always travels with the board
# state that produced it.
#
#   ./scripts/publish_results.sh orin-nano-baseline
#   ./scripts/publish_results.sh webcam-720p results/camera-matrix-20260824-1200
set -euo pipefail

NAME="${1:?usage: publish_results.sh <name> [source_dir]}"
SRC="${2:-results}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/benchmarks/$NAME"

shopt -s nullglob
files=("$SRC"/*.json)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "no result files in $SRC" >&2
  exit 1
fi

mkdir -p "$DEST"
cp "${files[@]}" "$DEST/"
echo "copied ${#files[@]} result(s) into benchmarks/$NAME/"

# The summary is generated, never hand-edited — regenerating it after adding
# a run is the whole point.
ohsb report "$DEST" --markdown -o "$DEST/README.md"

echo
echo "review, then:"
echo "  git add benchmarks/$NAME && git commit -m 'Add $NAME benchmark results' && git push"
