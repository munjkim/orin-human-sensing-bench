# Benchmark records

Committed results. Each subdirectory is one measurement session: the raw
result JSON files plus a generated `README.md` carrying the comparison table
and the board state behind it.

`results/` at the repo root is gitignored — that is scratch space where every
run lands. Only runs worth keeping get promoted here:

```bash
./scripts/publish_results.sh <name> [source_dir]
git add benchmarks/<name> && git commit -m "Add <name> benchmark results" && git push
```

## Why the board state is committed alongside

A latency number from an Orin Nano is meaningless without knowing whether
`jetson_clocks` was pinned, which `nvpmodel` mode was active, and how much
RAM was free. The board boots at 15 W with DVFS live (GPU 306→624 MHz), so
two runs of the same config can differ by more than 2x. Each generated
summary therefore includes the platform snapshot and any reproducibility
warnings the run recorded.

## Naming

`<board>-<what-varied>`, e.g.:

- `orin-nano-delegate-cpu-vs-gpu`
- `orin-nano-webcam-720p-matrix`
- `orin-nano-pose-model-variants`

Keep one session per directory. Mixing runs collected under different clock
states into one table is how a benchmark starts lying.
