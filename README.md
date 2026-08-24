# orin-human-sensing-bench

Inference benchmark for **MediaPipe human sensing tasks on the NVIDIA Jetson
Orin Nano** — latency, throughput, and power.

Measures pose landmarking, face detection, face mesh, and a face-recognition
pipeline across the axes that actually move the number: **CPU vs GPU
delegate, model variant, and input resolution**. Every run records the board
state that produced it, because on a 15 W Orin Nano with DVFS active the
same config can differ by more than 2x.

Two modes, because they answer different questions:

- **`ohsb run`** — pre-decoded frames, replayed from memory. What does the
  *model* cost?
- **`ohsb live`** — a real USB webcam, capture → convert → infer. What do
  you *actually get*, and is the camera or the Orin the ceiling?

This is a *performance* benchmark. It does not measure accuracy — detection
counts are reported only as a sanity check that the model is doing something.

## Quick start

On the Orin (see [docs/orin-setup.md](docs/orin-setup.md) first — Python 3.8,
MediaPipe install, and clock pinning all have board-specific answers):

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -U pip setuptools wheel    # JetPack 5 ships pip 20.0.2, too old for PEP 660
pip install -e '.[dev]'

./scripts/fetch_models.sh          # download MediaPipe model bundles
sudo ./scripts/setup_orin.sh --apply   # pin clocks — do this before measuring
ohsb doctor                        # verify prerequisites and board state

# offline: what the model costs
ohsb run -c configs/pose_landmarker.yaml
ohsb run -c configs/pose_landmarker.yaml --set task.delegate=gpu

# live: what the webcam + model actually deliver
ohsb cameras                                   # what modes the camera supports
ohsb live -c configs/webcam_pose.yaml

ohsb report results/
```

For the webcam workflow — USB bandwidth, MJPG vs YUYV, stale frames — see
[docs/webcam.md](docs/webcam.md).

## Commands

| | |
|---|---|
| `ohsb run -c <config>` | offline benchmark on pre-decoded frames; pure inference cost |
| `ohsb live -c <config>` | live USB camera pipeline, with a bottleneck verdict |
| `ohsb cameras` | connected cameras and the modes they really support |
| `ohsb list` | registered tasks, sources, power backends, configs |
| `ohsb doctor` | board state, prerequisites, reproducibility warnings |
| `ohsb doctor --dump-jtop` | raw jtop payload, to pin the power extractors |
| `ohsb report <files/dirs>` | compare stored results as one table |
| `ohsb report --markdown -o F` | generate a committable summary with the board state |

Any config value can be overridden without editing the file:

```bash
ohsb run -c configs/pose_landmarker.yaml \
  --set task.delegate=gpu \
  --set task.model=models/pose_landmarker_heavy.task \
  --set source.width=640 --set source.height=480
```

`./scripts/sweep.sh configs/pose_landmarker.yaml` runs the offline delegate x
resolution grid; `./scripts/camera_matrix.sh` runs the live resolution x task
x delegate grid. Both print the comparison table when they finish.

## Tasks

| `task.type` | what it measures | model bundle |
|---|---|---|
| `pose_landmarker` | 33-point body landmarks | `pose_landmarker_{lite,full,heavy}.task` |
| `face_detector` | BlazeFace bounding boxes | `blaze_face_short_range.tflite` |
| `face_landmarker` | 478-point face mesh | `face_landmarker.task` |
| `face_recognition` | detect → crop → embed, timed per stage | detector + embedder |
| `noop` | the harness floor — no model, no MediaPipe | — |

MediaPipe ships no single face-recognition task, so `face_recognition` is
the pipeline you would actually deploy, with a **per-stage latency
breakdown**. That breakdown is the point: it shows whether detection or
embedding dominates, and how that flips as the face count rises.

Always read a real result against `configs/noop.yaml`. It is the cost of the
measurement loop itself; if a task's latency approaches it, the number is
harness overhead, not inference.

## Live mode: attributing the bottleneck

A bare fps number cannot distinguish "the camera only delivers 30" from
"the model can only process 30". So every `ohsb live` run measures a
**capture-only baseline** first, then the full pipeline, and reports which
side is the constraint:

```
  camera max   30.1 fps (capture only, no inference)
  pipeline cap 86.5 fps (convert+infer, camera removed)
  realtime     99% of camera rate
  >> CAMERA-BOUND — compute could sustain 87 fps but the camera only
     delivers 30 fps. A faster model will not help; a better camera mode
     or sensor will.
```

On a cheap UVC webcam the answer is usually USB bandwidth rather than the
Orin: the same 720p frame is typically available at 30 fps in MJPG and 10
fps in raw YUYV. `ohsb cameras` shows that envelope before you measure
anything.

## How the numbers are produced

The measurement design is the part worth reviewing:

- **Decode happens before timing.** Sources materialise every frame up
  front, so a timed interval contains inference and nothing else.
- **Warmup is mandatory.** MediaPipe defers real delegate initialisation to
  the first `detect` call; iteration 0 is routinely an order of magnitude
  slower than steady state.
- **Repeats reuse one setup.** Model load is paid once, so repeating only
  the timed loop is what exposes thermal drift — a real effect on a
  passively-clocked 15 W board.
- **Monitors bracket the timed loop only**, so average power is the
  workload's, not the workload's plus model loading.
- **Energy is `VDD_IN` × wall time** — total module input power, not the
  accelerator's share. Right for a power budget, wrong for attributing
  energy to a model. Compare against an idle baseline.
- **In live mode one clock times every stage.** Capture, convert and infer
  are bracketed by the same `perf_counter`, so the breakdown sums to the
  frame time by construction rather than by two measurements agreeing.

## Results

One run writes one JSON file containing config, full platform snapshot,
latency distribution (mean/p50/p90/p95/p99/max), throughput, per-rail power,
energy per frame, fps/W, and the raw per-iteration latencies. Nothing about
a stored result needs the shell history that produced it.

Reproducibility warnings — DVFS active, wrong governor, low free memory —
are recorded *in* the result, so a run collected under sloppy conditions
stays interpretable later; it just is not comparable to a pinned one.

`results/` is gitignored scratch space. Runs worth keeping are promoted into
[`benchmarks/`](benchmarks/), which is committed:

```bash
./scripts/publish_results.sh orin-nano-delegate-cpu-vs-gpu
```

That copies the JSON files and generates a summary carrying the comparison
table *and* the board state behind it — a published latency number without
its clock state is not a result, it is an anecdote.

## Layout

```
src/ohsb/
  cli.py         run | list | doctor | report
  runner.py      the timed loop
  config.py      YAML schema; unknown keys are rejected, not ignored
  platform.py    board snapshot + reproducibility warnings
  results.py     result schema, persistence, rendering
  camera_probe.py  what a USB camera actually supports (v4l2-ctl)
  tasks/         MediaPipe workloads (registry: @register)
  sources/       synthetic | image_dir | video | webcam (live)
  monitors/      jtop (preferred) | tegrastats (fallback)
  metrics/       latency/percentile aggregation
configs/         one YAML per benchmark
scripts/         fetch_models | setup_orin | sweep | camera_matrix | publish_results
benchmarks/      committed measurement records (results/ is scratch)
docs/            board setup and gotchas
```

Adding a task is one file plus `@register("name")`; adding a power backend
is one `Monitor` subclass.

## Development

```bash
pip install -e '.[dev]'
pytest          # 58 tests, no board, camera or MediaPipe required
ruff check src tests
```

The test suite runs off-board: the `noop` task exercises both runners, a
fake camera drives the whole live path, and the tegrastats / jtop / v4l2-ctl
parsers are locked against recorded payloads from real devices.
