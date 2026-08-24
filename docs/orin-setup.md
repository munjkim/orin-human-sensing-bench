# Orin Nano setup

Target board, as captured in `jetson-orin-nano-spec/`:

| | |
|---|---|
| Module | Jetson Orin Nano 8GB (P3767-0005) on P3768 dev kit carrier |
| JetPack | 5.1.1 (L4T 35.3.1) |
| OS | Ubuntu 20.04 focal, kernel 5.10.104-tegra |
| Python | **3.8.10** (system) |
| CPU | 6x Cortex-A78AE, 1.5 GHz |
| iGPU | ga10b, 306 MHz idle → 624 MHz max |
| RAM | 6.3 GB usable, **shared with the iGPU** |
| CUDA / TensorRT | 11.4.315 / 8.5.2.2 |
| OpenCV | 4.5.4, **built without CUDA** |
| Power rails | `VDD_IN` (total), `VDD_CPU_GPU_CV`, `VDD_SOC` |
| Default state | 15 W mode, `jetson_clocks` inactive, GPU DVFS active |

Anything in this repo that looks over-defensive traces back to a row above.

## 1. Python 3.8 is the constraint

The package targets 3.8 because that is what JetPack 5.1.1 ships. Do not
reach for 3.9+ syntax (`dict[str, int]` outside annotations, `X | Y` unions,
`match`); CI on your dev machine will not catch it, the board will.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

`--system-site-packages` matters: it lets the venv see JetPack's system
`python3-opencv` (4.5.4) instead of pulling a CPU-only wheel from PyPI.

## 2. MediaPipe

`pip install mediapipe` may or may not resolve an aarch64 wheel for your
Python version. Check first:

```bash
pip download mediapipe --no-deps -d /tmp/mpcheck
```

If there is no wheel, the usual paths are a community aarch64 build or
building from source. Confirm whichever you land on:

```bash
ohsb doctor        # reports the mediapipe version it can import
```

Note that MediaPipe's GPU delegate on Jetson runs through **OpenGL ES
compute shaders, not CUDA or TensorRT**. `delegate: gpu` is therefore not a
TensorRT comparison, and on small models it can lose to CPU — the upload,
shader dispatch and readback can cost more than the inference saves. That
result is worth measuring, not worth assuming.

## 3. Put the board in a repeatable state

This is the single biggest source of unreproducible numbers. Out of the box
the GPU scales 306→624 MHz mid-run.

```bash
./scripts/setup_orin.sh              # dry run: prints what it would change
sudo ./scripts/setup_orin.sh --apply
```

It sets `nvpmodel -m 0` (15 W), runs `jetson_clocks` to pin CPU/GPU/EMC at
max, and forces the fan to full — `jetson_clocks` disables the thermal fan
curve, so without the last step a long sweep can throttle silently.

Optionally free the ~3 GB the desktop session holds:

```bash
sudo systemctl isolate multi-user.target   # drop to console
sudo systemctl start graphical.target      # restore afterwards
```

`ohsb doctor` warns about each of these if you skip them, and every warning
is stored in the result file, so a run collected under DVFS is still
interpretable later — just not comparable to a pinned one.

## 4. Power measurement

Two backends, selected by `monitor.power.backend`:

- **`jtop`** (preferred, and what `auto` picks first) — jetson-stats is
  already installed on this board. It reads the INA3221 rails through a root
  daemon, so sampling needs no sudo from us.
- **`tegrastats`** (fallback) — parsed from its text output. Needs root on
  most JetPack images:

  ```bash
  echo "$USER ALL=(root) NOPASSWD: /usr/bin/tegrastats" | sudo tee /etc/sudoers.d/ohsb-tegrastats
  ```

Reported energy comes from **`VDD_IN`, the module input rail**. That is
total board draw — CPU, display, everything — not the accelerator's share.
It is the right number for an edge power budget and the wrong number for
attributing energy to a model. Compare runs against each other, and always
against an idle baseline:

```bash
ohsb run -c configs/noop.yaml -n idle-baseline
```

### Pinning the jtop schema

jetson-stats payload shapes have changed between major versions, and this
board runs a newer jtop than the extractors in `monitors/jtop_monitor.py`
were written against. They are deliberately tolerant, but confirm the real
shape once on the board:

```bash
ohsb doctor --dump-jtop > docs/jtop-schema-observed.json
```

If a rail or the GPU load is missing from a run's output, that dump is what
tells you which key moved.

## 5. Sanity check before trusting anything

```bash
ohsb run -c configs/noop.yaml           # harness floor, no model
ohsb doctor                             # platform state + prerequisites
```

The `noop` run is the measurement floor. If it reports 0.05 ms and a task
reports 1.2 ms, harness overhead is ~4% and the task number is meaningful.
If `noop` ever creeps into the same order of magnitude as a real task, stop
and fix the harness before reading anything else.
