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
pip install -U pip setuptools wheel     # required — see below
pip install -e '.[dev]'
```

`--system-site-packages` matters: it lets the venv see JetPack's system
`python3-opencv` (4.5.4) instead of pulling a CPU-only wheel from PyPI.

**Upgrading pip first is not optional here.** Ubuntu 20.04 ships pip 20.0.2,
which predates PEP 660 and so cannot install a `pyproject.toml`-only project
in editable mode:

```
ERROR: File "setup.py" not found. Directory cannot be installed in editable mode
(A "pyproject.toml" file was found, but editable mode currently requires a
setup.py based build.)
```

PEP 660 landed in pip 21.3. On Python 3.8 the upgrade resolves to pip 24.3.1
(pip 25.0 dropped 3.8), which is fine. If you cannot upgrade pip — offline
box, locked-down proxy — install non-editable instead; old pip still does
PEP 517 builds, you just have to reinstall after editing the source:

```bash
pip install '.[dev]'
```

## 2. MediaPipe

**On this board, install exactly this:**

```bash
pip install 'mediapipe==0.10.9'
```

Or via the extra, which encodes the same constraint:

```bash
pip install -e '.[mediapipe]'
```

~112 MB of wheels, all prebuilt — nothing compiles.

### Why the version is pinned rather than lower-bounded

Three facts about what PyPI actually publishes for `linux aarch64`:

| release | aarch64 wheel | works on Python 3.8? |
|---|---|---|
| 0.10.5 – 0.10.9 | `cp38-cp38-manylinux_2_17_aarch64` | **yes** |
| 0.10.13 – 0.10.18 | cp39/310/311/312 only | no — cp38 tag was dropped |
| 1.0.x | `py3-none-manylinux_2_28_aarch64` | installs, then fails |

That last row is the trap. The `py3-none` tag claims any Python 3, and
`manylinux_2_28` needs glibc ≥ 2.28 which Ubuntu 20.04 satisfies (2.31) — so
`pip install mediapipe` on this board resolves to **1.0.1**, installs
cleanly, and only fails when you import it. `mediapipe>=0.10.9` has the same
problem. Pin it.

`0.10.9` is the last release with a real cp38 aarch64 wheel, and it has the
lighter dependency set: 0.10.13+ pulls in `jax` and `jaxlib`, which you do
not want on a 6.3 GB shared-memory board.

### What comes with it

`mediapipe` depends on `opencv-contrib-python`, which resolves to a
prebuilt `cp37-abi3` aarch64 wheel (4.8.1.78) — no source build. Because the
venv uses `--system-site-packages`, this pip build **shadows JetPack's system
OpenCV 4.5.4** inside the venv. For this harness that is fine: capture uses
`cv2.CAP_V4L2`, colour conversion and resize, all of which the pip wheel
supports.

Be aware of what the pip wheel does *not* have, in case you extend this
later: no CUDA, and no GStreamer. A CSI camera via `nvarguscamerasrc` needs
JetPack's OpenCV, so that path would require `--no-deps` and keeping the
system build:

```bash
pip install --no-deps 'mediapipe==0.10.9'
pip install protobuf==3.20.3 absl-py attrs flatbuffers matplotlib sounddevice
```

USB UVC cameras — what this repo benchmarks — do not need that.

### Confirm it

```bash
ohsb doctor        # reports the mediapipe version it can import
```

### The GPU delegate is not TensorRT

MediaPipe's GPU delegate on Jetson runs through **OpenGL ES compute shaders,
not CUDA or TensorRT**. `delegate: gpu` is therefore not a TensorRT
comparison, and on small models it can lose to CPU — the upload, shader
dispatch and readback can cost more than the inference saves. That result is
worth measuring, not worth assuming.

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

## 5. Model bundles

```bash
./scripts/fetch_models.sh
```

~53 MB across six bundles. JetPack images do not always ship `curl`; the
script uses `wget` when curl is absent and tells you what to install if
neither is there. To fetch them by hand instead:

```bash
./scripts/fetch_models.sh --print-urls
```

Downloads are verified by size — a rotated URL can return an HTML error page
with a 200 status, and a 2 kB "model" that fails at load time is far worse
than a failed download.

## 6. Sanity check before trusting anything

```bash
ohsb run -c configs/noop.yaml           # harness floor, no model
ohsb doctor                             # platform state + prerequisites
```

To hand the whole picture to someone else — or to your future self — capture
it in one shot instead of pasting terminal output:

```bash
./scripts/collect_env.sh
git add env && git commit -m "Add board environment snapshot" && git push
```

The `noop` run is the measurement floor. If it reports 0.05 ms and a task
reports 1.2 ms, harness overhead is ~4% and the task number is meaningful.
If `noop` ever creeps into the same order of magnitude as a real task, stop
and fix the harness before reading anything else.
