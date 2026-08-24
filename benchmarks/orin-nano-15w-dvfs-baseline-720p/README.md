| run | task | delegate | resolution | p50 ms | p95 ms | fps | camera fps | W | mJ/frame | bottleneck |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `noop-dvfs` | noop | cpu | n/a | 0.00 | 0.00 | 499619.8 | - | 5.10 | 0.0 | harness floor |
| `face-detect-720p-dvfs` | face_detector | cpu | 1280x720 | 32.11 | 36.51 | 29.7 | 30.0 | 5.14 | 172.8 | CAMERA-BOUND |
| `pose-lite-720p-dvfs` | pose_landmarker | cpu | 1280x720 | 44.11 | 48.67 | 22.0 | 30.0 | 5.34 | 243.1 | COMPUTE-BOUND |
| `face-mesh-720p-dvfs` | face_landmarker | cpu | 1280x720 | 36.37 | 50.17 | 25.8 | 30.0 | 5.63 | 218.1 | CAMERA-BOUND |

*`n/a` rows are the `noop` harness-calibration task: no camera, no model, synthetic frames only. It measures what the benchmark loop itself costs so every other row can be read against that floor — its resolution and bottleneck verdict are not comparable to a camera run's.*

## Board state

| | |
|---|---|
| model | NVIDIA Orin Nano Developer Kit |
| L4T | 35.3.1 |
| power mode | 15W |
| jetson_clocks --show (pre-run, best-effort) | INACTIVE (DVFS live) |
| GPU clock during these runs | not measured (no run had >=2 GPU frequency samples) |
| CPU frequency | - (0/6 cores) |
| GPU freq (sysfs, informational only) | 625 MHz, cur 625 (devfreq governor nvhost_podgov) |
| python | 3.8.10 |
| mediapipe | 0.10.9 |
| opencv | 5.0.0 |

## Reproducibility warnings

- jetson_clocks is inactive (GPU devfreq governor is 'nvhost_podgov', so DVFS is live): GPU and CPU frequencies will scale during the run. Run `sudo jetson_clocks` to pin them before collecting comparable numbers.
- CPU scaling governor is ['schedutil'], not 'performance'
- nvpmodel mode is '15W' — record it; it caps achievable clocks
- GPU devfreq governor is 'nvhost_podgov' (DVFS active); frequency will vary with load
- cannot verify whether jetson_clocks is active from a static probe on this board (every such probe tried has been unreliable — see docs/orin-setup.md). Run `sudo jetson_clocks` before benchmarking, and check each RESULT's `dvfs` field afterward: it reports whether the GPU clock actually varied during that specific run, which is the trustworthy version of this check.
