| run | task | delegate | resolution | p50 ms | p95 ms | fps | camera fps | W | mJ/frame | bottleneck |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `noop-dvfs` | noop | cpu | 1280x720 | 0.00 | 0.00 | 499619.8 | - | 5.10 | 0.0 | - |
| `face-detect-720p-dvfs` | face_detector | cpu | 1280x720 | 32.11 | 36.51 | 29.7 | 30.0 | 5.14 | 172.8 | CAMERA-BOUND |
| `pose-lite-720p-dvfs` | pose_landmarker | cpu | 1280x720 | 44.11 | 48.67 | 22.0 | 30.0 | 5.34 | 243.1 | COMPUTE-BOUND |

## Board state

| | |
|---|---|
| model | NVIDIA Orin Nano Developer Kit |
| L4T | 35.3.1 |
| power mode | 15W |
| jetson_clocks | INACTIVE (DVFS live) |
| GPU freq | pinned at 625 MHz |
| python | 3.8.10 |
| mediapipe | 0.10.9 |
| opencv | 5.0.0 |

## Reproducibility warnings

- jetson_clocks is inactive (GPU devfreq governor is 'nvhost_podgov', so DVFS is live): GPU and CPU frequencies will scale during the run. Run `sudo jetson_clocks` to pin them before collecting comparable numbers.
- CPU scaling governor is ['schedutil'], not 'performance'
- nvpmodel mode is '15W' — record it; it caps achievable clocks
