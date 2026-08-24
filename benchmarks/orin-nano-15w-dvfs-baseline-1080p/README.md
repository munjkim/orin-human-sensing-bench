| run | task | delegate | resolution | p50 ms | p95 ms | fps | camera fps | W | mJ/frame | bottleneck |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `noop-dvfs` | noop | cpu | 1280x720 | 0.00 | 0.00 | 414578.9 | - | 6.81 | 0.0 | - |
| `face-detect-1080p-dvfs` | face_detector | cpu | 1920x1080 | 32.33 | 36.84 | 29.7 | 29.0 | 6.46 | 217.3 | CAMERA-BOUND |
| `face-mesh-1080p-dvfs` | face_landmarker | cpu | 1920x1080 | 49.84 | 53.94 | 19.7 | 30.0 | 6.34 | 322.5 | BALANCED |
| `pose-lite-1080p-dvfs` | pose_landmarker | cpu | 1920x1080 | 55.69 | 61.42 | 17.5 | 30.0 | 6.70 | 382.7 | COMPUTE-BOUND |

## Board state

| | |
|---|---|
| model | NVIDIA Orin Nano Developer Kit |
| L4T | 35.3.1 |
| power mode | 15W |
| jetson_clocks --show (pre-run, best-effort) | pinned |
| GPU clock during these runs | not measured (no run had >=2 GPU frequency samples) |
| CPU frequency | pinned (6/6 cores) |
| GPU freq (sysfs, informational only) | 625 MHz, cur 625 (devfreq governor nvhost_podgov) |
| python | 3.8.10 |
| mediapipe | 0.10.9 |
| opencv | 5.0.0 |

## Reproducibility warnings

- GPU devfreq governor is 'nvhost_podgov' (DVFS active); frequency will vary with load
- nvpmodel mode is '15W' — record it; it caps achievable clocks
