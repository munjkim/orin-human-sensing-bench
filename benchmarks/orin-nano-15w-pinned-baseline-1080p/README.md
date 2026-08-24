| run | task | delegate | resolution | p50 ms | p95 ms | fps | camera fps | W | mJ/frame | bottleneck |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `noop-pinned` | noop | cpu | 1280x720 | 0.00 | 0.00 | 445814.4 | - | 7.18 | 0.0 | - |
| `face-detect-1080p-pinned` | face_detector | cpu | 1920x1080 | 33.25 | 39.69 | 29.4 | 30.0 | 6.59 | 224.2 | CAMERA-BOUND |
| `face-mesh-1080p-pinned` | face_landmarker | cpu | 1920x1080 | 49.97 | 55.49 | 19.6 | 30.0 | 6.31 | 321.2 | BALANCED |
| `pose-lite-1080p-pinned` | pose_landmarker | cpu | 1920x1080 | 55.05 | 60.84 | 17.7 | 29.0 | 6.99 | 394.7 | COMPUTE-BOUND |

## Board state

| | |
|---|---|
| model | NVIDIA Orin Nano Developer Kit |
| L4T | 35.3.1 |
| power mode | 15W |
| jetson_clocks (GPU, via jtop) | pinned |
| CPU frequency | pinned (6/6 cores) |
| GPU freq (sysfs, informational only) | 625 MHz, cur 625 (devfreq governor nvhost_podgov) |
| python | 3.8.10 |
| mediapipe | 0.10.9 |
| opencv | 5.0.0 |

## Reproducibility warnings

- GPU devfreq governor is 'nvhost_podgov' (DVFS active); frequency will vary with load
- nvpmodel mode is '15W' — record it; it caps achievable clocks
