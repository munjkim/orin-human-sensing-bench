# USB webcam benchmarking

Answering "how many fps do I actually get from this camera?" needs three
numbers, not one:

| number | what it is | how to get it |
|---|---|---|
| **camera baseline** | what the camera delivers with nothing downstream | measured automatically before every `ohsb live` run |
| **pipeline capable** | what the Orin could sustain if frames arrived instantly | `convert + infer` stage means |
| **achieved** | what you actually get end to end | the run's throughput |

A single "23 fps" cannot distinguish a camera that only delivers 23 from a
model that can only process 23. `ohsb live` measures all three and prints a
verdict naming which side is the constraint.

## 1. Find out what the camera supports

Do this first. Sweeping resolutions the device does not offer just measures
the driver substituting something else.

```bash
sudo apt install v4l-utils     # for the authoritative list
ohsb cameras
```

Real output from a Logitech C920 on the Orin Nano:

```
/dev/video0  HD Pro Webcam C920  (via v4l2-ctl)
  30 fps ceiling:  H264 1920x1080  |  MJPG 1920x1080  |  YUYV 800x448
  336 modes -> 53 (max fps per resolution)
  format     resolution  max fps   note
  MJPG      1920x1080      30.0   compressed
  MJPG      1280x720       30.0   compressed
  MJPG       640x480       30.0   compressed
  YUYV      1920x1080       5.0
  YUYV      1280x720       10.0
  YUYV       800x448       30.0
  ...

/dev/video1  HD Pro Webcam C920  (via v4l2-ctl)
  metadata node, not a capture device — this is normal
```

**The `30 fps ceiling` line is the whole story.** Same 1080p frame: 30 fps
compressed, 5 fps raw. USB 2.0 gives a UVC device roughly 300 Mbit/s of
usable bandwidth; raw YUYV at 1920x1080x16bpp x 30 fps needs ~1 Gbit/s, so
the camera does not refuse — it quietly offers 5 fps instead.

A camera exposing a second `/dev/video*` node with no formats is normal:
UVC devices commonly claim one node for frames and one for per-frame
metadata.

The listing is collapsed to the fps ceiling per resolution on purpose. A
C920 advertises 336 format/size/rate combinations, and nothing is learned
from seeing 1080p at 30, 24, 20, 15, 10, 7.5 and 5 fps.

MJPG buys the bandwidth back at the cost of JPEG decode on the CPU — and on
this board that decode competes with MediaPipe for the same six A78AE
cores. Which trade wins is exactly what the benchmark is for.

## 2. Run it

```bash
ohsb live -c configs/webcam_face_detect.yaml    # cheapest face task
ohsb live -c configs/webcam_face.yaml           # 478-point mesh
ohsb live -c configs/webcam_pose.yaml           # pose landmarks
```

Output — recorded on the target board (C920, 720p, DVFS still live):

```
camera=0 MJPG 1280x720@30
[face-detect-720p-dvfs] 300 iters (+30 warmup)
  latency ms   mean=32.11  p50=32.11  p95=36.51
  throughput   29.7 fps
    stage capture    mean=26.12 ms  (78% of frame)
    stage convert    mean=0.45 ms   (1% of frame)
    stage infer      mean=7.06 ms   (21% of frame)
  camera max   30.0 fps (capture only, no inference)
  pipeline cap 133.2 fps (convert+infer, camera removed)
  realtime     99% of camera rate
  >> CAMERA-BOUND — compute could sustain 133 fps but the camera only
     delivers 30 fps. A faster model will not help; a better camera mode
     or sensor will.
```

Read `capture` at 78% of the frame together with the verdict: the loop is
sitting and waiting for the camera. That is the signature of a
camera-bound pipeline, and it means a heavier model is free until
`pipeline cap` drops near 30 — this camera reaches 30 fps at 1920x1080 too
(see `ohsb cameras`), so the default configs now capture at 1080p rather
than 720p.

Pose landmarking on the same board is the other case — compute-bound:

```
[pose-lite-720p-dvfs] 300 iters (+30 warmup)
    stage capture    mean=8.41 ms   (19% of frame)
    stage convert    mean=0.45 ms   (1% of frame)
    stage infer      mean=36.60 ms  (81% of frame)
  camera max   30.0 fps
  pipeline cap 27.0 fps
  >> COMPUTE-BOUND — the camera offers 30 fps but the pipeline sustains
     only 27 fps ('infer' is the largest stage). A lighter model, a
     smaller input, or the GPU delegate is where the gain is.
```

Same camera, same board, opposite bottleneck — which is why every config
measures its own baseline rather than assuming one answer for every task.

## 3. Sweep it

```bash
./scripts/camera_matrix.sh                        # 640x480 + 720p + 1080p, cpu + gpu
./scripts/camera_matrix.sh "1280x720 1920x1080" "cpu"
ohsb report results/camera-matrix-*/
```

The comparison table carries the camera baseline and verdict per row, so
camera-bound rows are visible at a glance:

```
run                            del        res     p50     p95     fps     cam      W     mJ/f  bottleneck
webcam-face-detect-cpu-1280x720 cpu   1280x720   32.11   36.51    29.7    30.0   5.14    172.8  CAMERA-BOUND
webcam-pose-cpu-1280x720        cpu   1280x720   44.11   48.67    22.0    30.0   5.34    243.1  COMPUTE-BOUND
```

## Things that will bite you

**Requested mode ≠ negotiated mode.** UVC drivers silently substitute the
nearest supported mode rather than failing. The result records both and
raises a warning when they differ — check for `! driver substituted mode`
before trusting a row.

**Auto-exposure ramp.** A UVC camera takes a second or two to settle its
exposure and gain, and frames during the ramp are both darker and slower.
`camera.settle_frames` (default 30) discards them.

**Stale frames.** With the default V4L2 queue depth, a consumer slower than
the camera reads progressively older frames while appearing to keep up —
throughput looks fine while latency-to-reality grows without bound. The
configs pin `buffersize: 1`. If your backend ignores that, set
`camera.drain: true` to grab-drain to the newest frame before each read;
the run then reports how many frames it skipped.

Note that `drain: true` changes the question from "can I process every
frame?" to "how fresh is the frame I process?". Both are legitimate; do not
compare a drained run against an undrained one.

**`running_mode: video` vs `image`.** The webcam configs use `video`, which
lets MediaPipe reuse tracking between frames — what a real camera app does,
and usually faster than re-detecting each frame. Use `image` if you want
per-frame cost independent of tracking state, but do not mix the two in one
comparison.

**Resolution cuts both ways.** A smaller capture is less USB bandwidth
*and* less to convert and infer, so 640x480 can be more than 2x faster than
720p. But MediaPipe resizes internally to the model's input size anyway, so
past a point you are only paying for capture and conversion — which is why
the stage breakdown matters more than the total.
