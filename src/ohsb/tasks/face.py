"""MediaPipe face tasks: detection, landmarks, and a recognition pipeline."""

from __future__ import annotations

import time

import numpy as np

from ..sources.base import Frame
from . import register
from .base import InferResult, Task
from .mediapipe_common import (
    base_options,
    import_mediapipe,
    resolve_model,
    running_mode,
    take,
    to_mp_image,
)


@register("face_detector")
class FaceDetectorTask(Task):
    """BlazeFace bounding boxes. Model: ``blaze_face_short_range.tflite``."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._detector = None
        self._ts = 0

    def setup(self) -> None:
        _, _, vision = import_mediapipe()
        opts = self.cfg.options
        options = vision.FaceDetectorOptions(
            base_options=base_options(self.cfg, self.cfg.type),
            running_mode=running_mode(self.cfg),
            min_detection_confidence=take(opts, "min_detection_confidence", 0.5),
            min_suppression_threshold=take(opts, "min_suppression_threshold", 0.3),
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def infer(self, frame: Frame) -> InferResult:
        image = to_mp_image(frame.image)
        if self.cfg.running_mode == "video":
            self._ts += 33
            result = self._detector.detect_for_video(image, self._ts)
        else:
            result = self._detector.detect(image)
        return InferResult(payload=result, detections=len(result.detections))

    def teardown(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None


@register("face_landmarker")
class FaceLandmarkerTask(Task):
    """478-point face mesh (+ optional blendshapes). Model: ``face_landmarker.task``."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._landmarker = None
        self._ts = 0

    def setup(self) -> None:
        _, _, vision = import_mediapipe()
        opts = self.cfg.options
        options = vision.FaceLandmarkerOptions(
            base_options=base_options(self.cfg, self.cfg.type),
            running_mode=running_mode(self.cfg),
            num_faces=take(opts, "num_faces", 1),
            min_face_detection_confidence=take(opts, "min_face_detection_confidence", 0.5),
            min_face_presence_confidence=take(opts, "min_face_presence_confidence", 0.5),
            min_tracking_confidence=take(opts, "min_tracking_confidence", 0.5),
            output_face_blendshapes=take(opts, "output_face_blendshapes", False),
            output_facial_transformation_matrixes=take(
                opts, "output_facial_transformation_matrixes", False
            ),
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def infer(self, frame: Frame) -> InferResult:
        image = to_mp_image(frame.image)
        if self.cfg.running_mode == "video":
            self._ts += 33
            result = self._landmarker.detect_for_video(image, self._ts)
        else:
            result = self._landmarker.detect(image)
        return InferResult(payload=result, detections=len(result.face_landmarks))

    def teardown(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


@register("face_recognition")
class FaceRecognitionTask(Task):
    """Detect -> crop -> embed, timed as three stages.

    MediaPipe ships no single face-recognition task, so identity is measured
    as the realistic pipeline: ``FaceDetector`` locates faces, each crop is
    passed to ``ImageEmbedder``. The per-stage breakdown is what tells you
    whether detection or embedding dominates at a given face count.

    Config::

        task:
          type: face_recognition
          model: models/blaze_face_short_range.tflite   # detector
          options:
            embedder_model: models/mobilenet_v3_small_embedder.tflite
            max_faces: 4
            crop_margin: 0.2
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._detector = None
        self._embedder = None

    def setup(self) -> None:
        _, mp_python, vision = import_mediapipe()
        opts = self.cfg.options

        self._detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=base_options(self.cfg, self.cfg.type),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=take(opts, "min_detection_confidence", 0.5),
            )
        )

        embedder_model = opts.get("embedder_model")
        if not embedder_model:
            raise ValueError(
                "face_recognition needs task.options.embedder_model "
                "(e.g. models/mobilenet_v3_small_embedder.tflite)"
            )
        delegate = (
            mp_python.BaseOptions.Delegate.GPU
            if self.cfg.delegate == "gpu"
            else mp_python.BaseOptions.Delegate.CPU
        )
        self._embedder = vision.ImageEmbedder.create_from_options(
            vision.ImageEmbedderOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(resolve_model(embedder_model, "face_recognition")),
                    delegate=delegate,
                ),
                running_mode=vision.RunningMode.IMAGE,
                l2_normalize=take(opts, "l2_normalize", True),
                quantize=take(opts, "quantize", False),
            )
        )
        self._max_faces = take(opts, "max_faces", 4)
        self._margin = take(opts, "crop_margin", 0.2)

    def infer(self, frame: Frame) -> InferResult:
        stages = {}

        t0 = time.perf_counter()
        det = self._detector.detect(to_mp_image(frame.image))
        t1 = time.perf_counter()
        stages["detect"] = (t1 - t0) * 1e3

        boxes = [d.bounding_box for d in det.detections][: self._max_faces]
        crops = [_crop(frame.image, b, self._margin) for b in boxes]
        crops = [c for c in crops if c.size]
        t2 = time.perf_counter()
        stages["crop"] = (t2 - t1) * 1e3

        embeddings = [self._embedder.embed(to_mp_image(c)) for c in crops]
        stages["embed"] = (time.perf_counter() - t2) * 1e3

        return InferResult(payload=embeddings, detections=len(crops), stages=stages)

    def teardown(self) -> None:
        for attr in ("_detector", "_embedder"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.close()
                setattr(self, attr, None)


def _crop(image: np.ndarray, box, margin: float) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, bw, bh = box.origin_x, box.origin_y, box.width, box.height
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(w, x0 + bw + 2 * mx), min(h, y0 + bh + 2 * my)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return np.ascontiguousarray(image[y0:y1, x0:x1])
