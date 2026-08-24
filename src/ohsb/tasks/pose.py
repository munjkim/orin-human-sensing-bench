"""MediaPipe Pose Landmarker."""

from __future__ import annotations

from ..sources.base import Frame
from . import register
from .base import InferResult, Task
from .mediapipe_common import base_options, import_mediapipe, running_mode, take, to_mp_image


@register("pose_landmarker")
class PoseLandmarkerTask(Task):
    """33-point body landmarks. Model bundle: ``pose_landmarker_{lite,full,heavy}.task``."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._landmarker = None
        self._ts = 0

    def setup(self) -> None:
        _, _, vision = import_mediapipe()
        opts = self.cfg.options
        options = vision.PoseLandmarkerOptions(
            base_options=base_options(self.cfg, self.cfg.type),
            running_mode=running_mode(self.cfg),
            num_poses=take(opts, "num_poses", 1),
            min_pose_detection_confidence=take(opts, "min_pose_detection_confidence", 0.5),
            min_pose_presence_confidence=take(opts, "min_pose_presence_confidence", 0.5),
            min_tracking_confidence=take(opts, "min_tracking_confidence", 0.5),
            output_segmentation_masks=take(opts, "output_segmentation_masks", False),
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def infer(self, frame: Frame) -> InferResult:
        image = to_mp_image(frame.image)
        if self.cfg.running_mode == "video":
            self._ts += 33
            result = self._landmarker.detect_for_video(image, self._ts)
        else:
            result = self._landmarker.detect(image)
        return InferResult(payload=result, detections=len(result.pose_landmarks))

    def teardown(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
