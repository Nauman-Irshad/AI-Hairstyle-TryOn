"""
Step 1: Face landmarks via MediaPipe Tasks Face Landmarker (replaces legacy mp.solutions.face_mesh).

MediaPipe 0.10+ no longer exposes `mediapipe.solutions`; use `mediapipe.tasks.vision.FaceLandmarker`.
"""

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

LOGGER = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class FaceLandmarkerRunner:
    """
    Wraps Face Landmarker (IMAGE mode) with a local .task model file.
    """

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Face landmarker model missing: {model_path}")
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path.resolve())),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def process(self, image_rgb: np.ndarray):
        """
        image_rgb: HxWx3 uint8 RGB, contiguous.
        Returns mediapipe FaceLandmarkerResult (has .face_landmarks).
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        if not image_rgb.flags["C_CONTIGUOUS"]:
            image_rgb = np.ascontiguousarray(image_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self._landmarker.detect(mp_image)

    def close(self) -> None:
        self._landmarker.close()


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_face_landmarks(image_bgr: np.ndarray, runner: FaceLandmarkerRunner):
    """
    Input BGR uint8 -> FaceLandmarkerResult.
    Raises ValueError if no face detected.
    """
    rgb = bgr_to_rgb(image_bgr)
    res = runner.process(rgb)
    if res is None or not res.face_landmarks:
        raise ValueError("No face detected — please use a clear frontal face photo.")
    return res


# Backwards-compatible name used by pipeline
def detect_face_mesh(image_bgr: np.ndarray, runner: FaceLandmarkerRunner):
    """Alias for detect_face_landmarks (legacy pipeline name)."""
    return detect_face_landmarks(image_bgr, runner)
