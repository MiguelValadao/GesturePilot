"""MediaPipe Tasks hand tracking and OpenCV landmark rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np


# Landmark indices defined by MediaPipe's 21-point hand landmark topology.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
INDEX_TIP = 8


@dataclass(frozen=True)
class HandObservation:
    """One detected hand, with coordinates normalized to the input frame."""

    landmarks: tuple[tuple[float, float, float], ...]
    handedness: str
    score: float

    @property
    def index_tip(self) -> tuple[float, float]:
        return self.landmarks[INDEX_TIP][:2]


class HandTracker:
    """Owns a MediaPipe Tasks HandLandmarker configured for video frames."""

    def __init__(
        self,
        model_path: Path,
        max_hands: int = 1,
        detection_confidence: float = 0.6,
        presence_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmark model is missing: {model_path}. See README.md for setup."
            )
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandObservation]:
        """Detect hands in a BGR camera frame using a monotonic timestamp."""
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        observations: list[HandObservation] = []
        for index, landmarks in enumerate(result.hand_landmarks):
            categories = result.handedness[index] if index < len(result.handedness) else []
            category = categories[0] if categories else None
            observations.append(
                HandObservation(
                    landmarks=tuple((item.x, item.y, item.z) for item in landmarks),
                    handedness=category.category_name if category else "Unknown",
                    score=category.score if category else 0.0,
                )
            )
        return observations

    def close(self) -> None:
        """Release native MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def draw_hands(frame_bgr: np.ndarray, hands: Sequence[HandObservation]) -> np.ndarray:
    """Draw landmark points, connections, and the index fingertip on a frame."""
    height, width = frame_bgr.shape[:2]
    for hand in hands:
        pixels = [(round(x * (width - 1)), round(y * (height - 1))) for x, y, _ in hand.landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pixels[start], pixels[end], (118, 255, 102), 2, cv2.LINE_AA)
        for pixel in pixels:
            cv2.circle(frame_bgr, pixel, 3, (255, 211, 67), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, pixels[INDEX_TIP], 7, (50, 120, 255), 2, cv2.LINE_AA)
        label = f"{hand.handedness} {hand.score:.0%}"
        cv2.putText(frame_bgr, label, (pixels[0][0] + 10, pixels[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr

