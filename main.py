"""Stage 2 launcher: hand tracking, cursor movement, and mouse gestures."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import cv2

    from controller.mouse_controller import MouseController
    from utils.coordinate_mapping import ActiveArea
    from vision.gesture_recognition import GestureRecognizer
    from vision.hand_tracker import HandTracker, draw_hands
except ModuleNotFoundError as dependency_error:
    _DEPENDENCY_ERROR: ModuleNotFoundError | None = dependency_error
else:
    _DEPENDENCY_ERROR = None

PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"


def load_settings() -> dict[str, Any]:
    """Load settings with a clear error for invalid JSON."""
    try:
        with SETTINGS_PATH.open(encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {SETTINGS_PATH}: {error.msg}") from error


def draw_active_area(frame: Any, area: ActiveArea) -> None:
    """Overlay the region that maps to the entire desktop."""
    height, width = frame.shape[:2]
    top_left = (round(area.left * (width - 1)), round(area.top * (height - 1)))
    bottom_right = (round(area.right * (width - 1)), round(area.bottom * (height - 1)))
    cv2.rectangle(frame, top_left, bottom_right, (230, 180, 50), 2)


def run() -> int:
    if _DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "A required package is missing. Use Python 3.13 and run "
            "'python -m pip install -r requirements.txt'."
        ) from _DEPENDENCY_ERROR
    settings = load_settings()
    active_area = ActiveArea(**settings["active_area"])
    camera = cv2.VideoCapture(int(settings["camera_index"]), cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(int(settings["camera_index"]))
    if not camera.isOpened():
        raise RuntimeError("Unable to open the configured webcam. Check camera_index in settings.json.")

    mouse: MouseController | None = None
    previous_time = time.perf_counter()
    try:
        mouse = MouseController(
            active_area=active_area,
            sensitivity=float(settings["cursor_sensitivity"]),
            smoothing=float(settings["smoothing"]),
        )
        recognizer = GestureRecognizer(
            pinch_threshold=float(settings["pinch_threshold"]),
            pinch_min_seconds=float(settings["pinch_min_seconds"]),
            drag_hold_seconds=float(settings["drag_hold_seconds"]),
            right_click_debounce_ms=int(settings["right_click_debounce_ms"]),
            scroll_speed=float(settings["scroll_speed"]),
            scroll_flick_velocity_threshold=float(settings["scroll_flick_velocity_threshold"]),
            scroll_flick_gain=float(settings["scroll_flick_gain"]),
            scroll_flick_max_multiplier=float(settings["scroll_flick_max_multiplier"]),
        )
        with HandTracker(
            model_path=MODEL_PATH,
            max_hands=int(settings["max_hands"]),
            detection_confidence=float(settings["detection_confidence"]),
            presence_confidence=float(settings["presence_confidence"]),
            tracking_confidence=float(settings["tracking_confidence"]),
        ) as tracker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Webcam frame capture failed.")
                if settings["mirror_preview"]:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = round(time.perf_counter() * 1000)
                hands = tracker.detect(frame, timestamp_ms)
                decision = recognizer.process(hands[0] if hands else None, timestamp_ms)
                mouse.apply_actions(decision.actions)
                if hands and decision.allows_cursor:
                    mouse.move_to_landmark(hands[0].index_tip)
                draw_hands(frame, hands)
                draw_active_area(frame, active_area)

                now = time.perf_counter()
                fps = 1.0 / max(now - previous_time, 0.0001)
                previous_time = now
                cv2.putText(
                    frame,
                    f"{decision.gesture.value.upper()} | {fps:.1f} FPS | q: quit",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Hand Controller - Stage 2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if mouse is not None:
            mouse.release_all()
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
