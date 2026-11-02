"""Landmark-geometry gesture recognition with action debouncing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import acos, sqrt
from typing import Protocol


class HandLike(Protocol):
    """The part of a tracked hand needed for gesture recognition."""

    landmarks: tuple[tuple[float, float, float], ...]


class Gesture(str, Enum):
    MOVE = "Move cursor"
    PINCH = "Pinch"
    DRAG = "Drag"
    RIGHT_CLICK = "Right-click"
    SCROLL = "Scroll"
    PAUSED = "Paused (open palm)"
    DISABLED = "Disabled (closed fist)"
    NO_HAND = "Hand not found"
    UNKNOWN = "Unknown"


class MouseActionType(str, Enum):
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    LEFT_DOWN = "left_down"
    LEFT_UP = "left_up"
    SCROLL = "scroll"


@dataclass(frozen=True)
class MouseAction:
    """One safe, discrete instruction for the OS mouse controller."""

    type: MouseActionType
    amount: int = 0


@dataclass(frozen=True)
class GestureDecision:
    """Gesture label, cursor policy, and mouse actions for one video frame."""

    gesture: Gesture
    allows_cursor: bool
    actions: tuple[MouseAction, ...] = ()


@dataclass(frozen=True)
class FingerStates:
    """Whether each non-thumb finger is straight at its joints."""

    index: bool
    middle: bool
    ring: bool
    pinky: bool


class GestureRecognizer:
    """Translate a single hand's landmark geometry into debounced mouse actions.

    Finger extension is determined from the angle at each proximal joint plus
    the fingertip's distance from the wrist. Pinches are scaled by the palm
    size, which keeps thresholds useful at different distances from the camera.
    """

    def __init__(
        self,
        pinch_threshold: float = 0.35,
        pinch_min_seconds: float = 0.06,
        drag_hold_seconds: float = 0.45,
        right_click_debounce_ms: int = 220,
        scroll_speed: float = 45.0,
        scroll_flick_velocity_threshold: float = 0.7,
        scroll_flick_gain: float = 2.5,
        scroll_flick_max_multiplier: float = 3.0,
    ) -> None:
        if not 0.05 <= pinch_threshold <= 1.0:
            raise ValueError("pinch_threshold must be between 0.05 and 1.0.")
        if pinch_min_seconds < 0 or drag_hold_seconds <= pinch_min_seconds:
            raise ValueError("Drag hold time must be greater than pinch minimum time.")
        if right_click_debounce_ms < 0 or scroll_speed <= 0:
            raise ValueError("Debounce and scroll speed must be non-negative/positive.")
        if scroll_flick_velocity_threshold <= 0 or scroll_flick_gain < 0 or scroll_flick_max_multiplier < 1:
            raise ValueError("Scroll flick settings must be positive, non-negative, and at least one.")
        self._pinch_threshold = pinch_threshold
        self._pinch_min_ms = round(pinch_min_seconds * 1000)
        self._drag_hold_ms = round(drag_hold_seconds * 1000)
        self._right_click_debounce_ms = right_click_debounce_ms
        self._scroll_speed = scroll_speed
        self._scroll_flick_velocity_threshold = scroll_flick_velocity_threshold
        self._scroll_flick_gain = scroll_flick_gain
        self._scroll_flick_max_multiplier = scroll_flick_max_multiplier
        self._pinch_started_ms: int | None = None
        self._drag_active = False
        self._right_candidate_ms: int | None = None
        self._right_click_latched = False
        self._last_scroll_y: float | None = None
        self._last_scroll_timestamp_ms: int | None = None
        self._scroll_remainder = 0.0

    def process(self, hand: HandLike | None, timestamp_ms: int) -> GestureDecision:
        """Recognize one frame and emit only safe, debounced state changes."""
        if hand is None:
            return GestureDecision(Gesture.NO_HAND, False, self._cancel_pointer())

        landmarks = hand.landmarks
        fingers = FingerStates(
            index=self._is_finger_extended(landmarks, 5, 6, 8),
            middle=self._is_finger_extended(landmarks, 9, 10, 12),
            ring=self._is_finger_extended(landmarks, 13, 14, 16),
            pinky=self._is_finger_extended(landmarks, 17, 18, 20),
        )
        # A pinch is determined from palm-scaled thumb/index geometry. It is
        # intentionally evaluated before finger-pose gestures: curling other
        # fingers is common during a pinch and must not cancel a click or drag.
        pinch = self._is_pinch(landmarks)
        if pinch:
            self._reset_non_pinch_state()
            if self._pinch_started_ms is None:
                self._pinch_started_ms = timestamp_ms
            elapsed_ms = timestamp_ms - self._pinch_started_ms
            actions: tuple[MouseAction, ...] = ()
            if elapsed_ms >= self._drag_hold_ms and not self._drag_active:
                self._drag_active = True
                actions = (MouseAction(MouseActionType.LEFT_DOWN),)
            return GestureDecision(
                Gesture.DRAG if self._drag_active else Gesture.PINCH,
                True,
                actions,
            )

        actions = list(self._finish_pinch(timestamp_ms))
        is_scroll = fingers.index and fingers.middle and not fingers.ring and not fingers.pinky
        is_right_click = fingers.middle and not fingers.index and not fingers.ring and not fingers.pinky

        if is_scroll:
            self._reset_right_click()
            actions.extend(self._scroll_actions(landmarks, timestamp_ms))
            return GestureDecision(Gesture.SCROLL, False, tuple(actions))

        self._reset_scroll()
        if is_right_click:
            actions.extend(self._right_click_actions(timestamp_ms))
            return GestureDecision(Gesture.RIGHT_CLICK, False, tuple(actions))

        self._reset_right_click()
        if fingers.index:
            return GestureDecision(Gesture.MOVE, True, tuple(actions))
        # Open palms and closed hands are intentionally unassigned in Stage 2.
        # They neither pause nor disable the controller and emit no action.
        return GestureDecision(Gesture.UNKNOWN, False, tuple(actions))

    def _finish_pinch(self, timestamp_ms: int) -> tuple[MouseAction, ...]:
        if self._pinch_started_ms is None:
            return ()
        held_ms = timestamp_ms - self._pinch_started_ms
        self._pinch_started_ms = None
        if self._drag_active:
            self._drag_active = False
            return (MouseAction(MouseActionType.LEFT_UP),)
        if held_ms >= self._pinch_min_ms:
            return (MouseAction(MouseActionType.LEFT_CLICK),)
        return ()

    def _cancel_pointer(self) -> tuple[MouseAction, ...]:
        self._pinch_started_ms = None
        self._reset_right_click()
        self._reset_scroll()
        if self._drag_active:
            self._drag_active = False
            return (MouseAction(MouseActionType.LEFT_UP),)
        return ()

    def _reset_non_pinch_state(self) -> None:
        self._reset_right_click()
        self._reset_scroll()

    def _reset_right_click(self) -> None:
        self._right_candidate_ms = None
        self._right_click_latched = False

    def _right_click_actions(self, timestamp_ms: int) -> tuple[MouseAction, ...]:
        if self._right_click_latched:
            return ()
        if self._right_candidate_ms is None:
            self._right_candidate_ms = timestamp_ms
            return ()
        if timestamp_ms - self._right_candidate_ms >= self._right_click_debounce_ms:
            self._right_click_latched = True
            return (MouseAction(MouseActionType.RIGHT_CLICK),)
        return ()

    def _reset_scroll(self) -> None:
        self._last_scroll_y = None
        self._last_scroll_timestamp_ms = None
        self._scroll_remainder = 0.0

    def _scroll_actions(
        self,
        landmarks: tuple[tuple[float, float, float], ...],
        timestamp_ms: int,
    ) -> tuple[MouseAction, ...]:
        current_y = (landmarks[8][1] + landmarks[12][1]) / 2
        if self._last_scroll_y is None or self._last_scroll_timestamp_ms is None:
            self._last_scroll_y = current_y
            self._last_scroll_timestamp_ms = timestamp_ms
            return ()
        delta_y = self._last_scroll_y - current_y
        elapsed_seconds = max((timestamp_ms - self._last_scroll_timestamp_ms) / 1000, 0.001)
        velocity = abs(delta_y) / elapsed_seconds
        multiplier = 1.0
        if velocity > self._scroll_flick_velocity_threshold:
            multiplier = min(
                1.0 + (velocity - self._scroll_flick_velocity_threshold) * self._scroll_flick_gain,
                self._scroll_flick_max_multiplier,
            )
        self._scroll_remainder += delta_y * self._scroll_speed * multiplier
        self._last_scroll_y = current_y
        self._last_scroll_timestamp_ms = timestamp_ms
        steps = int(self._scroll_remainder)
        self._scroll_remainder -= steps
        return (MouseAction(MouseActionType.SCROLL, steps),) if steps else ()

    def _is_pinch(self, landmarks: tuple[tuple[float, float, float], ...]) -> bool:
        palm_size = self._distance(landmarks[0], landmarks[9])
        if palm_size < 0.001:
            return False
        return self._distance(landmarks[4], landmarks[8]) / palm_size <= self._pinch_threshold

    @classmethod
    def _is_finger_extended(
        cls,
        landmarks: tuple[tuple[float, float, float], ...],
        mcp: int,
        pip: int,
        tip: int,
    ) -> bool:
        """Require both a nearly straight PIP joint and an outward fingertip."""
        joint_angle = cls._angle(landmarks[mcp], landmarks[pip], landmarks[tip])
        return joint_angle >= 155.0 and cls._distance(landmarks[tip], landmarks[0]) > cls._distance(landmarks[pip], landmarks[0]) * 1.05

    @staticmethod
    def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))

    @classmethod
    def _angle(
        cls,
        first: tuple[float, float, float],
        vertex: tuple[float, float, float],
        third: tuple[float, float, float],
    ) -> float:
        first_vector = tuple(value - center for value, center in zip(first, vertex))
        third_vector = tuple(value - center for value, center in zip(third, vertex))
        first_length = sqrt(sum(value * value for value in first_vector))
        third_length = sqrt(sum(value * value for value in third_vector))
        if first_length < 0.00001 or third_length < 0.00001:
            return 0.0
        cosine = sum(a * b for a, b in zip(first_vector, third_vector)) / (first_length * third_length)
        return acos(min(1.0, max(-1.0, cosine))) * 180.0 / 3.141592653589793
