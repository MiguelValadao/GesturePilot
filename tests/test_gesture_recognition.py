"""Tests for Stage 2 gesture state transitions without camera dependencies."""

from dataclasses import dataclass
import unittest

from vision.gesture_recognition import Gesture, GestureRecognizer, MouseActionType


@dataclass(frozen=True)
class TestHand:
    landmarks: tuple[tuple[float, float, float], ...]


def make_hand(extended: set[str], pinch: bool = False, y_offset: float = 0.0) -> TestHand:
    """Build simple, geometrically valid normalized landmarks for a test hand."""
    points = [(0.5, 0.8, 0.0) for _ in range(21)]
    points[0] = (0.5, 0.8, 0.0)
    fingers = {
        "index": (5, 6, 8, 0.40),
        "middle": (9, 10, 12, 0.55),
        "ring": (13, 14, 16, 0.67),
        "pinky": (17, 18, 20, 0.77),
    }
    for name, (mcp, pip, tip, x) in fingers.items():
        points[mcp] = (x, 0.62 + y_offset, 0.0)
        points[pip] = (x, 0.50 + y_offset, 0.0)
        tip_y = 0.28 + y_offset if name in extended else 0.61 + y_offset
        points[tip] = (x, tip_y, 0.0)
    points[4] = (0.12, 0.50, 0.0)
    if pinch:
        index_tip = points[8]
        points[4] = (index_tip[0] + 0.01, index_tip[1], 0.0)
    return TestHand(tuple(points))


class GestureRecognizerTests(unittest.TestCase):
    def test_short_pinch_release_clicks_once(self) -> None:
        recognizer = GestureRecognizer()
        recognizer.process(make_hand({"index"}, pinch=True), 0)
        result = recognizer.process(make_hand({"index"}), 100)
        self.assertEqual(result.gesture, Gesture.MOVE)
        self.assertEqual([action.type for action in result.actions], [MouseActionType.LEFT_CLICK])

    def test_pinch_hold_starts_and_releases_drag(self) -> None:
        recognizer = GestureRecognizer()
        recognizer.process(make_hand({"index"}, pinch=True), 0)
        held = recognizer.process(make_hand({"index"}, pinch=True), 500)
        released = recognizer.process(make_hand({"index"}), 600)
        self.assertEqual(held.gesture, Gesture.DRAG)
        self.assertEqual([action.type for action in held.actions], [MouseActionType.LEFT_DOWN])
        self.assertEqual([action.type for action in released.actions], [MouseActionType.LEFT_UP])

    def test_right_click_requires_stable_gesture(self) -> None:
        recognizer = GestureRecognizer(right_click_debounce_ms=100)
        recognizer.process(make_hand({"middle"}), 0)
        result = recognizer.process(make_hand({"middle"}), 100)
        self.assertEqual([action.type for action in result.actions], [MouseActionType.RIGHT_CLICK])

    def test_scroll_uses_two_extended_fingers(self) -> None:
        recognizer = GestureRecognizer(scroll_speed=50, scroll_flick_velocity_threshold=100)
        recognizer.process(make_hand({"index", "middle"}), 0)
        result = recognizer.process(make_hand({"index", "middle"}, y_offset=-0.10), 20)
        self.assertEqual(result.gesture, Gesture.SCROLL)
        self.assertEqual(result.actions[0].type, MouseActionType.SCROLL)
        self.assertEqual(result.actions[0].amount, 5)

    def test_fast_two_finger_motion_gets_a_capped_scroll_boost(self) -> None:
        recognizer = GestureRecognizer(
            scroll_speed=50,
            scroll_flick_velocity_threshold=0.7,
            scroll_flick_gain=2.5,
            scroll_flick_max_multiplier=3.0,
        )
        recognizer.process(make_hand({"index", "middle"}), 0)
        result = recognizer.process(make_hand({"index", "middle"}, y_offset=-0.10), 20)
        self.assertEqual(result.actions[0].type, MouseActionType.SCROLL)
        self.assertEqual(result.actions[0].amount, 15)

    def test_closed_hand_is_unassigned(self) -> None:
        result = GestureRecognizer().process(make_hand(set()), 0)
        self.assertEqual(result.gesture, Gesture.UNKNOWN)
        self.assertFalse(result.allows_cursor)
        self.assertEqual(result.actions, ())


if __name__ == "__main__":
    unittest.main()
