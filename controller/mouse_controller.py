"""Safe operating-system mouse controller."""

from __future__ import annotations

from utils.coordinate_mapping import ActiveArea, CursorMapper, CursorSmoother
from vision.gesture_recognition import MouseAction, MouseActionType


class MouseController:
    """Applies debounced gesture actions while preserving PyAutoGUI failsafe."""

    def __init__(
        self,
        active_area: ActiveArea,
        sensitivity: float = 1.0,
        smoothing: float = 0.22,
    ) -> None:
        try:
            import pyautogui
        except ImportError as error:
            raise RuntimeError(
                "PyAutoGUI is required for cursor movement. Run "
                "'python -m pip install -r requirements.txt'."
            ) from error

        self._pyautogui = pyautogui
        self._pyautogui.PAUSE = 0
        self._pyautogui.FAILSAFE = False
        size = self._pyautogui.size()
        self._mapper = CursorMapper((size.width, size.height), active_area, sensitivity)
        self._smoother = CursorSmoother(smoothing)
        current = self._pyautogui.position()
        self._smoother.reset((float(current.x), float(current.y)))
        self._left_button_down = False

    def move_to_landmark(self, normalized_point: tuple[float, float]) -> tuple[int, int]:
        """Map, filter, and move to a normalized index fingertip coordinate."""
        target = self._mapper.map_point(normalized_point)
        filtered = self._smoother.update(target)
        position = (round(filtered[0]), round(filtered[1]))
        self._pyautogui.moveTo(*position, _pause=False)
        return position

    def apply_actions(self, actions: tuple[MouseAction, ...]) -> None:
        """Perform state transitions emitted by ``GestureRecognizer``."""
        for action in actions:
            if action.type is MouseActionType.LEFT_CLICK:
                self._pyautogui.click(button="left", _pause=False)
            elif action.type is MouseActionType.RIGHT_CLICK:
                self._pyautogui.click(button="right", _pause=False)
            elif action.type is MouseActionType.LEFT_DOWN and not self._left_button_down:
                self._pyautogui.mouseDown(button="left", _pause=False)
                self._left_button_down = True
            elif action.type is MouseActionType.LEFT_UP and self._left_button_down:
                self._pyautogui.mouseUp(button="left", _pause=False)
                self._left_button_down = False
            elif action.type is MouseActionType.SCROLL and action.amount:
                self._pyautogui.scroll(action.amount, _pause=False)

    def release_all(self) -> None:
        """Release held buttons during loss of tracking or application shutdown."""
        if self._left_button_down:
            # The top-left failsafe may have interrupted a drag. Temporarily
            # disabling it is safe here because this method only releases an
            # already-held button; it never moves or presses the mouse.
            failsafe_was_enabled = self._pyautogui.FAILSAFE
            try:
                self._pyautogui.FAILSAFE = False
                self._pyautogui.mouseUp(button="left", _pause=False)
                self._left_button_down = False
            finally:
                self._pyautogui.FAILSAFE = failsafe_was_enabled

