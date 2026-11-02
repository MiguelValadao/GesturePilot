"""Map normalized camera points into stable screen cursor coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ActiveArea:
    """Normalized portion of the camera feed used for cursor movement."""

    left: float = 0.12
    top: float = 0.12
    right: float = 0.88
    bottom: float = 0.88

    def __post_init__(self) -> None:
        if not (0.0 <= self.left < self.right <= 1.0):
            raise ValueError("Active area requires 0 <= left < right <= 1.")
        if not (0.0 <= self.top < self.bottom <= 1.0):
            raise ValueError("Active area requires 0 <= top < bottom <= 1.")


class CursorMapper:
    """Transforms a normalized index-tip coordinate to a screen coordinate."""

    def __init__(
        self,
        screen_size: tuple[int, int],
        active_area: ActiveArea,
        sensitivity: float = 1.0,
    ) -> None:
        if screen_size[0] < 1 or screen_size[1] < 1:
            raise ValueError("Screen dimensions must be positive.")
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be greater than zero.")
        self._width, self._height = screen_size
        self._area = active_area
        self._sensitivity = sensitivity

    def map_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """Return the target screen point, clamped to the active area."""
        x, y = point
        x = min(max(x, self._area.left), self._area.right)
        y = min(max(y, self._area.top), self._area.bottom)
        relative_x = (x - self._area.left) / (self._area.right - self._area.left)
        relative_y = (y - self._area.top) / (self._area.bottom - self._area.top)

        # Sensitivity expands or contracts movement around the display centre.
        relative_x = self._clamp((relative_x - 0.5) * self._sensitivity + 0.5)
        relative_y = self._clamp((relative_y - 0.5) * self._sensitivity + 0.5)
        return (relative_x * (self._width - 1), relative_y * (self._height - 1))

    @staticmethod
    def _clamp(value: float) -> float:
        return min(max(value, 0.0), 1.0)


class CursorSmoother:
    """Exponential moving average that suppresses landmark jitter."""

    def __init__(self, smoothing: float = 0.22) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("Smoothing must be in the interval (0, 1].")
        self._smoothing = smoothing
        self._position: Optional[tuple[float, float]] = None

    def reset(self, position: Optional[tuple[float, float]] = None) -> None:
        """Clear filtering state or seed it from the current cursor position."""
        self._position = position

    def update(self, target: tuple[float, float]) -> tuple[float, float]:
        """Blend a new target with prior state and return the filtered point."""
        if self._position is None:
            self._position = target
            return target
        old_x, old_y = self._position
        new_x, new_y = target
        alpha = self._smoothing
        self._position = (
            old_x + (new_x - old_x) * alpha,
            old_y + (new_y - old_y) * alpha,
        )
        return self._position

