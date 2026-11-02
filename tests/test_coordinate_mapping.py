"""Unit tests for dependency-free cursor coordinate calculations."""

import unittest

from utils.coordinate_mapping import ActiveArea, CursorMapper, CursorSmoother


class CursorMappingTests(unittest.TestCase):
    def test_active_area_edges_map_to_display_edges(self) -> None:
        mapper = CursorMapper((1920, 1080), ActiveArea())
        self.assertEqual(mapper.map_point((0.12, 0.12)), (0.0, 0.0))
        self.assertEqual(mapper.map_point((0.88, 0.88)), (1919.0, 1079.0))

    def test_points_outside_active_area_are_clamped(self) -> None:
        mapper = CursorMapper((100, 100), ActiveArea())
        self.assertEqual(mapper.map_point((-1.0, 2.0)), (0.0, 99.0))

    def test_smoother_blends_successive_targets(self) -> None:
        smoother = CursorSmoother(0.25)
        self.assertEqual(smoother.update((0.0, 0.0)), (0.0, 0.0))
        self.assertEqual(smoother.update((100.0, 100.0)), (25.0, 25.0))


if __name__ == "__main__":
    unittest.main()

