from __future__ import annotations

import unittest

from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    action_direction,
    action_grid,
    matching_indices,
    select_minimum_motion,
    state_matches,
)


TOLERANCE = {
    "centroid_vector_px": 1.0,
    "width_each_px": 2.0,
    "peak_relative": 0.05,
}


def state(x: float, peak: float = 10.0) -> dict:
    return {
        "centroid_x_px": x,
        "centroid_y_px": 0.0,
        "sigma_x_px": 100.0,
        "sigma_y_px": 100.0,
        "peak_intensity": peak,
    }


class BuildInverseTest(unittest.TestCase):
    def test_grid_has_81_unique_actions(self) -> None:
        grid = action_grid([-0.05, 0.0, 0.05], [-0.02, 0.0, 0.02])
        self.assertEqual(len(grid), 81)
        self.assertEqual(len({tuple(item.values()) for item in grid}), 81)

    def test_matching_uses_all_tolerances(self) -> None:
        self.assertTrue(state_matches(state(0.9, 10.4), state(0.0, 10.0), TOLERANCE))
        self.assertFalse(state_matches(state(1.1, 10.4), state(0.0, 10.0), TOLERANCE))
        self.assertFalse(state_matches(state(0.9, 10.6), state(0.0, 10.0), TOLERANCE))

    def test_minimum_motion_selects_smallest_matching_action(self) -> None:
        actions = [
            {"lens_x_delta_mm": 0.05, "lens_y_delta_mm": 0.0, "camera_x_delta_mm": 0.0, "camera_y_delta_mm": 0.0},
            {"lens_x_delta_mm": 0.0, "lens_y_delta_mm": 0.0, "camera_x_delta_mm": 0.02, "camera_y_delta_mm": 0.0},
        ]
        states = [state(0.1), state(0.2)]
        matches = matching_indices(states, state(0.0), TOLERANCE)
        self.assertEqual(select_minimum_motion(actions, states, state(0.0), matches, TOLERANCE), 1)

    def test_action_direction(self) -> None:
        action = {
            "lens_x_delta_mm": -0.05,
            "lens_y_delta_mm": 0.0,
            "camera_x_delta_mm": 0.02,
            "camera_y_delta_mm": 0.0,
        }
        self.assertEqual(
            action_direction(action),
            {"lens_x": "decrease", "lens_y": "no_change", "camera_x": "increase", "camera_y": "no_change"},
        )


if __name__ == "__main__":
    unittest.main()
