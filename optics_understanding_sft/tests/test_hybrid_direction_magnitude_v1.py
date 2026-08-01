from __future__ import annotations

import unittest

import numpy as np

from optics_understanding_sft.hybrid_direction_magnitude_v1 import (
    DIRECTION_FIELDS,
    action_for,
    direction_prompt,
    directions,
    sign_matrix,
)


class HybridDirectionMagnitudeV1Test(unittest.TestCase):
    def test_action_sampling_is_deterministic(self) -> None:
        self.assertEqual(action_for("group", 7, 42), action_for("group", 7, 42))
        self.assertNotEqual(action_for("group", 7, 42), action_for("group", 7, 43))

    def test_directions_use_sensor_thresholds(self) -> None:
        before = {
            "centroid_x_px": 500.0,
            "centroid_y_px": 500.0,
            "sigma_x_px": 100.0,
            "sigma_y_px": 100.0,
            "peak_intensity": 10.0,
        }
        change = {
            "centroid_x_px": 1.0,
            "centroid_y_px": -1.01,
            "sigma_x_px": 2.0,
            "sigma_y_px": 2.01,
            "peak_intensity": -0.51,
        }
        self.assertEqual(
            directions(change, before),
            {
                "centroid_x": "no_change",
                "centroid_y": "decrease",
                "sigma_x": "no_change",
                "sigma_y": "increase",
                "peak_intensity": "decrease",
            },
        )

    def test_sign_matrix_respects_declared_field_order(self) -> None:
        row = {"inputs": {"current_beam_state": {"peak_intensity": 1.0}}}
        labels = [{field: value for field, value in zip(
            DIRECTION_FIELDS,
            ("decrease", "no_change", "increase", "decrease", "increase"),
        )}]
        np.testing.assert_array_equal(sign_matrix([row], labels), [[-1, 0, 1, -1, 1]])

    def test_direction_prompt_contains_no_answer_or_after_state(self) -> None:
        row = {
            "example_id": "x",
            "group_id": "g",
            "inputs": {
                "setup": {"wavelength_nm": 633.0},
                "current_beam_state": {"peak_intensity": 1.0},
                "action": {"lens_x_delta_mm": 0.01},
            },
            "target": {
                "after_state": {"centroid_x_px": 999.0},
                "directions": {field: "increase" for field in DIRECTION_FIELDS},
            },
        }
        prompt = direction_prompt(row)
        self.assertNotIn("999.0", prompt["prompt"])
        self.assertNotIn("setup_state_handle", prompt["prompt"])
        self.assertNotIn("after_state", prompt["prompt_inputs"])


if __name__ == "__main__":
    unittest.main()
