from __future__ import annotations

import unittest

from optics_understanding_sft.direction_inverse_v1.train_direction_small import (
    FIELDS,
    feature_vector,
    labels,
)


class TrainDirectionSmallTest(unittest.TestCase):
    def test_single_field_masks_other_heads(self) -> None:
        record = {
            "task_type": "direction_single_field",
            "target": {"answer": {"field": "width_x", "direction": "increase"}},
        }
        values = labels(record)
        self.assertEqual(values[FIELDS.index("width_x")], 2)
        self.assertEqual(sum(value != -100 for value in values), 1)

    def test_feature_vector_has_no_target_dependency(self) -> None:
        record = {
            "prompt_inputs": {
                "setup": {
                    "wavelength_nm": 632.8,
                    "beam_waist_mm": 1.0,
                    "power_w": 1.0,
                    "lens_focal_length_mm": 100.0,
                    "lens_aperture_mm": 25.0,
                    "source_to_lens_mm": 200.0,
                    "lens_to_camera_mm": 150.0,
                    "lens_x_offset_mm": 0.0,
                    "lens_y_offset_mm": 0.0,
                    "camera_x_offset_mm": 0.0,
                    "camera_y_offset_mm": 0.0,
                    "pixel_size_um": 5.5,
                },
                "current_beam_state": {
                    "centroid_x_px": 500.0,
                    "centroid_y_px": 500.0,
                    "sigma_x_px": 100.0,
                    "sigma_y_px": 100.0,
                    "peak_intensity": 9.0,
                },
                "action": {
                    "lens_x_delta_mm": 0.0,
                    "lens_y_delta_mm": 0.0,
                    "camera_x_delta_mm": 0.0,
                    "camera_y_delta_mm": 0.0,
                },
            }
        }
        self.assertEqual(len(feature_vector(record)), 21)


if __name__ == "__main__":
    unittest.main()
