from __future__ import annotations

import unittest

from optics_understanding_sft.direction_inverse_v1.build_direction import (
    FIELD_MAP,
    all_record,
    prompt_text,
    single_record,
)


def source_row() -> dict:
    return {
        "example_id": "transition_1",
        "group_id": "group_1",
        "inputs": {
            "setup": {"wavelength_nm": 632.8},
            "current_beam_state": {
                "centroid_x_px": 500.0,
                "centroid_y_px": 501.0,
                "sigma_x_px": 100.0,
                "sigma_y_px": 101.0,
                "peak_intensity": 3.0,
            },
            "action": {"lens_x_delta_mm": 0.01},
        },
        "target": {
            "directions": {
                "centroid_x": "increase",
                "centroid_y": "decrease",
                "sigma_x": "no_change",
                "sigma_y": "increase",
                "peak_intensity": "decrease",
            }
        },
    }


class BuildDirectionTest(unittest.TestCase):
    def test_width_names_map_to_sigma_labels(self) -> None:
        record = single_record(source_row(), "train", "width_x")
        self.assertEqual(record["target"]["answer"]["direction"], "no_change")

    def test_all_record_contains_exact_five_public_names(self) -> None:
        record = all_record(source_row(), "val")
        self.assertEqual(set(record["target"]["answer"]["directions"]), set(FIELD_MAP))

    def test_prompt_has_no_after_state(self) -> None:
        row = source_row()
        text = prompt_text({**row["inputs"], "images": []}, field=None)
        self.assertNotIn("after_state", text)
        self.assertNotIn("setup_state_handle", text)


if __name__ == "__main__":
    unittest.main()
