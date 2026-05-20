#!/usr/bin/env python3
"""Smoke checks for the physics SFT prompt metadata policy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.physics.metadata_policy import (
    assert_no_prompt_leakage,
    filter_safe_prompt_metadata,
    find_leakage_fields,
)


def main() -> None:
    metadata = {
        "wavelength_nm": 632.8,
        "beam_waist_mm": 1.0,
        "power_w": 1.0,
        "lens_focal_length_mm": 100.0,
        "grid_extent_mm": 30.0,
        "centroid_error_px": {"x": 12.0, "y": -4.0},
    }
    safe = filter_safe_prompt_metadata(metadata)
    assert "wavelength_nm" in safe
    assert "grid_extent_mm" in safe
    assert "centroid_error_px" not in safe

    safe_prompt_inputs = {
        "images": {
            "current_image_path": "images/current.png",
            "target_image_path": "images/target.png",
        },
        "safe_setup_metadata": safe,
    }
    assert_no_prompt_leakage(safe_prompt_inputs)

    leaking_prompt_inputs = {
        "images": {
            "current_image_path": "images/current.png",
        },
        "safe_setup_metadata": {
            "wavelength_nm": 632.8,
            "target_state": {"centroid_px": {"x": 512.0, "y": 512.0}},
        },
        "candidate_actions": [
            {
                "true_control_plan": {
                    "lens_x_delta_mm": -0.02,
                    "lens_y_delta_mm": 0.01,
                    "camera_x_delta_mm": 0.0,
                    "camera_y_delta_mm": 0.0,
                }
            }
        ],
    }
    leakage_fields = find_leakage_fields(leaking_prompt_inputs)
    assert "safe_setup_metadata.target_state" in leakage_fields
    assert "safe_setup_metadata.target_state.centroid_px" in leakage_fields
    assert "candidate_actions[0].true_control_plan" in leakage_fields

    try:
        assert_no_prompt_leakage(leaking_prompt_inputs)
    except ValueError as exc:
        message = str(exc)
        assert "target_state" in message
        assert "true_control_plan" in message
    else:
        raise AssertionError("Expected leaking prompt inputs to fail")

    print("metadata_policy smoke check: OK")


if __name__ == "__main__":
    main()
