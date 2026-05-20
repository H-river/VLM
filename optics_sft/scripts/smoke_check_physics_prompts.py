#!/usr/bin/env python3
"""Smoke-check physics SFT prompt templates and builder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.physics.prompt_builder import build_physics_prompt, expected_image_slots


SAFE_METADATA = {
    "wavelength_nm": 632.8,
    "beam_waist_mm": 1.0,
    "power_w": 1.0,
    "lens_focal_length_mm": 100.0,
    "lens_aperture_mm": 25.0,
    "source_to_lens_mm": 200.0,
    "lens_to_camera_mm": 150.0,
    "sensor_resolution": [1024, 1024],
    "pixel_size_um": 5.5,
    "coordinate_convention": "positive lens x shifts beam centroid in the calibrated positive sensor x direction",
    "propagation_backend": "fresnel_numpy",
    "grid_size": 1024,
    "grid_extent_mm": 30.0,
}


def fake_rows() -> list[dict]:
    return [
        {
            "sample_id": "smoke_inverse",
            "sample_type": "inverse_control",
            "prompt_inputs": {
                "images": {
                    "current_image_path": "train/smoke_inverse_current.png",
                    "target_image_path": "train/smoke_inverse_target.png",
                },
                "safe_setup_metadata": dict(SAFE_METADATA),
            },
            "target": {},
            "private_eval": {},
            "split_tags": ["smoke"],
        },
        {
            "sample_id": "smoke_forward",
            "sample_type": "forward_transition",
            "prompt_inputs": {
                "images": {
                    "before_image_path": "train/smoke_forward_before.png",
                },
                "safe_setup_metadata": dict(SAFE_METADATA),
                "action": {
                    "lens_x_delta_mm": 0.02,
                    "lens_y_delta_mm": -0.01,
                    "camera_x_delta_mm": 0.0,
                    "camera_y_delta_mm": 0.0,
                },
            },
            "target": {},
            "private_eval": {},
            "split_tags": ["smoke"],
        },
        {
            "sample_id": "smoke_counterfactual",
            "sample_type": "counterfactual_pair",
            "prompt_inputs": {
                "images": {
                    "scenario_a_current_image_path": "train/smoke_cf_a_current.png",
                    "scenario_a_target_image_path": "train/smoke_cf_a_target.png",
                    "scenario_b_current_image_path": "train/smoke_cf_b_current.png",
                    "scenario_b_target_image_path": "train/smoke_cf_b_target.png",
                },
                "safe_setup_metadata": {
                    "scenario_a": dict(SAFE_METADATA),
                    "scenario_b": {
                        **SAFE_METADATA,
                        "lens_focal_length_mm": 130.0,
                    },
                },
            },
            "target": {},
            "private_eval": {},
            "split_tags": ["smoke"],
        },
        {
            "sample_id": "smoke_trajectory",
            "sample_type": "trajectory",
            "prompt_inputs": {
                "images": {
                    "initial_image_path": "train/smoke_traj_initial.png",
                    "target_image_path": "train/smoke_traj_target.png",
                    "step_00_image_path": "train/smoke_traj_step_00.png",
                },
                "safe_setup_metadata": dict(SAFE_METADATA),
            },
            "target": {},
            "private_eval": {},
            "split_tags": ["smoke"],
        },
    ]


def print_snippet(row: dict) -> None:
    prompt = build_physics_prompt(row)
    snippet = " ".join(prompt.split())[:360]
    slots = expected_image_slots(row, include_trajectory_step_images=True)
    print(f"--- {row['sample_type']} ---")
    print(f"slots={slots}")
    print(f"snippet={snippet}")


def verify_leakage_rejection() -> None:
    row = fake_rows()[0]
    row["prompt_inputs"]["safe_setup_metadata"]["target_state"] = {"centroid_x_px": 10.0}
    try:
        build_physics_prompt(row)
    except ValueError as exc:
        print(f"leakage_rejection=ok ({exc})")
        return
    raise AssertionError("Expected build_physics_prompt to reject leaking prompt_inputs")


def main() -> None:
    for row in fake_rows():
        print_snippet(row)
    verify_leakage_rejection()


if __name__ == "__main__":
    main()
