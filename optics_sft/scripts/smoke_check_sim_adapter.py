#!/usr/bin/env python3
"""Smoke check for the optics_sft to optical_sim adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_sft.physics.sim_adapter import (
    Action,
    apply_action_to_setup,
    residual_error_px,
    setup_to_safe_metadata,
    simulate_and_measure,
)


def compact_state(state: dict[str, float]) -> dict[str, float]:
    keys = (
        "centroid_x_m",
        "centroid_y_m",
        "centroid_x_px",
        "centroid_y_px",
        "sigma_x_px",
        "sigma_y_px",
        "peak_intensity",
    )
    return {key: round(float(state[key]), 6) for key in keys}


def main() -> None:
    config = load_yaml("optical_sim/configs/base_config.yaml")
    setup = setup_from_dict(config)

    before = simulate_and_measure(setup)
    action = Action(
        lens_x_delta_mm=0.02,
        lens_y_delta_mm=-0.01,
        camera_x_delta_mm=0.005,
        camera_y_delta_mm=0.0,
    )
    after_setup = apply_action_to_setup(setup, action)
    after = simulate_and_measure(after_setup)

    safe_metadata = setup_to_safe_metadata(setup)
    assert "wavelength_nm" in safe_metadata
    assert "centroid_x_px" not in safe_metadata
    assert "control_plan" not in safe_metadata

    centroid_delta_px = residual_error_px(after["state"], before["state"])
    print("before_state:", json.dumps(compact_state(before["state"]), sort_keys=True))
    print("after_state:", json.dumps(compact_state(after["state"]), sort_keys=True))
    print(f"centroid_delta_px: {centroid_delta_px:.6f}")
    print("sim_adapter smoke check: OK")


if __name__ == "__main__":
    main()
