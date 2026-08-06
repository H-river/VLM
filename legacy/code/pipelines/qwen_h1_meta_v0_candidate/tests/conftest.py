from __future__ import annotations

from typing import Any

import pytest


ACTION_FIELDS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
METRIC_FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
)


def zero_action() -> dict[str, float]:
    return {field: 0.0 for field in ACTION_FIELDS}


def metric(value: float = 0.0) -> dict[str, float]:
    return {field: float(value) for field in METRIC_FIELDS}


@pytest.fixture
def valid_output() -> dict[str, Any]:
    return {
        "schema_version": "qwen_h1_meta_v0",
        "decision": "run_guided_h1",
        "observation_request": "reuse_current",
        "objective_profile": "balanced",
        "mask_profile": "all_actuators",
        "directional_prior": {
            "lens_x_delta_mm": "increase",
            "lens_y_delta_mm": "decrease",
            "camera_x_delta_mm": "hold",
            "camera_y_delta_mm": "unknown",
        },
        "step_scale": "medium",
        "risk_mode": "standard",
        "confidence": "high",
        "reason_codes": ["centroid_error_dominant"],
    }


@pytest.fixture
def valid_input() -> dict[str, Any]:
    history = []
    for _ in range(3):
        history.append(
            {
                "valid": True,
                "executed_action_mm": zero_action(),
                "measured_beam_delta": metric(),
                "predicted_beam_delta": metric(),
                "prediction_residual": metric(),
                "ensemble_uncertainty": metric(0.1),
                "padding_reason": "none",
            }
        )
    semantics = []
    for action, position, bound in (
        ("lens_x_delta_mm", "lens_x_mm", 0.05),
        ("lens_y_delta_mm", "lens_y_mm", 0.05),
        ("camera_x_delta_mm", "camera_x_mm", 0.02),
        ("camera_y_delta_mm", "camera_y_mm", 0.02),
    ):
        semantics.append(
            {
                "action_id": action,
                "position_id": position,
                "unit": "mm",
                "positive_command_semantics": f"increase canonical {position}",
                "negative_command_semantics": f"decrease canonical {position}",
                "legal_per_step_bounds_mm": [-bound, bound],
                "absolute_limit_source": "repository_sampling_domain_not_hardware_limit",
            }
        )
    return {
        "schema_version": "qwen_h1_meta_input_v0",
        "current_beam_image": {
            "role": "current_sensor_frame_beam_image",
            "coordinate_frame": "camera_sensor_array",
            "display_normalization": "per_image_peak_normalized_for_qwen_only",
            "width_px": 1024,
            "height_px": 1024,
        },
        "current_beam_state": {
            "centroid_x_px": 511.0,
            "centroid_y_px": 512.0,
            "sigma_x_px": 11.0,
            "sigma_y_px": 12.0,
            "peak_intensity": 100.0,
        },
        "target_beam_state": {
            "centroid_x_px": 512.0,
            "centroid_y_px": 512.0,
            "sigma_x_px": 10.0,
            "sigma_y_px": 10.0,
            "peak_intensity": 105.0,
        },
        "normalized_signed_error": metric(0.0),
        "actuator_positions_mm": {
            "lens_x_mm": 0.0,
            "lens_y_mm": 0.0,
            "camera_x_mm": 0.0,
            "camera_y_mm": 0.0,
        },
        "actuator_semantics": semantics,
        "history": history,
        "forward_uncertainty": {
            "per_metric": metric(0.1),
            "mean": 0.1,
            "maximum": 0.1,
        },
        "remaining_budget": {"measurement_steps": 2, "control_steps": 4},
        "measurement_validity": {
            "state": "valid",
            "supervisor_diagnosis": "nominal",
            "measurement_policy": "standard",
        },
    }
