import pytest

from optics_understanding_sft.direct_reasoning_tools_v11 import (
    CAUSAL_TOOL,
    normalize_setup_summary,
    classify_registered_transition,
    run_mapped_direct_tool,
)
from optics_understanding_sft.evaluate_direct_tool_orchestration_v11 import wilson_interval


def test_setup_summary_uses_dataset_rounding_contract():
    result = normalize_setup_summary(
        {
            "source_to_lens_mm": 165.19162,
            "lens_to_camera_mm": 173.240311,
            "lens_focal_length_mm": 118.356828,
        },
        {"adjustable_parameters": ["lens_x", "lens_y", "camera_x", "camera_y"]},
    )
    assert result["total_source_to_sensor_mm"] == 338.4319
    assert result["lens_focal_length_m"] == 0.118357


def test_causal_transition_applies_all_three_deadbands():
    before = {
        "centroid_x_px": 10.0,
        "centroid_y_px": 10.0,
        "sigma_x_px": 5.0,
        "sigma_y_px": 5.0,
        "peak_intensity": 100.0,
    }
    after = {
        "centroid_x_px": 11.01,
        "centroid_y_px": 9.5,
        "sigma_x_px": 7.01,
        "sigma_y_px": 3.0,
        "peak_intensity": 105.0,
    }
    result = classify_registered_transition(
        before,
        after,
        {"centroid_px": 1.0, "sigma_px": 2.0, "peak_relative": 0.05},
    )
    assert result["effects"] == {
        "centroid_x": "increase",
        "centroid_y": "no_change",
        "sigma_x": "increase",
        "sigma_y": "no_change",
        "peak_intensity": "no_change",
    }


def test_mapped_causal_tool_rejects_missing_threshold_role():
    with pytest.raises(ValueError, match="source_map"):
        run_mapped_direct_tool(
            CAUSAL_TOOL,
            {
                "before_state_handle": "before",
                "after_state_handle": "after",
                "thresholds": {},
            },
            {
                "before_state_handle_path": "before_state_handle",
                "after_state_handle_path": "after_state_handle",
            },
            setup_registry={},
            observation_registry={},
        )


def test_confirmation_sample_has_stronger_perfect_score_bound():
    development = wilson_interval(30, 30)
    confirmation = wilson_interval(74, 74)
    assert development["low"] < 0.90
    assert confirmation["low"] > 0.95
    assert confirmation["low"] > development["low"]
