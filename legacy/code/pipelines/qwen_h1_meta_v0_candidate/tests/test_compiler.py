from __future__ import annotations

import copy

import numpy as np
import pytest

from continuous_control_v12.contracts import Bounds
from qwen_h1_meta_v0_candidate.compiler import (
    H1OnlyViolation,
    assert_h1_only_config,
    compile_guidance,
    default_h1_config,
)
from qwen_h1_meta_v0_candidate.contracts import parse_meta_output


def bounds() -> Bounds:
    return Bounds(
        action_low=np.asarray([-0.05, -0.05, -0.02, -0.02]),
        action_high=np.asarray([0.05, 0.05, 0.02, 0.02]),
        position_low=np.full(4, -3.0),
        position_high=np.full(4, 3.0),
    )


def test_compiler_fixed_mapping_and_nonexpansion(valid_output) -> None:
    output = copy.deepcopy(valid_output)
    output["objective_profile"] = "centroid_priority"
    output["mask_profile"] = "lens_only"
    output["step_scale"] = "fine"
    output["risk_mode"] = "conservative"
    compiled = compile_guidance(parse_meta_output(output), default_bounds=bounds())

    assert compiled.is_guided
    assert compiled.objective_multipliers == (1.25, 1.25, 0.85, 0.85, 0.85)
    assert compiled.allowed_dofs == ("lens_x", "lens_y")
    assert compiled.action_bound_scale == pytest.approx(0.35 * 0.8)
    assert compiled.uncertainty_weight_multiplier == 2.0
    guided_bounds = compiled.bounds_for(bounds())
    assert np.all(guided_bounds.action_high <= bounds().action_high)
    assert compiled.initial_mean_mm == pytest.approx(
        (0.5 * 0.05 * 0.35 * 0.8, -0.5 * 0.05 * 0.35 * 0.8, 0.0, 0.0)
    )


def test_balanced_view_is_bitwise_identity_and_priority_is_h1_view(valid_output) -> None:
    predicted = np.asarray([1.1, -2.2, 3.3, -4.4, 5.5], dtype=np.float64)
    target = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    balanced = compile_guidance(valid_output, default_bounds=bounds())
    viewed = balanced.objective_view(predicted, target)
    assert viewed.tobytes() == predicted.tobytes()

    priority_output = copy.deepcopy(valid_output)
    priority_output["objective_profile"] = "intensity_priority"
    priority = compile_guidance(priority_output, default_bounds=bounds())
    assert np.array_equal(
        priority.objective_view(predicted, target),
        target + np.asarray([0.9, 0.9, 0.9, 0.9, 1.25]) * (predicted - target),
    )


def test_default_and_low_confidence_are_canonicalized(valid_output) -> None:
    default_output = copy.deepcopy(valid_output)
    default_output["decision"] = "run_default_h1"
    default_output["objective_profile"] = "width_priority"
    default_output["mask_profile"] = "camera_only"
    default_output["step_scale"] = "fine"
    default_output["risk_mode"] = "conservative"
    compiled = compile_guidance(default_output, default_bounds=bounds())
    assert not compiled.is_guided
    assert compiled.objective_profile == "balanced"
    assert compiled.allowed_dofs == ("lens_x", "lens_y", "camera_x", "camera_y")
    assert compiled.action_bound_scale == 1.0
    assert compiled.initial_mean_mm == (0.0, 0.0, 0.0, 0.0)

    low = copy.deepcopy(valid_output)
    low["confidence"] = "low"
    compiled_low = compile_guidance(low, default_bounds=bounds())
    assert compiled_low.effective_decision == "run_default_h1"
    assert compiled_low.fallback_reason == "low_confidence_guidance"


def test_conservative_never_reduces_uncertainty_weight(valid_output) -> None:
    output = copy.deepcopy(valid_output)
    output["risk_mode"] = "conservative"
    compiled = compile_guidance(output, default_bounds=bounds())
    config = default_h1_config()
    compiled_config = compiled.planner_config(config)
    assert compiled_config["uncertainty_weight"] == pytest.approx(
        2.0 * config["uncertainty_weight"]
    )


@pytest.mark.parametrize("horizon", [0, 2, 3, 4, True, "1"])
def test_h3_and_every_non_integer_h1_config_are_impossible(horizon) -> None:
    config = default_h1_config()
    config["horizon"] = horizon
    with pytest.raises(H1OnlyViolation):
        assert_h1_only_config(config)
