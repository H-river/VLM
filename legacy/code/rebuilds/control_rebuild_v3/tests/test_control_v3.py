from __future__ import annotations

import numpy as np
import pytest

from control_rebuild_v3.common import (
    ACTION_GRID,
    ACTION_NORMALIZED,
    action_basis,
    inverse_candidate_features,
    residual_cost,
    select_minimum_cost,
)
from control_rebuild_v3.calibrate_forward import select_blend_candidate
from control_rebuild_v3.models import joint_forward_model, require_torch
from control_rebuild_v3.visual_inverse import (
    legacy_candidates_to_sensor,
    sensor_to_base_legacy,
)


def test_action_basis_is_zero_only_for_zero_action() -> None:
    basis = action_basis()
    assert basis.shape == (81, 37)
    zero = np.flatnonzero(np.all(ACTION_NORMALIZED == 0.0, axis=1))
    assert zero.tolist() == [40]
    assert np.all(basis[40] == 0.0)
    assert np.all(np.any(basis[np.arange(81) != 40] != 0.0, axis=1))


def test_joint_forward_zero_action_is_exactly_zero() -> None:
    torch = require_torch()
    basis = torch.from_numpy(action_basis())
    model = joint_forward_model(torch, context_dim=28, basis_dim=basis.shape[1])
    predicted = model(torch.randn(3, 28), basis)
    assert predicted.shape == (3, 81, 5)
    assert torch.equal(predicted[:, 40], torch.zeros_like(predicted[:, 40]))


def test_residual_cost_uses_declared_inverse_tolerances() -> None:
    desired = np.asarray([500.0, 500.0, 100.0, 100.0, 10.0])
    states = np.stack(
        [
            desired,
            desired + np.asarray([0.5, 0.0, 1.0, 1.0, 0.2]),
        ]
    )
    costs = residual_cost(states, desired)
    assert costs[0] == 0.0
    assert costs[1] == pytest.approx(np.sqrt(4.0 / 5.0), abs=1e-6)


def test_minimum_cost_tie_prefers_less_movement_then_index() -> None:
    costs = np.ones(len(ACTION_GRID), dtype=np.float32)
    selected = int(select_minimum_cost(costs))
    assert selected == 40


def test_blend_selection_enforces_forward_safety_constraints() -> None:
    def row(alpha: float, strict: float, mae: float, inverse: float) -> dict:
        return {
            "alpha": alpha,
            "forward": {
                "strict_all_five_success": strict,
                "mae_in_tolerance_units": mae,
            },
            "inverse_selection": {
                "target_success_feasible": inverse,
            },
        }

    baseline = row(0.0, 0.36, 0.64, 0.24)
    candidates = [
        baseline,
        row(0.2, 0.37, 0.63, 0.27),
        row(0.8, 0.33, 0.68, 0.31),
    ]
    selected = select_blend_candidate(candidates, baseline)
    assert selected["alpha"] == 0.2


def test_inverse_features_identify_exact_predicted_match() -> None:
    desired = np.asarray([[500.0, 500.0, 100.0, 100.0, 10.0]])
    states = np.broadcast_to(desired[:, None, :], (1, 81, 5)).copy()
    states[:, :, 0] += 5.0
    states[:, 40, :] = desired
    candidate, cost, status = inverse_candidate_features(states, desired)
    assert candidate.shape == (1, 81, 23)
    assert cost.shape == (1, 81)
    assert status.shape == (1, 12)
    assert cost[0, 40] == 0.0
    assert status[0, 5] == pytest.approx(1.0 / 81.0)


def test_visual_frame_adapter_is_action_specific() -> None:
    setup = {
        "pixel_size_um": 5.0,
        "camera_x_offset_mm": 0.025,
        "camera_y_offset_mm": -0.010,
    }
    sensor = np.asarray([[500.0, 510.0, 30.0, 31.0, 7.0]])
    legacy = sensor_to_base_legacy(sensor, [setup])
    assert legacy[0, 0] == pytest.approx(505.0)
    assert legacy[0, 1] == pytest.approx(508.0)

    candidates = np.broadcast_to(legacy[:, None, :], (1, 81, 5)).copy()
    converted = legacy_candidates_to_sensor(candidates, [setup])
    assert np.allclose(converted[0, 40], sensor[0])
    # Index 41 changes only camera_y by +0.02 mm = +4 sensor pixels.
    assert converted[0, 41, 1] == pytest.approx(sensor[0, 1] - 4.0)
