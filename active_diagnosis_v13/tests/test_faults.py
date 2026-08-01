from __future__ import annotations

import numpy as np

from active_diagnosis_v13.faults import (
    command_for_desired_physical_delta,
    effective_planning_bounds,
    realize_hidden_gain_step,
)
from continuous_control_v12.contracts import Bounds


def _bounds() -> Bounds:
    return Bounds(
        action_low=np.asarray([-0.05, -0.05, -0.02, -0.02]),
        action_high=np.asarray([0.05, 0.05, 0.02, 0.02]),
        position_low=np.full(4, -3.0),
        position_high=np.full(4, 3.0),
    )


def test_hidden_gain_separates_commanded_belief_and_true_position() -> None:
    step = realize_hidden_gain_step(
        true_position=np.zeros(4),
        commanded_position_belief=np.zeros(4),
        requested_command=np.asarray([0.01, -0.01, 0.002, -0.002]),
        true_gain=1.5,
        bounds=_bounds(),
    )
    assert np.allclose(
        step.next_commanded_position_belief,
        [0.01, -0.01, 0.002, -0.002],
    )
    assert np.allclose(
        step.next_true_position, [0.015, -0.015, 0.003, -0.003]
    )
    assert not step.step_saturated


def test_high_gain_saturates_at_original_physical_step_bound() -> None:
    step = realize_hidden_gain_step(
        true_position=np.zeros(4),
        commanded_position_belief=np.zeros(4),
        requested_command=np.asarray([0.05, -0.05, 0.02, -0.02]),
        true_gain=1.5,
        bounds=_bounds(),
    )
    assert np.allclose(step.realized_delta, [0.05, -0.05, 0.02, -0.02])
    assert step.step_saturated


def test_known_gain_command_recovers_requested_physical_delta() -> None:
    bounds = _bounds()
    desired = np.asarray([0.02, -0.01, 0.006, -0.004])
    for gain in (0.5, 0.75, 1.25, 1.5):
        reachable = effective_planning_bounds(bounds, gain)
        clipped_desired = np.clip(
            desired, reachable.action_low, reachable.action_high
        )
        command = command_for_desired_physical_delta(
            clipped_desired, gain, np.zeros(4), bounds
        )
        step = realize_hidden_gain_step(
            true_position=np.zeros(4),
            commanded_position_belief=np.zeros(4),
            requested_command=command,
            true_gain=gain,
            bounds=bounds,
        )
        assert np.allclose(step.realized_delta, clipped_desired)


def test_invalid_gain_is_rejected() -> None:
    for gain in (0.0, -1.0, float("nan")):
        try:
            effective_planning_bounds(_bounds(), gain)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid gain was accepted")
