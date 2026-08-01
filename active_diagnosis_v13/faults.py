"""Hidden command-gain dynamics with explicit belief/physical-state separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from continuous_control_v12.contracts import (
    Bounds,
    action_dict,
    action_vector,
    position_dict,
    position_vector,
    project_action,
)


@dataclass(frozen=True)
class GainStep:
    """One hidden-gain transition; true fields are evaluator-only."""

    requested_command: np.ndarray
    accepted_command: np.ndarray
    unconstrained_realized_delta: np.ndarray
    realized_delta: np.ndarray
    next_commanded_position_belief: np.ndarray
    next_true_position: np.ndarray
    step_saturated: bool
    absolute_position_saturated: bool

    def audit_dict(self) -> dict[str, Any]:
        return {
            "requested_command_mm": action_dict(self.requested_command),
            "accepted_command_mm": action_dict(self.accepted_command),
            "unconstrained_realized_delta_mm": action_dict(
                self.unconstrained_realized_delta
            ),
            "realized_delta_mm": action_dict(self.realized_delta),
            "next_commanded_position_belief_mm": position_dict(
                self.next_commanded_position_belief
            ),
            "evaluator_only_next_true_position_mm": position_dict(
                self.next_true_position
            ),
            "step_saturated": self.step_saturated,
            "absolute_position_saturated": self.absolute_position_saturated,
        }


def _validate_gain(gain: float) -> float:
    value = float(gain)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("gain must be positive and finite")
    return value


def realize_hidden_gain_step(
    *,
    true_position: Mapping[str, Any] | Sequence[float],
    commanded_position_belief: Mapping[str, Any] | Sequence[float],
    requested_command: Mapping[str, Any] | Sequence[float],
    true_gain: float,
    bounds: Bounds,
) -> GainStep:
    """Apply a command with hidden scalar gain and physical step saturation.

    The controller's pre-estimation position belief advances by the accepted
    command. The evaluator-only true position advances by gain times command,
    clipped to the original physical per-step and absolute-position bounds.
    """

    gain = _validate_gain(true_gain)
    true_position_v = position_vector(true_position)
    belief_v = position_vector(commanded_position_belief)
    requested_v = action_vector(requested_command)
    accepted = project_action(belief_v, requested_v, bounds)
    next_belief = belief_v + accepted
    unconstrained = gain * accepted
    step_limited = np.clip(
        unconstrained, bounds.action_low, bounds.action_high
    )
    realized = np.minimum(step_limited, bounds.position_high - true_position_v)
    realized = np.maximum(realized, bounds.position_low - true_position_v)
    next_true = true_position_v + realized
    return GainStep(
        requested_command=requested_v,
        accepted_command=accepted,
        unconstrained_realized_delta=unconstrained,
        realized_delta=realized.astype(np.float64),
        next_commanded_position_belief=next_belief.astype(np.float64),
        next_true_position=next_true.astype(np.float64),
        step_saturated=not np.allclose(
            unconstrained, step_limited, atol=1e-12, rtol=0.0
        ),
        absolute_position_saturated=not np.allclose(
            step_limited, realized, atol=1e-12, rtol=0.0
        ),
    )


def effective_planning_bounds(bounds: Bounds, gain_belief: float) -> Bounds:
    """Physical-delta bounds reachable through bounded commands at a gain."""

    gain = _validate_gain(gain_belief)
    scale = min(gain, 1.0)
    return Bounds(
        action_low=bounds.action_low * scale,
        action_high=bounds.action_high * scale,
        position_low=bounds.position_low.copy(),
        position_high=bounds.position_high.copy(),
    )


def command_for_desired_physical_delta(
    desired_delta: Mapping[str, Any] | Sequence[float],
    gain_belief: float,
    commanded_position_belief: Mapping[str, Any] | Sequence[float],
    bounds: Bounds,
) -> np.ndarray:
    """Convert a desired physical delta into a safe bounded command."""

    gain = _validate_gain(gain_belief)
    command = action_vector(desired_delta) / gain
    return project_action(commanded_position_belief, command, bounds)
