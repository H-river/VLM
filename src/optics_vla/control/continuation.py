"""Config-driven form of the frozen visible continuation rule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from optics_vla.common.config import ControllerConfig


def should_continue(
    step: Mapping[str, Any],
    *,
    minimum_improvement: float,
    maximum_distance: float,
    success_threshold: float = 1.0,
) -> bool:
    distance = float(step["actual_target_cost"])
    improvement = float(step["before_target_cost"]) - distance
    return bool(
        distance > success_threshold
        and improvement >= minimum_improvement
        and distance <= maximum_distance
    )


def continuation_horizon(
    trace: Sequence[Mapping[str, Any]], config: ControllerConfig
) -> int:
    """Choose the real-observation horizon without replaying controller physics."""

    horizon = min(config.initial_control_steps, len(trace))
    maximum = min(config.maximum_horizon, len(trace))
    while horizon < maximum:
        if not should_continue(
            trace[horizon - 1],
            minimum_improvement=config.minimum_last_step_improvement,
            maximum_distance=config.maximum_final_distance,
        ):
            break
        horizon += 1
    return horizon

