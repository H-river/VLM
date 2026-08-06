"""Deterministic tools for evidence-grounded optics decisions.

The tools in this module deliberately operate only on prompt-visible numeric
evidence.  They do not import the optical simulator, inspect private dataset
fields, or produce the final experiment answer.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


CONTROL_TOOL = "select_minimum_motion_action"
SUFFICIENCY_TOOL = "threshold_completion_directions"
TOOL_NAMES = (CONTROL_TOOL, SUFFICIENCY_TOOL)
DIRECTION_ORDER = ("decrease", "no_change", "increase")

TOOL_CATALOG: dict[str, dict[str, Any]] = {
    CONTROL_TOOL: {
        "description": (
            "Threshold every candidate residual, enumerate successful rows, and "
            "select the successful row with minimum absolute actuator motion."
        ),
        "required_arguments": [
            "candidate_residuals_px",
            "active_actuator_motions_mm",
            "success_tolerance_px",
            "allowed_order_indices",
        ],
    },
    SUFFICIENCY_TOOL: {
        "description": (
            "Threshold every compatible-completion delta and return the set of "
            "observed directions and the first conflicting pair."
        ),
        "required_arguments": ["measured_deltas_px", "direction_threshold_px"],
    },
}


def _finite_numbers(values: Sequence[Any], name: str) -> list[float]:
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"{name}[{index}] must be a finite number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be a finite number")
        result.append(number)
    return result


def _nonnegative_number(value: Any, name: str) -> float:
    values = _finite_numbers([value], name)
    if values[0] < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return values[0]


def threshold_completion_directions(
    measured_deltas_px: Sequence[Any], direction_threshold_px: Any
) -> dict[str, Any]:
    """Threshold completion deltas and expose their discrete evidence object."""

    deltas = _finite_numbers(measured_deltas_px, "measured_deltas_px")
    if not deltas:
        raise ValueError("measured_deltas_px must not be empty")
    threshold = _nonnegative_number(direction_threshold_px, "direction_threshold_px")
    directions = [
        "increase"
        if delta > threshold
        else "decrease"
        if delta < -threshold
        else "no_change"
        for delta in deltas
    ]
    observed = [direction for direction in DIRECTION_ORDER if direction in directions]
    conflict = None
    for left in range(len(directions)):
        for right in range(left + 1, len(directions)):
            if directions[left] != directions[right]:
                conflict = [left, right]
                break
        if conflict is not None:
            break
    return {
        "per_trial_directions": directions,
        "observed_direction_set": observed,
        "conflicting_pair_indices": conflict,
        "all_directions_agree": len(observed) == 1,
    }


def select_minimum_motion_action(
    candidate_residuals_px: Sequence[Any],
    active_actuator_motions_mm: Sequence[Any],
    success_tolerance_px: Any,
    allowed_order_indices: Sequence[Any],
) -> dict[str, Any]:
    """Exhaustively threshold and select a row using the declared tie-breaks."""

    residuals = _finite_numbers(candidate_residuals_px, "candidate_residuals_px")
    motions = _finite_numbers(active_actuator_motions_mm, "active_actuator_motions_mm")
    tolerance = _nonnegative_number(success_tolerance_px, "success_tolerance_px")
    if not residuals:
        raise ValueError("candidate_residuals_px must not be empty")
    if len(motions) != len(residuals) or len(allowed_order_indices) != len(residuals):
        raise ValueError("control argument arrays must have identical nonzero lengths")
    order: list[int] = []
    for index, value in enumerate(allowed_order_indices):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"allowed_order_indices[{index}] must be an integer")
        order.append(value)
    if sorted(order) != list(range(len(order))):
        raise ValueError("allowed_order_indices must be a permutation of 0..N-1")

    successful = [index for index, residual in enumerate(residuals) if residual <= tolerance]
    selected = (
        min(successful, key=lambda index: (abs(motions[index]), residuals[index], order[index]))
        if successful
        else None
    )
    best = min(
        range(len(residuals)),
        key=lambda index: (residuals[index], abs(motions[index]), order[index]),
    )
    return {
        "successful_action_indices": successful,
        "selected_index": selected,
        "best_residual_index": best,
        "feasible_within_tolerance": bool(successful),
    }


def run_tool(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and execute one registered decision tool."""

    if tool_name == CONTROL_TOOL:
        expected = set(TOOL_CATALOG[CONTROL_TOOL]["required_arguments"])
        if set(arguments) != expected:
            raise ValueError(f"{CONTROL_TOOL} arguments must be exactly {sorted(expected)}")
        return select_minimum_motion_action(**arguments)
    if tool_name == SUFFICIENCY_TOOL:
        expected = set(TOOL_CATALOG[SUFFICIENCY_TOOL]["required_arguments"])
        if set(arguments) != expected:
            raise ValueError(
                f"{SUFFICIENCY_TOOL} arguments must be exactly {sorted(expected)}"
            )
        return threshold_completion_directions(**arguments)
    raise ValueError(f"unknown tool: {tool_name}")


def tool_for_task(task_type: str) -> str:
    if task_type == "constrained_intervention":
        return CONTROL_TOOL
    if task_type == "information_sufficiency":
        return SUFFICIENCY_TOOL
    raise ValueError(f"no decision tool registered for task: {task_type}")
