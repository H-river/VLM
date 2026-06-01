"""Simulator-backed action search for physics-aware inverse-control labels."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from optical_sim.src.optical_elements import OpticalSetup
from optics_sft.physics.sim_adapter import (
    Action,
    apply_action_to_setup,
    residual_error_px,
    simulate_and_measure,
)


@dataclass(frozen=True)
class ActionBounds:
    """Actuator search ranges in millimeters."""

    lens_x_mm: tuple[float, float]
    lens_y_mm: tuple[float, float]
    camera_x_mm: tuple[float, float] = (0.0, 0.0)
    camera_y_mm: tuple[float, float] = (0.0, 0.0)


def clamp_action(action: Action, bounds: ActionBounds) -> Action:
    """Clamp an action to actuator bounds."""
    return Action(
        lens_x_delta_mm=float(np.clip(action.lens_x_delta_mm, *bounds.lens_x_mm)),
        lens_y_delta_mm=float(np.clip(action.lens_y_delta_mm, *bounds.lens_y_mm)),
        camera_x_delta_mm=float(np.clip(action.camera_x_delta_mm, *bounds.camera_x_mm)),
        camera_y_delta_mm=float(np.clip(action.camera_y_delta_mm, *bounds.camera_y_mm)),
    )


def score_action(
    setup: OpticalSetup,
    target_state: Mapping[str, Any],
    action: Action,
) -> dict[str, Any]:
    """Apply an action, simulate, and score centroid residual in pixels."""
    after_setup = apply_action_to_setup(setup, action)
    after = simulate_and_measure(after_setup)
    return {
        "action": action,
        "post_action_error_px": residual_error_px(after["state"], target_state),
        "after_state": after["state"],
    }


def _axis_values(bounds: tuple[float, float], grid_size: int) -> list[float]:
    if grid_size <= 1:
        return [(bounds[0] + bounds[1]) / 2.0]
    return [float(value) for value in np.linspace(bounds[0], bounds[1], grid_size)]


def _best_of_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        raise ValueError("No action scores were evaluated")
    return min(scores, key=lambda item: float(item["post_action_error_px"]))


def grid_search_action(
    setup: OpticalSetup,
    target_state: Mapping[str, Any],
    bounds: ActionBounds,
    grid_size: int = 5,
    enable_camera: bool = False,
) -> dict[str, Any]:
    """Search a small actuator grid and return the lowest-error action."""
    lens_x_values = _axis_values(bounds.lens_x_mm, grid_size)
    lens_y_values = _axis_values(bounds.lens_y_mm, grid_size)
    camera_x_values = _axis_values(bounds.camera_x_mm, grid_size) if enable_camera else [0.0]
    camera_y_values = _axis_values(bounds.camera_y_mm, grid_size) if enable_camera else [0.0]

    scores: list[dict[str, Any]] = []
    for lens_x, lens_y, camera_x, camera_y in itertools.product(
        lens_x_values,
        lens_y_values,
        camera_x_values,
        camera_y_values,
    ):
        action = clamp_action(
            Action(lens_x, lens_y, camera_x, camera_y),
            bounds,
        )
        scores.append(score_action(setup, target_state, action))
    best = _best_of_scores(scores)
    best["method"] = "grid_search"
    return best


def refine_action_local(
    setup: OpticalSetup,
    target_state: Mapping[str, Any],
    bounds: ActionBounds,
    best_action: Action,
    step_mm: float = 0.01,
    grid_size: int = 3,
    enable_camera: bool = False,
) -> dict[str, Any]:
    """Run a local grid around an existing action with a smaller step."""
    half = max(1, grid_size // 2)
    offsets = [step_mm * (index - half) for index in range(grid_size)]
    camera_offsets = offsets if enable_camera else [0.0]
    scores: list[dict[str, Any]] = []
    for dx, dy, dcx, dcy in itertools.product(offsets, offsets, camera_offsets, camera_offsets):
        action = clamp_action(
            Action(
                best_action.lens_x_delta_mm + dx,
                best_action.lens_y_delta_mm + dy,
                best_action.camera_x_delta_mm + dcx,
                best_action.camera_y_delta_mm + dcy,
            ),
            bounds,
        )
        scores.append(score_action(setup, target_state, action))
    best = _best_of_scores(scores)
    best["method"] = "local_refine"
    return best


def estimate_local_jacobian(
    setup: OpticalSetup,
    step_mm: float = 0.01,
    current_state: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Estimate centroid response to lens x/y action using finite differences."""
    if step_mm <= 0.0:
        raise ValueError("step_mm must be positive")
    base_state = current_state if current_state is not None else simulate_and_measure(setup)["state"]

    columns: list[list[float]] = []
    for axis in ("x", "y"):
        if axis == "x":
            plus = Action(step_mm, 0.0, 0.0, 0.0)
            minus = Action(-step_mm, 0.0, 0.0, 0.0)
        else:
            plus = Action(0.0, step_mm, 0.0, 0.0)
            minus = Action(0.0, -step_mm, 0.0, 0.0)
        plus_state = score_action(setup, base_state, plus)["after_state"]
        minus_state = score_action(setup, base_state, minus)["after_state"]
        columns.append(
            [
                (float(plus_state["centroid_x_px"]) - float(minus_state["centroid_x_px"])) / (2.0 * step_mm),
                (float(plus_state["centroid_y_px"]) - float(minus_state["centroid_y_px"])) / (2.0 * step_mm),
            ]
        )

    matrix = np.asarray(columns, dtype=np.float64).T
    if not np.all(np.isfinite(matrix)):
        return None
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return None
    if not math.isfinite(condition):
        return None
    return {
        "matrix": matrix,
        "condition": condition,
        "base_state": dict(base_state),
        "step_mm": step_mm,
    }


def _jacobian_action(
    jacobian: Mapping[str, Any],
    target_state: Mapping[str, Any],
    bounds: ActionBounds,
) -> Action | None:
    matrix = np.asarray(jacobian["matrix"], dtype=np.float64)
    base_state = jacobian["base_state"]
    desired = np.asarray(
        [
            float(target_state["centroid_x_px"]) - float(base_state["centroid_x_px"]),
            float(target_state["centroid_y_px"]) - float(base_state["centroid_y_px"]),
        ],
        dtype=np.float64,
    )
    try:
        lens_delta = np.linalg.solve(matrix, desired)
    except np.linalg.LinAlgError:
        try:
            lens_delta = np.linalg.lstsq(matrix, desired, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
    if not np.all(np.isfinite(lens_delta)):
        return None
    return clamp_action(
        Action(float(lens_delta[0]), float(lens_delta[1]), 0.0, 0.0),
        bounds,
    )


def choose_control_action(
    setup: OpticalSetup,
    target_state: Mapping[str, Any],
    bounds: ActionBounds,
    grid_size: int = 5,
    enable_camera: bool = False,
    jacobian_step_mm: float = 0.01,
    condition_limit: float = 1.0e4,
    current_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose the best simulator-scored action from Jacobian and grid candidates."""
    candidates: list[dict[str, Any]] = []
    no_action = score_action(setup, target_state, Action(0.0, 0.0, 0.0, 0.0))
    no_action["method"] = "no_action"
    candidates.append(no_action)

    jacobian = estimate_local_jacobian(
        setup,
        step_mm=jacobian_step_mm,
        current_state=current_state,
    )
    if jacobian is not None and float(jacobian["condition"]) <= condition_limit:
        action = _jacobian_action(jacobian, target_state, bounds)
        if action is not None:
            scored = score_action(setup, target_state, action)
            scored["method"] = "jacobian"
            scored["jacobian_condition"] = float(jacobian["condition"])
            candidates.append(scored)
            refined = refine_action_local(
                setup,
                target_state,
                bounds,
                action,
                step_mm=max(jacobian_step_mm / 2.0, 1e-6),
                grid_size=3,
                enable_camera=enable_camera,
            )
            refined["method"] = "jacobian_local_refine"
            refined["jacobian_condition"] = float(jacobian["condition"])
            candidates.append(refined)

    grid_best = grid_search_action(
        setup,
        target_state,
        bounds,
        grid_size=grid_size,
        enable_camera=enable_camera,
    )
    grid_best["method"] = "grid_search"
    candidates.append(grid_best)
    refined_grid = refine_action_local(
        setup,
        target_state,
        bounds,
        grid_best["action"],
        step_mm=max(
            (bounds.lens_x_mm[1] - bounds.lens_x_mm[0]) / max(grid_size - 1, 1) / 2.0,
            1e-6,
        ),
        grid_size=3,
        enable_camera=enable_camera,
    )
    refined_grid["method"] = "grid_search_local_refine"
    candidates.append(refined_grid)

    return _best_of_scores(candidates)
