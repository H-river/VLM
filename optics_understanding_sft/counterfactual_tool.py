"""Deterministic paired-simulator tool for quantitative counterfactual reasoning."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from optics_sft.physics.sim_adapter import simulate_and_measure

from .build_dataset import state_change
from .core import apply_action_dict, setup_from_dict
from .tool_path_adapter import resolve_path


COUNTERFACTUAL_TOOL = "simulate_paired_counterfactual"
COUNTERFACTUAL_SOURCE_MAP = {
    "scenario_a_state_handle_path": "scenario_a_state_handle",
    "scenario_b_state_handle_path": "scenario_b_state_handle",
    "action_path": "shared_action",
    "changed_parameter_path": "changed_parameter",
}
COUNTERFACTUAL_TOOL_SPEC = {
    "description": (
        "Replay two registered optical states under the same action and return their exact "
        "responses, response difference, and centroid-direction preservation decision."
    ),
    "required_source_roles": list(COUNTERFACTUAL_SOURCE_MAP),
}


def build_counterfactual_arguments(
    visible_evidence: Mapping[str, Any], source_map: Mapping[str, Any]
) -> dict[str, Any]:
    if set(source_map) != set(COUNTERFACTUAL_SOURCE_MAP) or any(
        not isinstance(value, str) for value in source_map.values()
    ):
        raise ValueError(
            "counterfactual source_map must contain exactly "
            f"{sorted(COUNTERFACTUAL_SOURCE_MAP)}"
        )
    state_a = resolve_path(visible_evidence, str(source_map["scenario_a_state_handle_path"]))
    state_b = resolve_path(visible_evidence, str(source_map["scenario_b_state_handle_path"]))
    action = resolve_path(visible_evidence, str(source_map["action_path"]))
    changed_parameter = resolve_path(
        visible_evidence, str(source_map["changed_parameter_path"])
    )
    if not isinstance(state_a, str) or not state_a or not isinstance(state_b, str) or not state_b:
        raise ValueError("counterfactual state handles must be nonempty strings")
    if not isinstance(action, Mapping) or not isinstance(changed_parameter, str):
        raise ValueError("counterfactual action must be an object and changed parameter a string")
    return {
        "scenario_a_state_handle": state_a,
        "scenario_b_state_handle": state_b,
        "shared_action": copy.deepcopy(dict(action)),
        "changed_parameter": changed_parameter,
    }


def _active_centroid_key(action: Mapping[str, Any]) -> str:
    x_motion = abs(float(action.get("lens_x_delta_mm", 0.0))) + abs(
        float(action.get("camera_x_delta_mm", 0.0))
    )
    y_motion = abs(float(action.get("lens_y_delta_mm", 0.0))) + abs(
        float(action.get("camera_y_delta_mm", 0.0))
    )
    if x_motion == y_motion:
        raise ValueError("paired counterfactual action must select exactly one centroid axis")
    return "centroid_x_px" if x_motion > y_motion else "centroid_y_px"


def _direction(value: float) -> float:
    return math.copysign(1.0, value) if abs(value) > 1.0 else 0.0


def simulate_paired_counterfactual(
    setup_a: Mapping[str, Any],
    setup_b: Mapping[str, Any],
    action: Mapping[str, Any],
    changed_parameter: str,
) -> dict[str, Any]:
    optical_a = setup_from_dict(copy.deepcopy(dict(setup_a)))
    optical_b = setup_from_dict(copy.deepcopy(dict(setup_b)))
    a_before = simulate_and_measure(optical_a)["state"]
    a_after = simulate_and_measure(apply_action_dict(optical_a, action))["state"]
    b_before = simulate_and_measure(optical_b)["state"]
    b_after = simulate_and_measure(apply_action_dict(optical_b, action))["state"]
    response_a = state_change(a_before, a_after)
    response_b = state_change(b_before, b_after)
    axis = _active_centroid_key(action)
    return {
        "changed_parameter": changed_parameter,
        "response_a": response_a,
        "response_b": response_b,
        "response_difference": {
            key: round(float(response_b[key]) - float(response_a[key]), 4)
            for key in response_a
        },
        "centroid_direction_preserved": _direction(float(response_a[axis]))
        == _direction(float(response_b[axis])),
    }


def run_mapped_counterfactual_tool(
    visible_evidence: Mapping[str, Any],
    source_map: Mapping[str, Any],
    state_registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    arguments = build_counterfactual_arguments(visible_evidence, source_map)
    state_a = arguments["scenario_a_state_handle"]
    state_b = arguments["scenario_b_state_handle"]
    if state_a not in state_registry or state_b not in state_registry:
        raise ValueError("unknown paired counterfactual state handle")
    return simulate_paired_counterfactual(
        state_registry[state_a],
        state_registry[state_b],
        arguments["shared_action"],
        arguments["changed_parameter"],
    )


def materialize_counterfactual_answer(tool_result: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "changed_parameter",
        "response_a",
        "response_b",
        "response_difference",
        "centroid_direction_preserved",
    }
    if set(tool_result) != expected:
        raise ValueError("counterfactual tool result has an unexpected schema")
    return {"status": "answerable", "answer": copy.deepcopy(dict(tool_result))}
