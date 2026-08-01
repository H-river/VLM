"""Deterministic simulator tool for quantitative forward prediction."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from optics_sft.physics.sim_adapter import simulate_and_measure

from .build_dataset import rounded_state, state_change
from .core import apply_action_dict, setup_from_dict
from .tool_path_adapter import resolve_path


FORWARD_TOOL = "simulate_optical_transition"
FORWARD_SOURCE_MAP = {"state_handle_path": "setup_state_handle", "action_path": "action"}
FORWARD_TOOL_SPEC = {
    "description": (
        "Replay the visible optical setup and action with the deterministic simulator, then return "
        "the sensor after-state and change."
    ),
    "required_source_roles": ["state_handle_path", "action_path"],
}


def build_forward_arguments(
    visible_evidence: Mapping[str, Any], source_map: Mapping[str, Any]
) -> dict[str, Any]:
    if set(source_map) != set(FORWARD_SOURCE_MAP) or any(
        not isinstance(value, str) for value in source_map.values()
    ):
        raise ValueError(
            f"source_map for {FORWARD_TOOL} must contain exactly {sorted(FORWARD_SOURCE_MAP)}"
        )
    state_handle = resolve_path(visible_evidence, str(source_map["state_handle_path"]))
    action = resolve_path(visible_evidence, str(source_map["action_path"]))
    if not isinstance(state_handle, str) or not state_handle or not isinstance(action, Mapping):
        raise ValueError("forward state handle must be a nonempty string and action must be an object")
    return {"state_handle": state_handle, "action": copy.deepcopy(dict(action))}


def simulate_optical_transition(
    setup_config: Mapping[str, Any], action: Mapping[str, Any]
) -> dict[str, Any]:
    optical_setup = setup_from_dict(copy.deepcopy(dict(setup_config)))
    before = simulate_and_measure(optical_setup)["state"]
    after = simulate_and_measure(apply_action_dict(optical_setup, action))["state"]
    return {
        "after_state": rounded_state(after),
        "change": state_change(before, after),
    }


def run_mapped_forward_tool(
    visible_evidence: Mapping[str, Any],
    source_map: Mapping[str, Any],
    state_registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    arguments = build_forward_arguments(visible_evidence, source_map)
    state_handle = arguments["state_handle"]
    if state_handle not in state_registry:
        raise ValueError(f"unknown simulator state handle: {state_handle}")
    return simulate_optical_transition(state_registry[state_handle], arguments["action"])


def materialize_forward_answer(tool_result: Mapping[str, Any]) -> dict[str, Any]:
    if set(tool_result) != {"after_state", "change"}:
        raise ValueError("forward tool result has an unexpected schema")
    return {
        "status": "answerable",
        "answer": {
            "after_state": copy.deepcopy(tool_result["after_state"]),
            "change": copy.deepcopy(tool_result["change"]),
        },
    }
