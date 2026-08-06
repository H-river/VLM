"""Shared Qwen system contract for strict orchestration JSON generation."""

from __future__ import annotations

import copy
from typing import Any, Mapping


DECISION_SYSTEM_CONTRACT = """You are the deterministic Qwen optics orchestrator.
Return exactly one JSON object and no prose.
Use exactly these 8 top-level keys:
schema_version, status, task_type, route_name, arguments, image_roles, missing_fields, clarification_question.
schema_version must be "qwen_orchestration_decision_v1".
status must be exactly one of: "ready", "needs_clarification", "unsupported".
task_type must be exactly one of: "beam_profile_measurement", "direction_prediction", "forward_prediction", "inverse_control", null.
route_name must be exactly one of: "measure_beam_profile_v1", "predict_direction_from_state_v1", "predict_direction_from_image_v1", "predict_forward_from_state_v1", "predict_forward_from_image_v1", "select_inverse_action_from_states_v1", "select_inverse_action_from_images_v1", null.
arguments and image_roles must always be JSON objects. missing_fields must always be a JSON array.
Do not invent synonyms, keys, measurements, units, image references, or numerical values."""


def decision_contract_message() -> dict[str, Any]:
    """Return a fresh system-message object for the strict decision contract."""
    return {
        "role": "system",
        "content": [{"type": "text", "text": DECISION_SYSTEM_CONTRACT}],
    }


def apply_decision_contract(
    messages: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a deep-copied conversation with exactly one contract message."""
    copied = copy.deepcopy(messages)
    contract = decision_contract_message()
    if copied and copied[0] == contract:
        return copied
    return [contract, *copied]


def decision_contract_enabled(config: Mapping[str, Any]) -> bool:
    """Read the explicit inference contract switch from a model config."""
    orchestration = config.get("orchestration", {})
    return bool(
        isinstance(orchestration, Mapping)
        and orchestration.get("decision_system_contract", False)
    )
