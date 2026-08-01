"""Frozen supervisor v1 enums, mappings, and visible-message contract."""

from __future__ import annotations

import json
from typing import Any, Mapping

SCHEMA_VERSION = "qwen_vl_supervisor_manifest_v1.0.0"
TASK_TYPE = "static_anomaly_policy"

DIAGNOSES = ("nominal", "sensor_saturation", "secondary_reflection")
MEASUREMENT_POLICIES = (
    "standard",
    "lower_exposure_reacquire",
    "primary_spot",
)
SUPERVISOR_ACTIONS = ("reacquire", "switch_measurement", "execute", "continue", "stop")
TARGET_KEYS = ("diagnosis", "measurement_policy", "supervisor_action")

SOURCE_POLICY_TO_CANONICAL = {
    "standard_metrics": "standard",
    "reduce_exposure_reacquire": "lower_exposure_reacquire",
    "primary_spot_specialist": "primary_spot",
}
CANONICAL_POLICY_TO_SOURCE = {value: key for key, value in SOURCE_POLICY_TO_CANONICAL.items()}

# This is an explicit, versioned semantic projection from a static recovery-policy
# decision to the high-level action. It creates no temporal continue/stop label.
POLICY_TO_STATIC_ACTION = {
    "standard": "execute",
    "lower_exposure_reacquire": "reacquire",
    "primary_spot": "switch_measurement",
}
STATIC_ACTION_MAPPING_VERSION = "static_policy_action_mapping_v1"

SYSTEM_PROMPT = """You are the high-level supervisor for a frozen optical control system.
Use the current beam image and structured numerical state to diagnose the observation, select a measurement/recovery policy, and select a high-level supervisor action.
Never output continuous actuator commands, actuator deltas, positions, or other numeric control actions. The frozen numerical controller alone chooses continuous actions.
Choose exactly one value for each field from these frozen enums:
diagnosis: nominal, sensor_saturation, secondary_reflection
measurement_policy: standard, lower_exposure_reacquire, primary_spot
supervisor_action: reacquire, switch_measurement, execute, continue, stop
Return exactly one JSON object with exactly the keys diagnosis, measurement_policy, and supervisor_action. Return no rationale, confidence, markdown, or extra text."""


def canonical_target_text(target: Mapping[str, Any]) -> str:
    """Serialize a target in the sole accepted key order and compact JSON form."""

    validate_target(target)
    ordered = {key: target[key] for key in TARGET_KEYS}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def validate_target(target: Mapping[str, Any]) -> None:
    if not isinstance(target, Mapping):
        raise ValueError("target must be an object")
    if set(target) != set(TARGET_KEYS):
        raise ValueError(f"target must contain exactly {TARGET_KEYS}")
    if target["diagnosis"] not in DIAGNOSES:
        raise ValueError(f"invalid diagnosis: {target['diagnosis']!r}")
    if target["measurement_policy"] not in MEASUREMENT_POLICIES:
        raise ValueError(f"invalid measurement_policy: {target['measurement_policy']!r}")
    if target["supervisor_action"] not in SUPERVISOR_ACTIONS:
        raise ValueError(f"invalid supervisor_action: {target['supervisor_action']!r}")


def parse_target_strict(text: str) -> dict[str, str]:
    """Parse a whole-string canonical decision; embedded JSON is rejected."""

    if not isinstance(text, str):
        raise ValueError("model output must be text")
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed_object: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed_object:
                raise ValueError(f"duplicate JSON key {key!r}")
            parsed_object[key] = value
        return parsed_object

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid whole-string JSON: {exc}") from exc
    validate_target(parsed)
    return {key: parsed[key] for key in TARGET_KEYS}


def model_visible_state(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the allow-listed structured state placed in a user message."""

    model_input = record["model_input"]
    return {
        "decision_context": record["task_type"],
        "current_metrics": model_input["current_metrics"],
        "goal_metrics": model_input["goal_metrics"],
        "recent_history": model_input["recent_history"],
        "remaining_step_budget": model_input["remaining_step_budget"],
        "actuator_constraints": model_input["actuator_constraints"],
    }
