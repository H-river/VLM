"""Validation for profile2setup reasoning-layer commands."""

from __future__ import annotations

import re
from typing import Any

from .schema import (
    CANONICAL_VARIABLE_ORDER,
    CHANGE_MASK_PRIORS,
    EXPECTED_TOP_LEVEL_FIELDS,
    PROFILE_CHANGE_VALUES,
    VALID_TASK_TYPES,
)

LEGACY_VARIABLES = {"alignment", "alignment_x", "alignment_y"}
_CANONICAL_SET = set(CANONICAL_VARIABLE_ORDER)
_LEGACY_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(alignment|alignment_x|alignment_y)(?![A-Za-z0-9_])")


def _path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def _assert_json_like(obj: Any, path: str = "command") -> None:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            _assert_json_like(item, f"{path}[{idx}]")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key: {key!r}")
            _assert_json_like(value, _path(path, key))
        return
    raise ValueError(f"{path} contains non-JSON-like value of type {type(obj).__name__}")


def _find_legacy_token(obj: Any, path: str = "command") -> tuple[str, str] | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in LEGACY_VARIABLES:
                return path, key
            match = _find_legacy_token(value, _path(path, key))
            if match is not None:
                return match
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            match = _find_legacy_token(item, f"{path}[{idx}]")
            if match is not None:
                return match
    elif isinstance(obj, str):
        match = _LEGACY_PATTERN.search(obj)
        if match is not None:
            return path, match.group(1)
    return None


def _require_dict(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_string_list(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")

    values: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{path}[{idx}] must be a string")
        values.append(item)
    return values


def _validate_variable_list(value: Any, path: str) -> list[str]:
    variables = _require_string_list(value, path)
    for variable in variables:
        if variable not in _CANONICAL_SET:
            raise ValueError(f"{path} contains non-canonical variable: {variable}")
    return variables


def _validate_change_mask_prior(value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")

    for variable, prior in value.items():
        if variable not in _CANONICAL_SET:
            raise ValueError(f"{path} contains non-canonical variable key: {variable}")
        if prior not in CHANGE_MASK_PRIORS:
            raise ValueError(
                f"{path}.{variable} must be one of {list(CHANGE_MASK_PRIORS)}, got {prior!r}"
            )


def _validate_observed_profile_change(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value not in PROFILE_CHANGE_VALUES:
            raise ValueError(
                "observed_profile_change must use known profile-change values "
                f"or an object containing them; got {value!r}"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("observed_profile_change keys must be strings")
            if isinstance(item, str) and item not in PROFILE_CHANGE_VALUES:
                raise ValueError(
                    f"observed_profile_change.{key} must be one of "
                    f"{list(PROFILE_CHANGE_VALUES)}, got {item!r}"
                )
        return
    raise ValueError("observed_profile_change must be a string, object, or null")


def validate_reasoning_command(command: dict) -> dict:
    """Validate and return a reasoning command.

    The command remains a plain dict. This function enforces the Stage 1
    profile2setup contract and rejects legacy variable names anywhere in the
    object, including free-text fields.
    """
    if not isinstance(command, dict):
        raise ValueError("command must be a JSON-like object")

    _assert_json_like(command)
    legacy = _find_legacy_token(command)
    if legacy is not None:
        path, token = legacy
        raise ValueError(f"legacy variable {token!r} is not allowed at {path}")

    missing = [field for field in EXPECTED_TOP_LEVEL_FIELDS if field not in command]
    if missing:
        raise ValueError(f"command missing required top-level fields: {missing}")

    if not isinstance(command["valid"], bool):
        raise ValueError("valid must be a boolean")
    if command["task_type"] not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {list(VALID_TASK_TYPES)}")

    _validate_observed_profile_change(command["observed_profile_change"])

    if not isinstance(command["requested_goal"], str):
        raise ValueError("requested_goal must be a string")
    if not isinstance(command["canonical_prompt"], str):
        raise ValueError("canonical_prompt must be a string")
    if not isinstance(command["conflict_detected"], bool):
        raise ValueError("conflict_detected must be a boolean")
    if not isinstance(command["reasoning_summary"], str):
        raise ValueError("reasoning_summary must be a string")
    if not isinstance(command["rejection_reason"], str):
        raise ValueError("rejection_reason must be a string")

    unsupported = _require_string_list(command["unsupported_requests"], "unsupported_requests")
    if command["valid"] and command["rejection_reason"]:
        raise ValueError("valid commands must not include a rejection_reason")
    if not command["valid"] and not (
        command["rejection_reason"] or command["conflict_detected"] or unsupported
    ):
        raise ValueError(
            "invalid commands must include rejection_reason, conflict_detected, "
            "or unsupported_requests"
        )

    constraints = _require_dict(command["constraints"], "constraints")
    control_plan = _require_dict(command["control_plan"], "control_plan")

    _validate_variable_list(constraints.get("fixed_variables", []), "constraints.fixed_variables")
    _validate_variable_list(constraints.get("allowed_variables", []), "constraints.allowed_variables")
    _validate_variable_list(
        control_plan.get("likely_relevant_variables", []),
        "control_plan.likely_relevant_variables",
    )
    _validate_variable_list(control_plan.get("avoid_variables", []), "control_plan.avoid_variables")
    _validate_change_mask_prior(constraints.get("change_mask_prior", {}), "constraints.change_mask_prior")

    return command
