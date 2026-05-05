"""Validation helpers for profile2setup LLM API JSON outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from .schema import (
    CANONICAL_VARIABLE_ORDER,
    EXPECTED_TOP_LEVEL_FIELDS,
    LEGACY_VARIABLES,
    VALID_TASK_TYPES,
)

_CANONICAL_SET = set(CANONICAL_VARIABLE_ORDER)
_LEGACY_SET = set(LEGACY_VARIABLES)
_LEGACY_TOKEN_ALTERNATION = "|".join(
    re.escape(token) for token in sorted(_LEGACY_SET, key=len, reverse=True)
)
_LEGACY_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_])({_LEGACY_TOKEN_ALTERNATION})(?![A-Za-z0-9_])"
)


def _path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _assert_json_like(obj: Any, path: str = "output") -> None:
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


def _find_legacy_token(obj: Any, path: str = "output") -> tuple[str, str] | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _LEGACY_SET:
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


def validate_variable_dict(obj: Any, path: str = "variables") -> dict:
    """Validate a numeric dict with exactly the canonical 7 variables."""
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be an object")

    keys = set(obj.keys())
    missing = [key for key in CANONICAL_VARIABLE_ORDER if key not in obj]
    extra = sorted(keys - _CANONICAL_SET)
    if missing:
        raise ValueError(f"{path} missing canonical variables: {missing}")
    if extra:
        raise ValueError(f"{path} contains non-canonical variables: {extra}")

    for key in CANONICAL_VARIABLE_ORDER:
        if not _is_number(obj[key]):
            raise ValueError(f"{path}.{key} must be numeric")
    return obj


def _looks_like_variable_dict(obj: dict) -> bool:
    return bool(set(obj.keys()) & _CANONICAL_SET)


def validate_all_variable_dicts(obj: Any, path: str = "output") -> None:
    """Check that every variable dictionary uses exactly the canonical 7 keys.

    A nested object is treated as a variable dictionary when it contains any
    canonical variable key. Legacy-key rejection is handled separately for the
    whole response before this check runs.
    """
    if path.endswith(".change_direction"):
        return
    if isinstance(obj, dict):
        if _looks_like_variable_dict(obj):
            validate_variable_dict(obj, path)
        for key, value in obj.items():
            validate_all_variable_dicts(value, _path(path, key))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            validate_all_variable_dicts(item, f"{path}[{idx}]")


def validate_predicted_setup(obj: Any) -> dict:
    """Validate a predicted absolute setup dictionary."""
    return validate_variable_dict(obj, "predicted_setup")


def validate_predicted_delta(obj: Any) -> dict:
    """Validate a predicted target-current setup delta dictionary."""
    return validate_variable_dict(obj, "predicted_delta")


def _validate_nullable_variable_dict(obj: Any, path: str) -> dict | None:
    if obj is None:
        return None
    return validate_variable_dict(obj, path)


def validate_llm_output(obj: Any) -> dict:
    """Validate and return a strict profile2setup LLM API output object."""
    if not isinstance(obj, dict):
        raise ValueError("output must be a JSON object")

    _assert_json_like(obj)
    legacy = _find_legacy_token(obj)
    if legacy is not None:
        path, token = legacy
        raise ValueError(f"legacy variable {token!r} is not allowed at {path}")

    expected = set(EXPECTED_TOP_LEVEL_FIELDS)
    keys = set(obj.keys())
    missing = [field for field in EXPECTED_TOP_LEVEL_FIELDS if field not in obj]
    extra = sorted(keys - expected)
    if missing:
        raise ValueError(f"output missing required top-level fields: {missing}")
    if extra:
        raise ValueError(f"output contains unexpected top-level fields: {extra}")

    if not isinstance(obj["valid"], bool):
        raise ValueError("valid must be a boolean")
    if obj["task_type"] not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {list(VALID_TASK_TYPES)}")
    if not isinstance(obj["observed_profile_change"], dict):
        raise ValueError("observed_profile_change must be an object")

    setup_understanding = _require_dict(obj["setup_understanding"], "setup_understanding")
    allowed_understanding = {
        "current_setup",
        "target_setup",
        "changed_variables",
        "change_direction",
        "notes",
    }
    missing_understanding = [
        key
        for key in (
            "current_setup",
            "target_setup",
            "changed_variables",
            "change_direction",
            "notes",
        )
        if key not in setup_understanding
    ]
    if missing_understanding:
        raise ValueError(
            "setup_understanding missing required fields: "
            f"{missing_understanding}"
        )
    unknown_understanding = sorted(set(setup_understanding.keys()) - allowed_understanding)
    if unknown_understanding:
        raise ValueError(
            "setup_understanding contains unexpected fields: "
            f"{unknown_understanding}"
        )
    if not isinstance(setup_understanding["notes"], str):
        raise ValueError("setup_understanding.notes must be a string")
    _validate_nullable_variable_dict(
        setup_understanding.get("current_setup"),
        "setup_understanding.current_setup",
    )
    _validate_nullable_variable_dict(
        setup_understanding.get("target_setup"),
        "setup_understanding.target_setup",
    )
    changed_variables = setup_understanding["changed_variables"]
    if not isinstance(changed_variables, list):
        raise ValueError("setup_understanding.changed_variables must be a list")
    for idx, variable in enumerate(changed_variables):
        if variable not in CANONICAL_VARIABLE_ORDER:
            raise ValueError(
                "setup_understanding.changed_variables "
                f"contains non-canonical variable at index {idx}: {variable!r}"
            )
    if len(set(changed_variables)) != len(changed_variables):
        raise ValueError("setup_understanding.changed_variables must not contain duplicates")

    change_direction = _require_dict(
        setup_understanding["change_direction"],
        "setup_understanding.change_direction",
    )
    direction_keys = set(change_direction.keys())
    missing_direction = [key for key in CANONICAL_VARIABLE_ORDER if key not in change_direction]
    extra_direction = sorted(direction_keys - _CANONICAL_SET)
    if missing_direction:
        raise ValueError(
            "setup_understanding.change_direction missing canonical variables: "
            f"{missing_direction}"
        )
    if extra_direction:
        raise ValueError(
            "setup_understanding.change_direction contains non-canonical variables: "
            f"{extra_direction}"
        )
    for key in CANONICAL_VARIABLE_ORDER:
        if change_direction[key] not in {"increase", "decrease", "unchanged"}:
            raise ValueError(
                "setup_understanding.change_direction "
                f"{key} must be increase, decrease, or unchanged"
            )

    _validate_nullable_variable_dict(obj["predicted_delta"], "predicted_delta")
    _validate_nullable_variable_dict(obj["predicted_setup"], "predicted_setup")
    validate_all_variable_dicts(obj)

    if not _is_number(obj["confidence"]):
        raise ValueError("confidence must be numeric")
    confidence = float(obj["confidence"])
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    if not isinstance(obj["reasoning_summary"], str):
        raise ValueError("reasoning_summary must be a string")
    if not isinstance(obj["rejection_reason"], str):
        raise ValueError("rejection_reason must be a string")

    if obj["valid"]:
        if obj["rejection_reason"]:
            raise ValueError("valid outputs must not include a rejection_reason")
        if obj["predicted_delta"] is None and obj["predicted_setup"] is None:
            raise ValueError("valid outputs must include predicted_delta or predicted_setup")
    elif not obj["rejection_reason"]:
        raise ValueError("invalid outputs must include a rejection_reason")

    return obj


def parse_llm_json_text(text: str) -> dict:
    """Parse strict JSON text and return a validated LLM API output."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM output text must be a non-empty string")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid strict JSON: {exc}") from exc
    return validate_llm_output(parsed)


def parse_json_text(text: str) -> dict:
    """Alias for parse_llm_json_text for callers that use generic naming."""
    return parse_llm_json_text(text)
