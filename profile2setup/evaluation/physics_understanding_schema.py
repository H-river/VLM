"""Schema helpers for physics-understanding diagnostic probe JSONL files."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from profile2setup.schema import VARIABLE_ORDER, contains_forbidden_v2_keys, validate_setup_dict

CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)
INPUT_MODES = {
    "prompt_only",
    "images_only",
    "prompt_plus_images",
    "shuffled_prompt",
    "conflict",
}
CHANGE_DIRECTIONS = {"increase", "decrease", "unchanged"}
TASK_TYPES = {"absolute", "edit", "paired_no_setup"}
_FORBIDDEN_ROOT = "align" + "ment"
FORBIDDEN_LEGACY_VARIABLES = {
    _FORBIDDEN_ROOT,
    f"{_FORBIDDEN_ROOT}_x",
    f"{_FORBIDDEN_ROOT}_y",
}
REQUIRED_FIELDS = (
    "probe_id",
    "probe_type",
    "base_record_id",
    "task_type",
    "prompt",
    "input_mode",
    "current_profile_path",
    "target_profile_path",
    "current_setup",
    "expected_valid",
    "expected_changed_variables",
    "expected_change_direction",
    "fixed_variables",
    "allowed_variables",
    "expected_rejection_keywords",
    "notes",
)
__all__ = [
    "CANONICAL_VARIABLE_ORDER",
    "CHANGE_DIRECTIONS",
    "FORBIDDEN_LEGACY_VARIABLES",
    "INPUT_MODES",
    "REQUIRED_FIELDS",
    "TASK_TYPES",
    "load_probe_jsonl",
    "validate_probe_record",
    "write_probe_jsonl",
]


def _fail(message: str) -> bool:
    raise ValueError(message)


def _is_string_or_null(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _is_profile_path_or_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and bool(value) and value.endswith(".npy"))


def _legacy_token_pattern() -> re.Pattern[str]:
    tokens = sorted(FORBIDDEN_LEGACY_VARIABLES, key=len, reverse=True)
    alternation = "|".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![A-Za-z0-9_])(?:{alternation})(?![A-Za-z0-9_])")


def _contains_legacy_variable_string(obj: Any) -> bool:
    pattern = _legacy_token_pattern()

    def _walk(value: Any) -> bool:
        if isinstance(value, str):
            return pattern.search(value) is not None
        if isinstance(value, dict):
            return any(_walk(key) or _walk(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return any(_walk(item) for item in value)
        return False

    return _walk(obj)


def _validate_variable_list(value: Any, *, field_name: str, nullable: bool) -> bool:
    if value is None:
        if nullable:
            return True
        return _fail(f"{field_name} must be a list")
    if not isinstance(value, list):
        return _fail(f"{field_name} must be a list")
    invalid = [item for item in value if item not in CANONICAL_VARIABLE_ORDER]
    if invalid:
        return _fail(f"{field_name} contains non-canonical variables: {invalid}")
    if len(value) != len(set(value)):
        return _fail(f"{field_name} must not contain duplicate variables")
    return True


def _validate_change_direction(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict):
        return _fail("expected_change_direction must be a dict or null")
    invalid_keys = [key for key in value if key not in CANONICAL_VARIABLE_ORDER]
    if invalid_keys:
        return _fail(f"expected_change_direction contains non-canonical variables: {invalid_keys}")
    invalid_values = {
        key: direction
        for key, direction in value.items()
        if direction not in CHANGE_DIRECTIONS
    }
    if invalid_values:
        return _fail(
            "expected_change_direction values must be increase, decrease, or unchanged: "
            f"{invalid_values}"
        )
    return True


def _validate_string_list(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, list):
        return _fail(f"{field_name} must be a list")
    if not all(isinstance(item, str) for item in value):
        return _fail(f"{field_name} must contain only strings")
    return True


def validate_probe_record(record: dict) -> bool:
    """Validate one physics-understanding probe record.

    Returns True when valid and raises ValueError with a field-specific message
    otherwise.
    """
    if not isinstance(record, dict):
        return _fail("probe record must be a dict")
    if contains_forbidden_v2_keys(record) or _contains_legacy_variable_string(record):
        return _fail("probe record contains forbidden legacy variable names")

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        return _fail(f"probe record missing required fields: {missing}")
    extra = sorted(set(record) - set(REQUIRED_FIELDS))
    if extra:
        return _fail(f"probe record contains unexpected fields: {extra}")

    if not isinstance(record["probe_id"], str) or not record["probe_id"].strip():
        return _fail("probe_id must be a non-empty string")
    if not isinstance(record["probe_type"], str) or not record["probe_type"].strip():
        return _fail("probe_type must be a non-empty string")
    if not _is_string_or_null(record["base_record_id"]):
        return _fail("base_record_id must be a string or null")
    if record["base_record_id"] == "":
        return _fail("base_record_id must be non-empty when provided")

    if record["task_type"] not in TASK_TYPES:
        return _fail(f"task_type must be one of {sorted(TASK_TYPES)}")
    if not isinstance(record["prompt"], str):
        return _fail("prompt must be a string")
    if record["input_mode"] not in INPUT_MODES:
        return _fail(f"input_mode must be one of {sorted(INPUT_MODES)}")

    if not _is_profile_path_or_null(record["current_profile_path"]):
        return _fail("current_profile_path must be null or a non-empty .npy path")
    if not _is_profile_path_or_null(record["target_profile_path"]):
        return _fail("target_profile_path must be null or a non-empty .npy path")

    current_setup = record["current_setup"]
    if current_setup is not None and not validate_setup_dict(current_setup):
        return _fail("current_setup must be null or a canonical setup dict")

    if not isinstance(record["expected_valid"], bool):
        return _fail("expected_valid must be a bool")
    _validate_variable_list(
        record["expected_changed_variables"],
        field_name="expected_changed_variables",
        nullable=True,
    )
    _validate_change_direction(record["expected_change_direction"])
    _validate_variable_list(record["fixed_variables"], field_name="fixed_variables", nullable=False)
    _validate_variable_list(record["allowed_variables"], field_name="allowed_variables", nullable=True)
    _validate_string_list(record["expected_rejection_keywords"], field_name="expected_rejection_keywords")
    if not isinstance(record["notes"], str):
        return _fail("notes must be a string")

    return True


def load_probe_jsonl(path) -> list[dict]:
    """Load and validate a physics-understanding probe JSONL file."""
    probe_path = Path(path)
    records: list[dict] = []
    with probe_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{probe_path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{probe_path}:{line_number} record must be a JSON object")
            try:
                validate_probe_record(record)
            except ValueError as exc:
                raise ValueError(f"{probe_path}:{line_number}: {exc}") from exc
            records.append(record)
    return records


def write_probe_jsonl(records: Iterable[dict], path) -> int:
    """Validate and write physics-understanding probe records as JSONL."""
    probe_path = Path(path)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with probe_path.open("w", encoding="utf-8") as f:
        for index, record in enumerate(records, start=1):
            try:
                validate_probe_record(record)
            except ValueError as exc:
                raise ValueError(f"record {index}: {exc}") from exc
            f.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def _minimal_setup() -> dict[str, float]:
    return {name: 0.1 for name in CANONICAL_VARIABLE_ORDER}


def _self_check() -> None:
    """Run a small dependency-free validation self-check."""
    good = {
        "probe_id": "probe_0001",
        "probe_type": "direction",
        "base_record_id": "edit_0001",
        "task_type": "edit",
        "prompt": "move the beam up and keep camera fixed",
        "input_mode": "prompt_plus_images",
        "current_profile_path": "current/intensity.npy",
        "target_profile_path": "target/intensity.npy",
        "current_setup": _minimal_setup(),
        "expected_valid": True,
        "expected_changed_variables": ["lens_y"],
        "expected_change_direction": {"lens_y": "increase", "camera_x": "unchanged"},
        "fixed_variables": ["camera_x", "camera_y"],
        "allowed_variables": ["lens_y"],
        "expected_rejection_keywords": [],
        "notes": "synthetic self-check record",
    }
    assert validate_probe_record(good) is True

    bad = dict(good)
    bad["fixed_variables"] = [_FORBIDDEN_ROOT]
    try:
        validate_probe_record(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy variable names must be rejected")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "probes.jsonl"
        assert write_probe_jsonl([good], path) == 1
        loaded = load_probe_jsonl(path)
        assert loaded == [good]


if __name__ == "__main__":
    _self_check()
