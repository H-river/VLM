"""Schema and validation helpers for profile2setup v2 datasets."""

from __future__ import annotations

from typing import Any

VARIABLE_ORDER = [
    "source_to_lens",
    "lens_to_camera",
    "focal_length",
    "lens_x",
    "lens_y",
    "camera_x",
    "camera_y",
]

NUM_VARIABLES = 7

_FORBIDDEN_KEYS = {"alignment", "alignment_x", "alignment_y"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def missing_setup_keys(setup: dict) -> list[str]:
    """Return canonical v2 keys that are missing from a setup dict."""
    if not isinstance(setup, dict):
        return list(VARIABLE_ORDER)
    return [key for key in VARIABLE_ORDER if key not in setup]


def contains_forbidden_v2_keys(obj: Any) -> bool:
    """Recursively detect forbidden keys in nested dict/list structures."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in _FORBIDDEN_KEYS:
                return True
            if contains_forbidden_v2_keys(value):
                return True
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if contains_forbidden_v2_keys(item):
                return True
    return False


def validate_setup_dict(setup: dict) -> bool:
    """Return True if setup is exactly the canonical 7-variable v2 setup dict."""
    if not isinstance(setup, dict):
        return False
    if contains_forbidden_v2_keys(setup):
        return False
    if set(setup.keys()) != set(VARIABLE_ORDER):
        return False
    if missing_setup_keys(setup):
        return False
    return all(_is_number(setup[key]) for key in VARIABLE_ORDER)


def compute_delta_setup(current_setup: dict, target_setup: dict) -> dict:
    """Compute target-current deltas for all canonical variables."""
    if not validate_setup_dict(current_setup):
        raise ValueError("Invalid current_setup; expected canonical v2 setup dict")
    if not validate_setup_dict(target_setup):
        raise ValueError("Invalid target_setup; expected canonical v2 setup dict")

    return {
        key: float(target_setup[key]) - float(current_setup[key])
        for key in VARIABLE_ORDER
    }


def setup_to_ordered_list(setup: dict) -> list[float]:
    """Convert canonical setup dict to ordered list using VARIABLE_ORDER."""
    if not validate_setup_dict(setup):
        raise ValueError("Invalid setup dict; expected canonical v2 setup dict")
    return [float(setup[key]) for key in VARIABLE_ORDER]


def ordered_list_to_setup(values: list[float]) -> dict:
    """Convert ordered values back into canonical setup dict."""
    if not isinstance(values, list):
        raise ValueError("values must be a list")
    if len(values) != NUM_VARIABLES:
        raise ValueError(f"values must have length {NUM_VARIABLES}")
    if any(not _is_number(v) for v in values):
        raise ValueError("values must contain numeric entries")

    return {key: float(values[i]) for i, key in enumerate(VARIABLE_ORDER)}


def validate_dataset_record(record: dict, strict: bool = True) -> bool:
    """Validate profile2setup dataset record for absolute/edit tasks."""

    def _fail(message: str) -> bool:
        if strict:
            raise ValueError(message)
        return False

    if not isinstance(record, dict):
        return _fail("record must be a dict")

    if contains_forbidden_v2_keys(record):
        return _fail("record contains forbidden v2 keys (alignment/alignment_x/alignment_y)")

    required_top = [
        "id",
        "task_type",
        "prompt",
        "target_profile_path",
        "target_setup",
        "profile_loss_reference",
    ]
    for key in required_top:
        if key not in record:
            return _fail(f"record missing required key: {key}")

    if not record["id"]:
        return _fail("record.id must be non-empty")
    if record["task_type"] not in {"absolute", "edit"}:
        return _fail("record.task_type must be 'absolute' or 'edit'")
    if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
        return _fail("record.prompt must be a non-empty string")
    if not isinstance(record.get("target_profile_path"), str) or not record["target_profile_path"]:
        return _fail("record.target_profile_path must be a non-empty string")

    if not validate_setup_dict(record.get("target_setup")):
        return _fail("record.target_setup must be canonical 7-variable v2 setup")

    plr = record.get("profile_loss_reference")
    if not isinstance(plr, dict):
        return _fail("record.profile_loss_reference must be a dict")
    if not isinstance(plr.get("target_profile_path"), str) or not plr["target_profile_path"]:
        return _fail("profile_loss_reference.target_profile_path is required")

    if record["task_type"] == "absolute":
        if record.get("current_profile_path") is not None:
            return _fail("absolute record current_profile_path must be null")
        if record.get("current_setup") is not None:
            return _fail("absolute record current_setup must be null")
        if record.get("target_delta") is not None:
            return _fail("absolute record target_delta must be null")

    if record["task_type"] == "edit":
        if not isinstance(record.get("current_profile_path"), str) or not record["current_profile_path"]:
            return _fail("edit record current_profile_path must be a non-empty string")
        if not validate_setup_dict(record.get("current_setup")):
            return _fail("edit record current_setup must be canonical 7-variable v2 setup")
        if not validate_setup_dict(record.get("target_delta")):
            return _fail("edit record target_delta must be canonical 7-variable delta dict")
        if not isinstance(plr.get("current_profile_path"), str) or not plr["current_profile_path"]:
            return _fail("edit profile_loss_reference.current_profile_path is required")
        if not isinstance(plr.get("target_profile_path"), str) or not plr["target_profile_path"]:
            return _fail("edit profile_loss_reference.target_profile_path is required")

    return True
