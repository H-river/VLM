"""Strict JSON schema for evaluating profile2setup LLM API outputs."""

from __future__ import annotations

from profile2setup.schema import VARIABLE_ORDER

CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)
LEGACY_VARIABLES = ("alignment", "alignment_x", "alignment_y")

TASK_TYPE_EDIT = "edit"
TASK_TYPE_ABSOLUTE = "absolute"
TASK_TYPE_PAIRED_NO_SETUP = "paired_no_setup"
VALID_TASK_TYPES = (
    TASK_TYPE_EDIT,
    TASK_TYPE_ABSOLUTE,
    TASK_TYPE_PAIRED_NO_SETUP,
)

EXPECTED_TOP_LEVEL_FIELDS = (
    "valid",
    "task_type",
    "observed_profile_change",
    "setup_understanding",
    "predicted_delta",
    "predicted_setup",
    "confidence",
    "reasoning_summary",
    "rejection_reason",
)
OBSERVED_PROFILE_CHANGE_FIELDS = (
    "centroid_x",
    "centroid_y",
    "beam_width_x",
    "beam_width_y",
    "peak_intensity",
    "total_intensity",
)


def _variable_schema(description: str) -> dict:
    return {
        "type": "object",
        "description": description,
        "additionalProperties": False,
        "required": list(CANONICAL_VARIABLE_ORDER),
        "properties": {
            variable: {"type": "number"} for variable in CANONICAL_VARIABLE_ORDER
        },
    }


def _nullable_variable_schema(description: str) -> dict:
    return {
        "anyOf": [
            _variable_schema(description),
            {"type": "null"},
        ]
    }


def default_output_schema() -> dict:
    """Return the strict JSON Schema expected from an LLM API response.

    This is suitable as the schema body for API-side structured-output checks.
    Runtime validation still rejects legacy variable names in string values,
    which JSON Schema cannot express portably.
    """
    setup_schema = _variable_schema(
        "A complete optical setup dictionary using exactly the canonical variables."
    )
    nullable_setup_schema = _nullable_variable_schema(
        "A complete optical setup dictionary, or null when no prediction is available."
    )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(EXPECTED_TOP_LEVEL_FIELDS),
        "properties": {
            "valid": {"type": "boolean"},
            "task_type": {"type": "string", "enum": list(VALID_TASK_TYPES)},
            "observed_profile_change": {
                "type": "object",
                "additionalProperties": False,
                "required": list(OBSERVED_PROFILE_CHANGE_FIELDS),
                "properties": {
                    field: {"type": "string"} for field in OBSERVED_PROFILE_CHANGE_FIELDS
                },
            },
            "setup_understanding": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "current_setup",
                    "target_setup",
                    "changed_variables",
                    "change_direction",
                    "notes",
                ],
                "properties": {
                    "current_setup": nullable_setup_schema,
                    "target_setup": nullable_setup_schema,
                    "changed_variables": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(CANONICAL_VARIABLE_ORDER)},
                    },
                    "change_direction": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(CANONICAL_VARIABLE_ORDER),
                        "properties": {
                            variable: {
                                "type": "string",
                                "enum": ["increase", "decrease", "unchanged"],
                            }
                            for variable in CANONICAL_VARIABLE_ORDER
                        },
                    },
                    "notes": {"type": "string"},
                },
            },
            "predicted_delta": _nullable_variable_schema(
                "Predicted target-current setup delta using exactly the canonical variables."
            ),
            "predicted_setup": _nullable_variable_schema(
                "Predicted absolute setup using exactly the canonical variables."
            ),
            "confidence": {"type": "number"},
            "reasoning_summary": {"type": "string"},
            "rejection_reason": {"type": "string"},
        },
    }
    return schema
