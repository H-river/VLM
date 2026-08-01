"""Safe canonicalization for decisions that never execute a specialist."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TASK_TYPES = {
    "beam_profile_measurement",
    "direction_prediction",
    "forward_prediction",
    "inverse_control",
}


def normalize_nonexecuting_decision(
    decision: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Normalize only clarification/unsupported decisions.

    Ready decisions are returned unchanged so normalization can never choose a
    route, alter a numerical argument, or alter an image binding that will be
    sent to a specialist.
    """
    status = decision.get("status")
    if status == "needs_clarification":
        task_type = decision.get("task_type")
        normalized = {
            "schema_version": decision.get("schema_version"),
            "status": status,
            "task_type": task_type if task_type in TASK_TYPES else None,
            "route_name": None,
            "arguments": (
                dict(decision["arguments"])
                if isinstance(decision.get("arguments"), Mapping)
                else {}
            ),
            "image_roles": (
                dict(decision["image_roles"])
                if isinstance(decision.get("image_roles"), Mapping)
                else {}
            ),
            "missing_fields": decision.get("missing_fields"),
            "clarification_question": decision.get("clarification_question"),
        }
    elif status == "unsupported":
        normalized = {
            "schema_version": decision.get("schema_version"),
            "status": status,
            "task_type": None,
            "route_name": None,
            "arguments": {},
            "image_roles": {},
            "missing_fields": [],
            "clarification_question": None,
        }
    else:
        return decision, False
    return normalized, normalized != decision
