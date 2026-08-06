"""Deterministic user-facing formatting for orchestration outcomes."""

from __future__ import annotations

import json
from typing import Any, Mapping


def format_outcome(outcome: Mapping[str, Any]) -> str:
    """Format without changing, rounding, or inventing numerical values."""
    status = outcome["status"]
    if status == "needs_clarification":
        return str(outcome["clarification_question"])
    if status == "unsupported":
        supported = ", ".join(outcome["supported_task_types"])
        return f"Unsupported request. Supported task types: {supported}."
    result = json.dumps(
        outcome["result"],
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return f"Executed {outcome['route_name']}. Result: {result}"
