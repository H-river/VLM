"""Strict structured-goal boundary between language/VLM parsing and MPC."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from continuous_control_v12.contracts import (
    OUTPUT_FIELDS,
    assert_no_q_star,
    tolerance_vector,
)
from continuous_control_v12.schema import validate_json_schema


@dataclass(frozen=True)
class GoalParseResult:
    status: str
    goal: dict[str, Any] | None
    errors: tuple[str, ...]


class StructuredGoalAdapter:
    """Deterministic replaceable adapter; it never emits actuator actions."""

    def parse(
        self,
        request: str | Mapping[str, Any],
        *,
        current_metrics: Mapping[str, Any],
    ) -> GoalParseResult:
        try:
            value = json.loads(request) if isinstance(request, str) else dict(request)
        except json.JSONDecodeError as exc:
            return GoalParseResult(
                status="needs_clarification",
                goal=None,
                errors=(f"request is not structured JSON: {exc.msg}",),
            )
        try:
            assert_no_q_star(value)
        except ValueError as exc:
            return GoalParseResult(
                status="contradictory",
                goal=None,
                errors=(str(exc),),
            )
        forbidden = set(value) & {
            "action",
            "action_mm",
            "lens_x_delta_mm",
            "lens_y_delta_mm",
            "camera_x_delta_mm",
            "camera_y_delta_mm",
        }
        if forbidden:
            return GoalParseResult(
                status="contradictory",
                goal=None,
                errors=(
                    "language/VLM goals cannot directly contain actuator actions: "
                    + ", ".join(sorted(forbidden)),
                ),
            )
        if "target_metrics" not in value:
            return GoalParseResult(
                status="needs_clarification",
                goal=None,
                errors=("target_metrics is required",),
            )
        tolerance = tolerance_vector(current_metrics)
        goal = {
            "target_metrics": {
                field: float(value["target_metrics"][field])
                for field in OUTPUT_FIELDS
            },
            "metric_tolerances": value.get(
                "metric_tolerances",
                {
                    field: float(tolerance[index])
                    for index, field in enumerate(OUTPUT_FIELDS)
                },
            ),
            "target_image": value.get("target_image"),
            "allowed_dofs": value.get(
                "allowed_dofs",
                ["lens_x", "lens_y", "camera_x", "camera_y"],
            ),
            "priority": value.get(
                "priority", ["centroid", "width", "peak_intensity"]
            ),
            "control_mode": value.get("control_mode", "closed_loop"),
        }
        if "constraints" in value:
            goal["constraints"] = value["constraints"]
        errors = []
        for field, metric in goal["target_metrics"].items():
            if not math.isfinite(float(metric)):
                errors.append(f"target metric {field} is non-finite")
        if not goal["allowed_dofs"]:
            errors.append("allowed_dofs cannot be empty for active control")
        try:
            validate_json_schema(goal, "goal_v12.schema.json")
        except Exception as exc:
            errors.append(str(exc))
        if errors:
            return GoalParseResult(
                status="contradictory", goal=None, errors=tuple(errors)
            )
        return GoalParseResult(status="ready", goal=goal, errors=())


def vlm_goal_parser_integration_point(
    request: Any,
    *,
    current_metrics: Mapping[str, Any],
    external_parser: Any | None = None,
) -> GoalParseResult:
    """Use a replaceable VLM parser, then enforce the deterministic boundary."""

    structured = (
        external_parser(request) if external_parser is not None else request
    )
    return StructuredGoalAdapter().parse(
        structured, current_metrics=current_metrics
    )

