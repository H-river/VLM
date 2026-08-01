"""Resolve compact semantic source mappings into validated decision-tool calls."""

from __future__ import annotations

from typing import Any, Mapping

from .decision_tools import CONTROL_TOOL, SUFFICIENCY_TOOL, run_tool


CONTROL_SOURCE_MAP = {
    "candidate_rows_path": "candidate_action_trials",
    "residual_field": "measured_residual_px",
    "action_field": "action",
    "active_actuator_path": "actuator_constraints.active_actuator",
    "allowed_values_path": "actuator_constraints.allowed_values_mm",
    "tolerance_path": "actuator_constraints.success_tolerance_px",
}

SUFFICIENCY_SOURCE_MAP = {
    "completion_rows_path": "compatible_completion_trials",
    "delta_field": "measured_delta_px",
    "threshold_path": "direction_threshold_px",
}

SOURCE_MAPS = {
    CONTROL_TOOL: CONTROL_SOURCE_MAP,
    SUFFICIENCY_TOOL: SUFFICIENCY_SOURCE_MAP,
}


def resolve_path(root: Mapping[str, Any], path: str) -> Any:
    """Resolve a restricted dot path containing mapping keys only."""

    if not path or path.startswith(".") or path.endswith("."):
        raise ValueError(f"invalid source path: {path!r}")
    value: Any = root
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"source path does not exist: {path}")
        value = value[key]
    return value


def _allowed_index(allowed: list[Any], motion: float) -> int:
    matches = [
        index for index, value in enumerate(allowed) if abs(float(value) - motion) <= 1e-9
    ]
    if len(matches) != 1:
        raise ValueError(f"motion {motion} does not uniquely match the allowed grid")
    return matches[0]


def build_tool_arguments(
    tool_name: str,
    visible_evidence: Mapping[str, Any],
    source_map: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a compact mapping and construct the literal registered arguments."""

    expected = SOURCE_MAPS.get(tool_name)
    if expected is None:
        raise ValueError(f"unsupported mapped tool: {tool_name}")
    if set(source_map) != set(expected) or any(
        not isinstance(value, str) for value in source_map.values()
    ):
        raise ValueError(f"source_map for {tool_name} must contain exactly {sorted(expected)}")

    if tool_name == CONTROL_TOOL:
        rows = resolve_path(visible_evidence, str(source_map["candidate_rows_path"]))
        active = resolve_path(visible_evidence, str(source_map["active_actuator_path"]))
        allowed = resolve_path(visible_evidence, str(source_map["allowed_values_path"]))
        tolerance = resolve_path(visible_evidence, str(source_map["tolerance_path"]))
        if not isinstance(rows, list) or not rows:
            raise ValueError("candidate row source must resolve to a nonempty list")
        if not isinstance(allowed, list) or len(allowed) != len(rows):
            raise ValueError("allowed-value source must match the candidate row count")
        residuals = []
        motions = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("candidate rows must contain objects")
            residual = row.get(str(source_map["residual_field"]))
            action = row.get(str(source_map["action_field"]))
            if not isinstance(action, Mapping) or active not in action:
                raise ValueError("action source does not contain the active actuator")
            residuals.append(residual)
            motions.append(float(action[active]))
        return {
            "candidate_residuals_px": residuals,
            "active_actuator_motions_mm": motions,
            "success_tolerance_px": tolerance,
            "allowed_order_indices": [_allowed_index(allowed, motion) for motion in motions],
        }

    rows = resolve_path(visible_evidence, str(source_map["completion_rows_path"]))
    threshold = resolve_path(visible_evidence, str(source_map["threshold_path"]))
    if not isinstance(rows, list) or not rows:
        raise ValueError("completion row source must resolve to a nonempty list")
    deltas = []
    for row in rows:
        if not isinstance(row, Mapping) or str(source_map["delta_field"]) not in row:
            raise ValueError("completion delta source is missing")
        deltas.append(row[str(source_map["delta_field"])])
    return {
        "measured_deltas_px": deltas,
        "direction_threshold_px": threshold,
    }


def run_mapped_tool(
    tool_name: str,
    visible_evidence: Mapping[str, Any],
    source_map: Mapping[str, Any],
) -> dict[str, Any]:
    return run_tool(tool_name, build_tool_arguments(tool_name, visible_evidence, source_map))


def compact_decision(tool_name: str, tool_result: Mapping[str, Any]) -> dict[str, Any]:
    """Project a validated tool result to the smallest LLM interpretation contract."""

    if tool_name == CONTROL_TOOL:
        required = {
            "successful_action_indices",
            "selected_index",
            "best_residual_index",
            "feasible_within_tolerance",
        }
        if set(tool_result) != required:
            raise ValueError("control tool result has an unexpected schema")
        feasible = bool(tool_result["feasible_within_tolerance"])
        return {
            "status": "feasible" if feasible else "infeasible_within_limits",
            "successful_action_indices": tool_result["successful_action_indices"],
            "selected_index": tool_result["selected_index"],
            "best_residual_index": tool_result["best_residual_index"],
        }
    if tool_name == SUFFICIENCY_TOOL:
        required = {
            "per_trial_directions",
            "observed_direction_set",
            "conflicting_pair_indices",
            "all_directions_agree",
        }
        if set(tool_result) != required:
            raise ValueError("sufficiency tool result has an unexpected schema")
        answerable = bool(tool_result["all_directions_agree"])
        return {
            "status": "answerable" if answerable else "insufficient_information",
            "observed_direction_set": tool_result["observed_direction_set"],
            "conflicting_pair_indices": tool_result["conflicting_pair_indices"],
        }
    raise ValueError(f"unsupported mapped tool: {tool_name}")


def materialize_final_answer(
    tool_name: str,
    tool_result: Mapping[str, Any],
    visible_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically turn validated evidence into the original task response."""

    decision = compact_decision(tool_name, tool_result)
    if tool_name == CONTROL_TOOL:
        rows = visible_evidence.get("candidate_action_trials")
        if not isinstance(rows, list) or not rows:
            raise ValueError("control evidence must contain candidate_action_trials")
        selected = decision["selected_index"]
        best = decision["best_residual_index"]
        if not isinstance(best, int) or not 0 <= best < len(rows):
            raise ValueError("best_residual_index is outside the candidate grid")
        if selected is None:
            answer = {
                "control_plan": None,
                "expected_residual_px": None,
                "best_achievable_residual_px": rows[best]["measured_residual_px"],
            }
        else:
            if not isinstance(selected, int) or not 0 <= selected < len(rows):
                raise ValueError("selected_index is outside the candidate grid")
            answer = {
                "control_plan": rows[selected]["action"],
                "expected_residual_px": rows[selected]["measured_residual_px"],
                "best_achievable_residual_px": None,
            }
        return {"status": decision["status"], "answer": answer}

    rows = visible_evidence.get("compatible_completion_trials")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sufficiency evidence must contain compatible_completion_trials")
    if decision["status"] == "answerable":
        observed = decision["observed_direction_set"]
        if not isinstance(observed, list) or len(observed) != 1:
            raise ValueError("answerable sufficiency result must have one direction")
        answer = {
            "centroid_x_direction": observed[0],
            "missing_fields": [],
            "nonidentifiable_output": None,
            "visible_conflicting_witness": None,
        }
    else:
        pair = decision["conflicting_pair_indices"]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(index, int) and 0 <= index < len(rows) for index in pair)
        ):
            raise ValueError("insufficient result must identify two compatible completions")
        answer = {
            "centroid_x_direction": None,
            "missing_fields": [visible_evidence["hidden_action_field"]],
            "nonidentifiable_output": visible_evidence["questioned_output"],
            "visible_conflicting_witness": [rows[index]["hidden_value_mm"] for index in pair],
        }
    return {"status": decision["status"], "answer": answer}
