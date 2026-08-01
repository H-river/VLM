#!/usr/bin/env python3
"""Audit the design-only evidence-grounded v5 probe and simple baselines."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .analyze_status_shortcuts import best_binary_rule, best_threshold
from .core import centroid_distance, read_jsonl


FORBIDDEN_INPUT_KEYS = {
    "after_state",
    "best_action",
    "completion_evidence",
    "derived_direction",
    "expected_state",
    "grid_scores",
    "label",
    "status",
    "success",
    "target_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in FORBIDDEN_INPUT_KEYS:
                found.add(str(key))
            found.update(forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(forbidden_keys(item))
    return found


def direction(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "increase"
    if delta < -threshold:
        return "decrease"
    return "no_change"


def action_value(trial: Mapping[str, Any], actuator: str) -> float:
    return float(trial["action"][actuator])


def control_solution(record: Mapping[str, Any]) -> dict[str, Any]:
    inputs = record["prompt_inputs"]
    constraints = inputs["actuator_constraints"]
    actuator = str(constraints["active_actuator"])
    allowed = [float(value) for value in constraints["allowed_values_mm"]]
    tolerance = float(constraints["success_tolerance_px"])
    trials = inputs["candidate_action_trials"]
    scored = []
    for trial in trials:
        value = action_value(trial, actuator)
        matches = [index for index, allowed_value in enumerate(allowed) if abs(value - allowed_value) <= 1e-9]
        if len(matches) != 1:
            raise ValueError(f"trial action is not uniquely allowed: {record['example_id']}")
        residual = centroid_distance(
            {
                "centroid_x_px": trial["measured_centroid_x_px"],
                "centroid_y_px": trial["measured_centroid_y_px"],
            },
            inputs["target_observation"],
        )
        scored.append((trial, residual, matches[0]))
    if len(scored) != len(allowed) or len({index for _, _, index in scored}) != len(allowed):
        raise ValueError(f"incomplete action evidence: {record['example_id']}")
    successful = [item for item in scored if item[1] <= tolerance]
    if not successful:
        return {
            "status": "infeasible_within_limits",
            "control_plan": None,
            "expected_residual_px": None,
            "best_achievable_residual_px": min(item[1] for item in scored),
        }
    selected = min(
        successful,
        key=lambda item: (abs(action_value(item[0], actuator)), item[1], item[2]),
    )
    return {
        "status": "feasible",
        "control_plan": selected[0]["action"],
        "expected_residual_px": selected[1],
        "best_achievable_residual_px": None,
    }


def sufficiency_solution(record: Mapping[str, Any]) -> dict[str, Any]:
    inputs = record["prompt_inputs"]
    threshold = float(inputs["direction_threshold_px"])
    current = float(inputs["current_observation"]["centroid_x_px"])
    trials = inputs["compatible_completion_trials"]
    directions = [
        direction(
            float(trial["measured_centroid_x_px"]) - current,
            threshold,
        )
        for trial in trials
    ]
    if len(set(directions)) == 1:
        return {
            "status": "answerable",
            "centroid_x_direction": directions[0],
            "visible_conflicting_witness": None,
        }
    witness = None
    for left_index, left_direction in enumerate(directions):
        for right_index in range(left_index + 1, len(directions)):
            if left_direction != directions[right_index]:
                witness = [
                    float(trials[left_index]["hidden_value_mm"]),
                    float(trials[right_index]["hidden_value_mm"]),
                ]
                break
        if witness is not None:
            break
    return {
        "status": "insufficient_information",
        "centroid_x_direction": None,
        "visible_conflicting_witness": witness,
    }


def rounded_equal(left: Any, right: Any, tolerance: float = 5e-4) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def solution_matches(record: Mapping[str, Any], solution: Mapping[str, Any]) -> bool:
    target = record["target"]
    if solution["status"] != target["status"]:
        return False
    answer = target["answer"]
    if record["task_type"] == "information_sufficiency":
        return solution["centroid_x_direction"] == answer["centroid_x_direction"]
    if solution["control_plan"] != answer["control_plan"]:
        return False
    return rounded_equal(solution["expected_residual_px"], answer["expected_residual_px"]) and rounded_equal(
        solution["best_achievable_residual_px"], answer["best_achievable_residual_px"]
    )


def pair_invariant(record: Mapping[str, Any]) -> dict[str, Any]:
    inputs = json.loads(json.dumps(record["prompt_inputs"], sort_keys=True))
    if record["task_type"] == "constrained_intervention":
        inputs.pop("target_observation")
        for trial in inputs["candidate_action_trials"]:
            trial.pop("measured_residual_px")
    else:
        inputs.pop("compatible_hidden_values_mm")
        inputs.pop("compatible_completion_trials")
    return inputs


def audit(dataset_dir: Path) -> dict[str, Any]:
    records = read_jsonl(dataset_dir / "canonical" / "probe.jsonl")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    tasks = Counter(record["task_type"] for record in records)
    statuses = Counter(record["target"]["status"] for record in records)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["provenance"]["match_group_id"])].append(record)

    expected_scenarios = int(manifest["scenario_count"])
    if len(records) != expected_scenarios * 4:
        failures.append("record count is not four per scenario")
    if tasks != Counter(
        {
            "constrained_intervention": expected_scenarios * 2,
            "information_sufficiency": expected_scenarios * 2,
        }
    ):
        failures.append(f"unexpected task balance: {dict(tasks)}")
    if statuses != Counter(
        {
            "feasible": expected_scenarios,
            "infeasible_within_limits": expected_scenarios,
            "answerable": expected_scenarios,
            "insufficient_information": expected_scenarios,
        }
    ):
        failures.append(f"unexpected status balance: {dict(statuses)}")

    solver_failures = []
    scaffold_measurement_failures = []
    forbidden = Counter()
    nonfinite = []
    for record in records:
        keys = forbidden_keys(record["prompt_inputs"])
        forbidden.update(keys)
        if not finite_tree(record):
            nonfinite.append(record["example_id"])
        solution = (
            control_solution(record)
            if record["task_type"] == "constrained_intervention"
            else sufficiency_solution(record)
        )
        if not solution_matches(record, solution):
            solver_failures.append({"example_id": record["example_id"], "solution": solution})
        inputs = record["prompt_inputs"]
        if record["task_type"] == "constrained_intervention":
            for trial in inputs["candidate_action_trials"]:
                expected = centroid_distance(
                    {
                        "centroid_x_px": trial["measured_centroid_x_px"],
                        "centroid_y_px": trial["measured_centroid_y_px"],
                    },
                    inputs["target_observation"],
                )
                if not rounded_equal(trial.get("measured_residual_px"), expected):
                    scaffold_measurement_failures.append(record["example_id"])
                    break
        else:
            current = float(inputs["current_observation"]["centroid_x_px"])
            threshold = float(inputs["direction_threshold_px"])
            for trial in inputs["compatible_completion_trials"]:
                expected = float(trial["measured_centroid_x_px"]) - current
                expected_direction = direction(expected, threshold)
                if not rounded_equal(trial.get("measured_delta_px"), expected) or (
                    trial.get("measured_direction") != expected_direction
                ):
                    scaffold_measurement_failures.append(record["example_id"])
                    break
    if forbidden:
        failures.append(f"forbidden derived input keys: {dict(forbidden)}")
    if nonfinite:
        failures.append(f"non-finite records: {nonfinite[:3]}")
    if solver_failures:
        failures.append(f"evidence solver mismatches: {solver_failures[:3]}")
    if scaffold_measurement_failures:
        failures.append(
            "scaffold measurements or directions disagree with raw centroids: "
            f"{scaffold_measurement_failures[:3]}"
        )

    pair_failures = []
    for group_id, pair in groups.items():
        if len(pair) != 2 or pair[0]["task_type"] != pair[1]["task_type"]:
            pair_failures.append(group_id)
            continue
        if pair_invariant(pair[0]) != pair_invariant(pair[1]):
            pair_failures.append(group_id)
    if pair_failures:
        failures.append(f"pair invariants differ: {pair_failures[:3]}")

    control_rows = [record for record in records if record["task_type"] == "constrained_intervention"]
    current_error_rule = best_threshold(
        [
            (
                centroid_distance(
                    record["prompt_inputs"]["current_observation"],
                    record["prompt_inputs"]["target_observation"],
                ),
                record["target"]["status"] == "feasible",
            )
            for record in control_rows
        ]
    )
    table_size = len(control_rows[0]["prompt_inputs"]["candidate_action_trials"])
    fixed_position_accuracies = []
    for position in range(table_size):
        correct = 0
        for record in control_rows:
            inputs = record["prompt_inputs"]
            trial = inputs["candidate_action_trials"][position]
            succeeds = (
                centroid_distance(
                    {
                        "centroid_x_px": trial["measured_centroid_x_px"],
                        "centroid_y_px": trial["measured_centroid_y_px"],
                    },
                    inputs["target_observation"],
                )
                <= float(inputs["actuator_constraints"]["success_tolerance_px"])
            )
            correct += succeeds == (record["target"]["status"] == "feasible")
        fixed_position_accuracies.append(correct / len(control_rows))

    sufficiency_rows = [
        record for record in records if record["task_type"] == "information_sufficiency"
    ]
    sorted_values_accuracy = best_binary_rule(
        sufficiency_rows,
        lambda record: list(record["prompt_inputs"]["compatible_hidden_values_mm"])
        == sorted(record["prompt_inputs"]["compatible_hidden_values_mm"]),
    )
    both_signs_accuracy = best_binary_rule(
        sufficiency_rows,
        lambda record: min(record["prompt_inputs"]["compatible_hidden_values_mm"])
        < 0
        < max(record["prompt_inputs"]["compatible_hidden_values_mm"]),
    )
    shortcut_metrics = {
        "control_current_error_best_threshold_accuracy": current_error_rule["accuracy"],
        "control_fixed_table_position_max_accuracy": max(fixed_position_accuracies),
        "control_fixed_table_position_accuracies": fixed_position_accuracies,
        "sufficiency_sorted_values_accuracy": sorted_values_accuracy,
        "sufficiency_both_signs_accuracy": both_signs_accuracy,
    }
    if float(current_error_rule["accuracy"]) > 0.65:
        failures.append("current-error threshold exceeds 0.65")
    if max(fixed_position_accuracies) > 0.65:
        failures.append("fixed table-position baseline exceeds 0.65")
    if float(sorted_values_accuracy) > 0.65 or float(both_signs_accuracy) > 0.65:
        failures.append("sufficiency list shortcut exceeds 0.65")

    return {
        "protocol": "evidence_grounded_v5_design_probe",
        "design_only": True,
        "future_evaluation_eligible": False,
        "record_count": len(records),
        "physical_scenario_count": len({record["group_id"] for record in records}),
        "minimal_pair_count": len(groups),
        "task_counts": dict(tasks),
        "status_counts": dict(statuses),
        "evidence_solver_match_count": len(records) - len(solver_failures),
        "evidence_solver_failure_count": len(solver_failures),
        "scaffold_measurement_failure_count": len(scaffold_measurement_failures),
        "pair_invariant_failure_count": len(pair_failures),
        "forbidden_input_key_counts": dict(forbidden),
        "nonfinite_count": len(nonfinite),
        "shortcut_metrics": shortcut_metrics,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.dataset_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
