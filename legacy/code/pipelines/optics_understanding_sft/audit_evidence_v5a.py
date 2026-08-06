#!/usr/bin/env python3
"""Audit fresh v5A splits, visible evidence, shortcuts, and reference overlap."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .analyze_status_shortcuts import best_threshold
from .audit_evidence_v5_probe import (
    control_solution,
    direction,
    forbidden_keys,
    pair_invariant,
    rounded_equal,
    sufficiency_solution,
)
from .core import centroid_distance, read_jsonl, stable_json_hash


SPLITS = ("train", "dev", "confirmation")
EXPECTED_STATUSES = {
    "feasible",
    "infeasible_within_limits",
    "answerable",
    "insufficient_information",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, action="append", default=[])
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


def witness_valid(record: Mapping[str, Any]) -> bool:
    answer = record["target"]["answer"]
    witness = answer.get("visible_conflicting_witness")
    trials = record["prompt_inputs"]["compatible_completion_trials"]
    by_value = {
        round(float(trial["hidden_value_mm"]), 9): str(trial["measured_direction"])
        for trial in trials
    }
    if record["target"]["status"] == "answerable":
        return witness is None and len(set(by_value.values())) == 1
    if not isinstance(witness, list) or len(witness) != 2:
        return False
    left, right = (round(float(value), 9) for value in witness)
    return left in by_value and right in by_value and by_value[left] != by_value[right]


def solution_matches(record: Mapping[str, Any]) -> bool:
    solution = (
        control_solution(record)
        if record["task_type"] == "constrained_intervention"
        else sufficiency_solution(record)
    )
    target = record["target"]
    if solution["status"] != target["status"]:
        return False
    answer = target["answer"]
    if record["task_type"] == "information_sufficiency":
        return (
            solution["centroid_x_direction"] == answer["centroid_x_direction"]
            and witness_valid(record)
        )
    return (
        solution["control_plan"] == answer["control_plan"]
        and rounded_equal(solution["expected_residual_px"], answer["expected_residual_px"])
        and rounded_equal(
            solution["best_achievable_residual_px"], answer["best_achievable_residual_px"]
        )
    )


def scaffold_valid(record: Mapping[str, Any]) -> bool:
    inputs = record["prompt_inputs"]
    if record["task_type"] == "constrained_intervention":
        return all(
            rounded_equal(
                trial.get("measured_residual_px"),
                centroid_distance(
                    {
                        "centroid_x_px": trial["measured_centroid_x_px"],
                        "centroid_y_px": trial["measured_centroid_y_px"],
                    },
                    inputs["target_observation"],
                ),
            )
            for trial in inputs["candidate_action_trials"]
        )
    current = float(inputs["current_observation"]["centroid_x_px"])
    threshold = float(inputs["direction_threshold_px"])
    return all(
        rounded_equal(
            trial.get("measured_delta_px"),
            float(trial["measured_centroid_x_px"]) - current,
        )
        and trial.get("measured_direction")
        == direction(float(trial["measured_centroid_x_px"]) - current, threshold)
        for trial in inputs["compatible_completion_trials"]
    )


def shortcut_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    controls = [record for record in records if record["task_type"] == "constrained_intervention"]
    sufficiency = [record for record in records if record["task_type"] == "information_sufficiency"]
    current_error = best_threshold(
        [
            (
                centroid_distance(
                    record["prompt_inputs"]["current_observation"],
                    record["prompt_inputs"]["target_observation"],
                ),
                record["target"]["status"] == "feasible",
            )
            for record in controls
        ]
    )
    table_size = len(controls[0]["prompt_inputs"]["candidate_action_trials"])
    fixed_positions = []
    for position in range(table_size):
        correct = 0
        for record in controls:
            inputs = record["prompt_inputs"]
            residual = float(inputs["candidate_action_trials"][position]["measured_residual_px"])
            predicted = residual <= float(inputs["actuator_constraints"]["success_tolerance_px"])
            correct += predicted == (record["target"]["status"] == "feasible")
        fixed_positions.append(correct / len(controls))
    same_direction = sum(
        (len({trial["measured_direction"] for trial in record["prompt_inputs"]["compatible_completion_trials"]}) == 1)
        == (record["target"]["status"] == "answerable")
        for record in sufficiency
    ) / len(sufficiency)
    return {
        "control_current_error_threshold_accuracy": current_error["accuracy"],
        "control_fixed_position_max_accuracy": max(fixed_positions),
        "sufficiency_visible_evidence_rule_accuracy": same_direction,
    }


def reference_overlap(cases: list[dict[str, Any]], reference_dirs: list[Path]) -> dict[str, Any]:
    group_ids = {str(case["group_id"]) for case in cases}
    seeds = {int(case["scenario_seed"]) for case in cases}
    setups = {stable_json_hash(case["setup_config"]) for case in cases}
    result = {}
    for directory in reference_dirs:
        reference_cases = read_jsonl(directory / "master" / "cases.jsonl")
        result[directory.name] = {
            "group_id_overlap": len(group_ids & {str(case["group_id"]) for case in reference_cases}),
            "scenario_seed_overlap": len(seeds & {int(case["scenario_seed"]) for case in reference_cases}),
            "setup_hash_overlap": len(setups & {stable_json_hash(case["setup_config"]) for case in reference_cases}),
        }
    return result


def audit(dataset_dir: Path, reference_dirs: list[Path]) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    records = {
        "train": read_jsonl(dataset_dir / "canonical" / "train.jsonl"),
        "dev": read_jsonl(dataset_dir / "canonical" / "dev.jsonl"),
        "confirmation": read_jsonl(dataset_dir / "private" / "confirmation_records.jsonl"),
    }
    confirmation_prompts = read_jsonl(dataset_dir / "canonical" / "confirmation_prompts.jsonl")
    trial = read_jsonl(dataset_dir / "exports" / "qwen" / "train_trial50.jsonl")
    cases = read_jsonl(dataset_dir / "master" / "cases.jsonl")
    failures: list[str] = []

    for split in SPLITS:
        expected_scenarios = int(manifest["split_scenario_counts"][split])
        split_records = records[split]
        if len(split_records) != expected_scenarios * 4:
            failures.append(f"{split}: unexpected record count")
        statuses = Counter(record["target"]["status"] for record in split_records)
        if statuses != Counter({status: expected_scenarios for status in EXPECTED_STATUSES}):
            failures.append(f"{split}: status imbalance {dict(statuses)}")

    groups_by_split = {
        split: {str(record["group_id"]) for record in split_records}
        for split, split_records in records.items()
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if groups_by_split[left] & groups_by_split[right]:
                failures.append(f"scenario overlap between {left} and {right}")
    all_records = [record for split in SPLITS for record in records[split]]
    if len({record["example_id"] for record in all_records}) != len(all_records):
        failures.append("duplicate example IDs")
    if len(confirmation_prompts) != len(records["confirmation"]) or any(
        "target" in record for record in confirmation_prompts
    ):
        failures.append("confirmation prompts are missing or contain targets")

    solver_failures = [record["example_id"] for record in all_records if not solution_matches(record)]
    scaffold_failures = [record["example_id"] for record in all_records if not scaffold_valid(record)]
    forbidden = Counter(
        key for record in all_records for key in forbidden_keys(record["prompt_inputs"])
    )
    nonfinite = [record["example_id"] for record in all_records if not finite_tree(record)]
    if solver_failures:
        failures.append(f"visible-evidence solver failures: {solver_failures[:3]}")
    if scaffold_failures:
        failures.append(f"scaffold derivation failures: {scaffold_failures[:3]}")
    if forbidden:
        failures.append(f"forbidden prompt fields: {dict(forbidden)}")
    if nonfinite:
        failures.append(f"nonfinite records: {nonfinite[:3]}")

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        pairs[str(record["provenance"]["match_group_id"])].append(record)
    pair_failures = []
    for match_id, pair in pairs.items():
        if len(pair) != 2 or pair[0]["task_type"] != pair[1]["task_type"]:
            pair_failures.append(match_id)
        elif pair_invariant(pair[0]) != pair_invariant(pair[1]):
            pair_failures.append(match_id)
    if pair_failures:
        failures.append(f"minimal-pair invariant failures: {pair_failures[:3]}")

    trial_ids = [row["example_id"] for row in trial]
    train_ids = {record["example_id"] for record in records["train"]}
    if len(trial) != int(manifest["trial_train_record_count"]) or len(set(trial_ids)) != len(trial):
        failures.append("trial curriculum count or uniqueness failure")
    if not set(trial_ids) <= train_ids:
        failures.append("trial curriculum contains non-training records")
    trial_statuses = Counter(
        next(record["target"]["status"] for record in records["train"] if record["example_id"] == example_id)
        for example_id in trial_ids
    )
    if trial_statuses != Counter({status: len(trial) // 4 for status in EXPECTED_STATUSES}):
        failures.append(f"trial curriculum status imbalance: {dict(trial_statuses)}")

    shortcuts = {split: shortcut_metrics(records[split]) for split in SPLITS}
    for split, metrics in shortcuts.items():
        if metrics["control_current_error_threshold_accuracy"] > 0.65:
            failures.append(f"{split}: current-error shortcut too accurate")
        if metrics["control_fixed_position_max_accuracy"] > 0.65:
            failures.append(f"{split}: fixed-position shortcut too accurate")
        if metrics["sufficiency_visible_evidence_rule_accuracy"] != 1.0:
            failures.append(f"{split}: visible sufficiency evidence is not decisive")

    overlap = reference_overlap(cases, reference_dirs)
    if any(any(count for count in values.values()) for values in overlap.values()):
        failures.append(f"reference overlap: {overlap}")
    split_case_sets = {
        split: [case for case in cases if case["split"] == split] for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        left_seeds = {case["scenario_seed"] for case in split_case_sets[left]}
        left_setups = {stable_json_hash(case["setup_config"]) for case in split_case_sets[left]}
        for right in SPLITS[left_index + 1 :]:
            if left_seeds & {case["scenario_seed"] for case in split_case_sets[right]}:
                failures.append(f"scenario seed overlap between {left} and {right}")
            if left_setups & {stable_json_hash(case["setup_config"]) for case in split_case_sets[right]}:
                failures.append(f"setup overlap between {left} and {right}")

    return {
        "passed": not failures,
        "failures": failures,
        "dataset_version": manifest["version"],
        "record_count": len(all_records),
        "physical_scenario_count": len(cases),
        "minimal_pair_count": len(pairs),
        "split_record_counts": {split: len(value) for split, value in records.items()},
        "solver_failure_count": len(solver_failures),
        "scaffold_failure_count": len(scaffold_failures),
        "pair_invariant_failure_count": len(pair_failures),
        "forbidden_input_key_counts": dict(forbidden),
        "nonfinite_count": len(nonfinite),
        "confirmation_target_leak_count": sum("target" in record for record in confirmation_prompts),
        "trial_record_count": len(trial),
        "trial_status_counts": dict(trial_statuses),
        "shortcut_metrics": shortcuts,
        "reference_overlap": overlap,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.dataset_dir, args.reference_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
