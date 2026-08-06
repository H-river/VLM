#!/usr/bin/env python3
"""Build a design-only evidence-grounded probe from already-used v4 train groups."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .build_hard_pairs_v4 import replace_contract
from .core import centroid_distance, read_jsonl, stable_json_hash, write_jsonl


PROBE_SUFFIX = "__evidence_v5_probe"
SUFFICIENCY_EVIDENCE_CONTRACT = {
    "status": "answerable | insufficient_information",
    "answer": {
        "centroid_x_direction": "increase | decrease | no_change | null",
        "missing_fields": ["field_name"],
        "nonidentifiable_output": "field_name | null",
        "visible_conflicting_witness": "two numeric hidden values | null",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl", type=Path, required=True)
    parser.add_argument("--master-jsonl", type=Path, required=True)
    parser.add_argument("--used-curriculum-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=15051)
    return parser.parse_args()


def contract_tail(prompt: str) -> str:
    marker = "\n\nReturn only strict JSON"
    prefix, separator, suffix = prompt.partition(marker)
    if not separator:
        raise ValueError("prompt is missing the output-contract tail")
    return separator + suffix


def grid_specs(private: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    specs = [spec for spec in private["replay_specs"] if str(spec["name"]).startswith("grid_")]
    return sorted(specs, key=lambda spec: int(str(spec["name"]).split("_")[1]))


def completion_specs(private: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    specs = [
        spec for spec in private["replay_specs"] if str(spec["name"]).startswith("completion_")
    ]
    return sorted(specs, key=lambda spec: int(str(spec["name"]).split("_")[1]))


def shuffled_indices(count: int, group_id: str, seed: int, namespace: str) -> list[int]:
    rng = random.Random(f"{seed}:{namespace}:{group_id}")
    indices = list(range(count))
    rng.shuffle(indices)
    return indices


def control_trials(
    record: Mapping[str, Any], private: Mapping[str, Any], permutation: list[int]
) -> list[dict[str, Any]]:
    specs = grid_specs(private)
    allowed = record["prompt_inputs"]["actuator_constraints"]["allowed_values_mm"]
    if len(specs) != len(allowed):
        raise ValueError("control evidence does not match the declared action grid")
    trials = []
    for index, spec in enumerate(specs):
        active = str(private["active_actuator"])
        if abs(float(spec["action"][active]) - float(allowed[index])) > 1e-9:
            raise ValueError("control replay order does not match allowed values")
        trials.append(
            {
                "action": copy.deepcopy(spec["action"]),
                "measured_centroid_x_px": float(spec["expected_state"]["centroid_x_px"]),
                "measured_centroid_y_px": float(spec["expected_state"]["centroid_y_px"]),
                "measured_residual_px": round(
                    centroid_distance(
                        spec["expected_state"], record["prompt_inputs"]["target_observation"]
                    ),
                    4,
                ),
            }
        )
    return [trials[index] for index in permutation]


def sufficiency_trials(record: Mapping[str, Any], private: Mapping[str, Any]) -> list[dict[str, Any]]:
    hidden = str(record["prompt_inputs"]["hidden_action_field"])
    specs = completion_specs(private)
    by_value: dict[float, Mapping[str, Any]] = {}
    for spec in specs:
        value = round(float(spec["action"][hidden]), 9)
        if value in by_value:
            raise ValueError("duplicate completion value")
        by_value[value] = spec
    trials = []
    for raw_value in record["prompt_inputs"]["compatible_hidden_values_mm"]:
        value = round(float(raw_value), 9)
        if value not in by_value:
            raise ValueError(f"missing completion evidence for {value}")
        delta = (
            float(by_value[value]["expected_state"]["centroid_x_px"])
            - float(record["prompt_inputs"]["current_observation"]["centroid_x_px"])
        )
        measured_direction = (
            "increase" if delta > 1.0 else "decrease" if delta < -1.0 else "no_change"
        )
        trials.append(
            {
                "hidden_value_mm": float(raw_value),
                "measured_centroid_x_px": float(
                    by_value[value]["expected_state"]["centroid_x_px"]
                ),
                "measured_delta_px": round(delta, 4),
                "measured_direction": measured_direction,
            }
        )
    return trials


def conflicting_witness(trials: list[Mapping[str, Any]]) -> list[float] | None:
    for left_index, left in enumerate(trials):
        for right in trials[left_index + 1 :]:
            if left["measured_direction"] != right["measured_direction"]:
                return [float(left["hidden_value_mm"]), float(right["hidden_value_mm"])]
    return None


def evidence_prompt(
    record: Mapping[str, Any],
    private: Mapping[str, Any],
    seed: int,
    *,
    include_witness: bool = False,
) -> str:
    inputs = copy.deepcopy(record["prompt_inputs"])
    task = str(record["task_type"])
    match_group_id = str(record["provenance"]["match_group_id"])
    if task == "constrained_intervention":
        permutation = shuffled_indices(len(grid_specs(private)), match_group_id, seed, "control")
        inputs["candidate_action_trials"] = control_trials(record, private, permutation)
        instruction = (
            "Use every row in the calibration table; do not infer outcomes from action sign. The "
            "measurement adapter supplies measured_residual_px, computed from the raw centroids. A row "
            "succeeds only when measured_residual_px is at most success_tolerance_px. If no row "
            "succeeds, return infeasible_within_limits, a "
            "null control_plan, and the smallest residual as best_achievable_residual_px. Otherwise "
            "return feasible and the successful action with the smallest absolute active-actuator "
            "motion; break an exact motion tie by smaller residual, then allowed_values_mm order. "
            "JSON numeric fields must be numbers, not quoted strings."
        )
    elif task == "information_sufficiency":
        inputs["direction_threshold_px"] = 1.0
        inputs["compatible_completion_trials"] = sufficiency_trials(record, private)
        instruction = (
            "Use every row in the compatible-completion table. The measurement adapter supplies "
            "measured_delta_px and its thresholded measured_direction. Return answerable only if every "
            "row has the same measured_direction. If any two rows have different measured_direction, "
            "return insufficient_information with centroid_x_direction null, lens_x_delta_mm in "
            "missing_fields, and centroid_x_direction as nonidentifiable_output. "
            + (
                "Also return two hidden_value_mm numbers whose measured_direction values differ in "
                "visible_conflicting_witness; use null when all directions agree. "
                if include_witness
                else ""
            )
            + "Use JSON null, not the string 'null'."
        )
    else:
        raise ValueError(f"unsupported probe task: {task}")
    return (
        instruction
        + "\n\nInput data:\n"
        + json.dumps(inputs, indent=2, sort_keys=True)
        + contract_tail(str(record["prompt"]))
    )


def control_pair_overrides(items: list[Mapping[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    feasible_item = next(
        item for item in items if item["record"]["target"]["status"] == "feasible"
    )
    infeasible_item = next(
        item
        for item in items
        if item["record"]["target"]["status"] == "infeasible_within_limits"
    )
    feasible_record = feasible_item["record"]
    private = feasible_item["private_eval"]
    specs = grid_specs(private)
    active = str(private["active_actuator"])
    current = feasible_record["prompt_inputs"]["current_observation"]
    infeasible_target = copy.deepcopy(
        infeasible_item["record"]["prompt_inputs"]["target_observation"]
    )
    infeasible_current_error = centroid_distance(current, infeasible_target)
    candidates = []
    for index, spec in enumerate(specs):
        value = float(spec["action"][active])
        if abs(value) <= 1e-9:
            continue
        state = spec["expected_state"]
        nearest_other = min(
            centroid_distance(state, other["expected_state"])
            for other_index, other in enumerate(specs)
            if other_index != index
        )
        current_error = centroid_distance(current, state)
        if nearest_other <= 0.05 or current_error <= 0.05:
            continue
        candidates.append(
            (
                abs(current_error - infeasible_current_error),
                -nearest_other,
                abs(value),
                index,
                nearest_other,
                current_error,
            )
        )
    if not candidates:
        raise ValueError("no unique nonzero action is available for the evidence probe")
    _, _, _, selected_index, nearest_other, current_error = min(candidates)
    tolerance = min(1.5, 0.4 * nearest_other, 0.45 * current_error)
    if tolerance <= 0.02:
        raise ValueError("evidence-probe tolerance is below sensor-relevant precision")
    selected_spec = specs[selected_index]
    target = copy.deepcopy(selected_spec["expected_state"])
    offset = 0.35 * tolerance
    sign = -1.0 if random.Random(f"{seed}:target-offset:{private['match_group_id']}").random() < 0.5 else 1.0
    target["centroid_x_px"] = round(float(target["centroid_x_px"]) + sign * offset, 4)
    selected_residual = centroid_distance(selected_spec["expected_state"], target)
    all_residuals = [centroid_distance(spec["expected_state"], target) for spec in specs]
    if sum(residual <= tolerance for residual in all_residuals) != 1:
        raise ValueError("designed feasible target is not uniquely reachable")
    infeasible_residuals = [
        centroid_distance(spec["expected_state"], infeasible_target) for spec in specs
    ]
    if min(infeasible_residuals) <= tolerance:
        raise ValueError("designed infeasible target became reachable")
    allowed = feasible_record["prompt_inputs"]["actuator_constraints"]["allowed_values_mm"]
    selected_action = copy.deepcopy(selected_spec["action"])
    if abs(float(selected_action[active]) - float(allowed[selected_index])) > 1e-9:
        raise ValueError("selected action does not match the declared grid")
    return {
        str(feasible_record["example_id"]): {
            "target_observation": target,
            "success_tolerance_px": round(tolerance, 4),
            "target": {
                "status": "feasible",
                "answer": {
                    "best_achievable_residual_px": None,
                    "control_plan": selected_action,
                    "expected_residual_px": round(selected_residual, 4),
                },
            },
        },
        str(infeasible_item["record"]["example_id"]): {
            "target_observation": infeasible_target,
            "success_tolerance_px": round(tolerance, 4),
            "target": {
                "status": "infeasible_within_limits",
                "answer": {
                    "best_achievable_residual_px": round(min(infeasible_residuals), 4),
                    "control_plan": None,
                    "expected_residual_px": None,
                },
            },
        },
    }


def convert_item(
    item: Mapping[str, Any],
    new_group_id: str,
    seed: int,
    control_override: Mapping[str, Any] | None = None,
    *,
    record_suffix: str = PROBE_SUFFIX,
    split: str = "design_probe",
    dataset_version: str = "evidence_grounded_v5_probe",
    design_only: bool = True,
    source_dataset: str = "hard_pairs_v4_used_train_groups",
    include_witness: bool = False,
) -> dict[str, Any]:
    private = copy.deepcopy(item["private_eval"])
    record = copy.deepcopy(item["record"])
    source_example_id = str(record["example_id"])
    if control_override is not None:
        record["prompt_inputs"]["target_observation"] = copy.deepcopy(
            control_override["target_observation"]
        )
        record["prompt_inputs"]["actuator_constraints"]["success_tolerance_px"] = float(
            control_override["success_tolerance_px"]
        )
        record["target"] = copy.deepcopy(control_override["target"])
    if include_witness and record["task_type"] == "information_sufficiency":
        trials = sufficiency_trials(record, private)
        record["target"]["answer"]["visible_conflicting_witness"] = conflicting_witness(trials)
        record["prompt"] = replace_contract(str(record["prompt"]), SUFFICIENCY_EVIDENCE_CONTRACT)
    record["example_id"] = source_example_id + record_suffix
    record["group_id"] = new_group_id
    record["split"] = split
    record["prompt"] = evidence_prompt(
        record, private, seed, include_witness=include_witness
    )
    record["prompt_inputs"] = json.loads(
        record["prompt"].split("\n\nInput data:\n", 1)[1].split("\n\nReturn only strict JSON", 1)[0]
    )
    record["provenance"] = {
        **record["provenance"],
        "dataset_version": dataset_version,
        "design_only": design_only,
        "source_example_id": source_example_id,
        "source_dataset": source_dataset,
        "evidence_grounded": True,
    }
    return {"private_eval": private, "record": record}


def selected_groups(curriculum_path: Path, count: int) -> list[str]:
    groups = []
    seen = set()
    for row in read_jsonl(curriculum_path):
        group_id = str(row["group_id"])
        if group_id not in seen:
            seen.add(group_id)
            groups.append(group_id)
        if len(groups) == count:
            return groups
    raise ValueError(f"requested {count} used groups but found {len(groups)}")


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical = {row["example_id"]: row for row in read_jsonl(args.canonical_jsonl)}
    master_cases = {str(case["group_id"]): case for case in read_jsonl(args.master_jsonl)}
    group_ids = selected_groups(args.used_curriculum_jsonl, args.scenarios)
    output_cases = []
    output_records = []
    for group_id in group_ids:
        if group_id not in master_cases:
            raise ValueError(f"missing master case: {group_id}")
        source_case = master_cases[group_id]
        new_group_id = group_id + PROBE_SUFFIX
        source_control_items = [
            item
            for item in source_case["records"]
            if item["record"]["task_type"] == "constrained_intervention"
        ]
        overrides = control_pair_overrides(source_control_items, args.seed)
        items = [
            convert_item(
                item,
                new_group_id,
                args.seed,
                overrides.get(str(item["record"]["example_id"])),
            )
            for item in source_case["records"]
        ]
        for item in items:
            source_id = item["record"]["provenance"]["source_example_id"]
            if source_id not in canonical:
                raise ValueError(f"missing canonical source record: {source_id}")
            output_records.append(copy.deepcopy(item["record"]))
        output_cases.append(
            {
                **{key: copy.deepcopy(value) for key, value in source_case.items() if key != "records"},
                "group_id": new_group_id,
                "split": "design_probe",
                "distribution": "used_train_design_probe",
                "records": items,
            }
        )
    return output_records, output_cases


def main() -> None:
    args = parse_args()
    if args.scenarios <= 0:
        raise ValueError("--scenarios must be positive")
    records, cases = build(args)
    canonical_path = args.output_dir / "canonical" / "probe.jsonl"
    master_path = args.output_dir / "master" / "cases.jsonl"
    write_jsonl(canonical_path, records)
    write_jsonl(master_path, cases)
    manifest = {
        "dataset_version": "evidence_grounded_v5_probe",
        "design_only": True,
        "future_evaluation_eligible": False,
        "source": "hard_pairs_v4 used warmup groups",
        "seed": args.seed,
        "scenario_count": len(cases),
        "record_count": len(records),
        "task_counts": dict(Counter(record["task_type"] for record in records)),
        "status_counts": dict(Counter(record["target"]["status"] for record in records)),
        "canonical_hash": stable_json_hash(records),
        "master_hash": stable_json_hash(cases),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
