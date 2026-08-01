#!/usr/bin/env python3
"""Audit action-first control and completion-evidence supervision."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import ACTION_KEYS, centroid_distance, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def completion_text(row: Mapping[str, Any]) -> str:
    return str(row["completion"][0]["content"][0]["text"])


def audit(dataset_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(dataset_dir / "canonical" / "train.jsonl")
    qwen = {row["example_id"]: row for row in read_jsonl(dataset_dir / "exports" / "qwen" / "train.jsonl")}
    masters = read_jsonl(dataset_dir / "master" / "cases.jsonl")
    private = {
        item["record"]["example_id"]: item["private_eval"]
        for master in masters
        for item in master["records"]
    }
    controls = [row for row in rows if row["task_type"] == "constrained_intervention"]
    sufficiency = [row for row in rows if row["task_type"] == "information_sufficiency"]
    failures: list[str] = []

    statuses = Counter(row["target"]["status"] for row in controls)
    modes = Counter(
        row["provenance"].get("control_action_mode")
        for row in controls
        if row["target"]["status"] == "feasible"
    )
    signs = Counter(
        row["provenance"].get("control_action_sign")
        for row in controls
        if row["target"]["status"] == "feasible"
    )
    per_actuator_sign: dict[str, Counter[str]] = defaultdict(Counter)
    movement_ratios: dict[str, list[float]] = defaultdict(list)
    for row in controls:
        target = row["target"]
        action = target.get("action")
        if not isinstance(action, Mapping):
            failures.append(f"missing action-first block: {row['example_id']}")
            continue
        actuator = str(row["prompt_inputs"]["actuator_constraints"]["active_actuator"])
        allowed = [abs(float(value)) for value in row["prompt_inputs"]["actuator_constraints"]["allowed_values_mm"]]
        bound = max(allowed)
        movement = float(action.get("signed_movement_mm", float("nan")))
        residual = float(action.get("predicted_residual_px", float("nan")))
        executable = action.get("executable_valid")
        if action.get("actuator") != actuator or not math.isfinite(movement) or not math.isfinite(residual):
            failures.append(f"invalid action-first values: {row['example_id']}")
        if executable is not (target["status"] == "feasible"):
            failures.append(f"executable/status mismatch: {row['example_id']}")
        expected_action = private[row["example_id"]]["canonical_action"]
        if abs(movement - float(expected_action[actuator])) > 1e-6:
            failures.append(f"action-first movement disagrees with simulator optimum: {row['example_id']}")
        text = completion_text(qwen[row["example_id"]])
        positions = [text.find(f'"{key}"') for key in ("action", "status", "answer")]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            failures.append(f"completion is not serialized action-first: {row['example_id']}")
        if target["status"] == "feasible":
            sign = "positive" if movement > 0 else "negative" if movement < 0 else "zero"
            per_actuator_sign[actuator][sign] += 1
            movement_ratios[str(row["provenance"].get("control_action_mode"))].append(
                abs(movement) / bound
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in controls:
        groups[str(row["provenance"].get("match_group_id"))].append(row)
    pair_error_ranges: list[float] = []
    pair_kinds: Counter[str] = Counter()
    infeasible_matched = 0
    for match_id, group_rows in groups.items():
        kind = str(group_rows[0]["provenance"].get("match_group_kind"))
        pair_kinds[kind] += 1
        feasible = [row for row in group_rows if row["target"]["status"] == "feasible"]
        infeasible_matched += sum(row["target"]["status"] == "infeasible_within_limits" for row in group_rows)
        if len(feasible) != 2:
            failures.append(f"control action group lacks two feasible records: {match_id}")
            continue
        errors = [
            centroid_distance(row["prompt_inputs"]["current_observation"], row["prompt_inputs"]["target_observation"])
            for row in feasible
        ]
        pair_error_ranges.append(max(errors) - min(errors))

    suff_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    insufficient_evidence = 0
    invariant_evidence = 0
    for row in sufficiency:
        suff_groups[str(row["provenance"].get("match_group_id"))].append(row)
        answer = row["target"]["answer"]
        compatible = answer.get("compatible_completions")
        changing = answer.get("answer_changing_completions")
        if not isinstance(compatible, list) or len(compatible) != 5:
            failures.append(f"missing five completion witnesses: {row['example_id']}")
            continue
        outputs = {item.get("centroid_x_direction") for item in compatible if isinstance(item, Mapping)}
        if row["target"]["status"] == "answerable":
            invariant_evidence += len(outputs) == 1 and changing == []
            if len(outputs) != 1 or changing != []:
                failures.append(f"answerable completion evidence is not invariant: {row['example_id']}")
        else:
            changing_outputs = {
                item.get("centroid_x_direction") for item in changing if isinstance(item, Mapping)
            } if isinstance(changing, list) else set()
            insufficient_evidence += len(changing_outputs) >= 2
            if len(changing_outputs) < 2:
                failures.append(f"insufficient record lacks answer-changing completions: {row['example_id']}")
    for match_id, group_rows in suff_groups.items():
        if len(group_rows) != 2 or {row["target"]["status"] for row in group_rows} != {
            "answerable",
            "insufficient_information",
        }:
            failures.append(f"invalid sufficiency contrast pair: {match_id}")
            continue
        hidden_fields = {private[row["example_id"]].get("hidden_field") for row in group_rows}
        if len(hidden_fields) != 1:
            failures.append(f"sufficiency pair changes masked field: {match_id}")

    pair_mean = statistics.fmean(pair_error_ranges) if pair_error_ranges else None
    pair_p95 = percentile(pair_error_ranges, 0.95)
    if statuses != Counter({"feasible": 240, "infeasible_within_limits": 80}):
        failures.append(f"unexpected control status counts: {dict(statuses)}")
    if modes != Counter({"minimum_motion": 120, "near_boundary": 120}):
        failures.append(f"unexpected feasible mode counts: {dict(modes)}")
    if signs != Counter({"positive": 120, "negative": 120}):
        failures.append(f"unexpected feasible sign counts: {dict(signs)}")
    if pair_kinds != Counter({"action_diversity_triplet": 80, "action_diversity_pair": 40}):
        failures.append(f"unexpected action group counts: {dict(pair_kinds)}")
    if infeasible_matched != 80:
        failures.append(f"not every infeasible record is matched: {infeasible_matched}/80")
    if pair_mean is None or pair_mean > 1.5 or pair_p95 is None or pair_p95 > 3.0:
        failures.append(f"feasible action-pair error mismatch too large: mean={pair_mean}, p95={pair_p95}")
    if movement_ratios["minimum_motion"] and max(movement_ratios["minimum_motion"]) > 0.35 + 1e-9:
        failures.append("minimum-motion examples exceed 35% of their action bound")
    if movement_ratios["near_boundary"] and min(movement_ratios["near_boundary"]) < 0.50 - 1e-9:
        failures.append("near-boundary examples fall below 50% of their action bound")
    if len(suff_groups) != 160 or invariant_evidence != 160 or insufficient_evidence != 160:
        failures.append(
            f"incomplete sufficiency evidence coverage: groups={len(suff_groups)}, "
            f"invariant={invariant_evidence}, changing={insufficient_evidence}"
        )

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "record_count": len(rows),
        "scenario_sampling": {
            "scenario_count": len(masters),
            "resampled_scenario_count": sum(
                int(master["records"][0]["record"]["provenance"].get("scenario_sampling_attempt", 0)) > 0
                for master in masters
            ),
            "maximum_attempt": max(
                (
                    int(master["records"][0]["record"]["provenance"].get("scenario_sampling_attempt", 0))
                    for master in masters
                ),
                default=0,
            ),
        },
        "control": {
            "status_counts": dict(statuses),
            "feasible_mode_counts": dict(modes),
            "feasible_sign_counts": dict(signs),
            "per_actuator_sign_counts": {
                key: dict(value) for key, value in sorted(per_actuator_sign.items())
            },
            "match_group_counts": dict(pair_kinds),
            "infeasible_matched": infeasible_matched,
            "action_pair_error_range_px_mean": pair_mean,
            "action_pair_error_range_px_p95": pair_p95,
            "minimum_motion_ratio_max": max(movement_ratios["minimum_motion"], default=None),
            "near_boundary_ratio_min": min(movement_ratios["near_boundary"], default=None),
        },
        "sufficiency": {
            "pair_count": len(suff_groups),
            "invariant_evidence_count": invariant_evidence,
            "answer_changing_evidence_count": insufficient_evidence,
        },
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
