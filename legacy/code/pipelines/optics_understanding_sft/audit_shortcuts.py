#!/usr/bin/env python3
"""Audit label shortcuts, match quality, and a fixed-gain control baseline."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from .core import ACTION_KEYS, centroid_distance, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def majority_accuracy(rows: list[dict[str, Any]], bucket: Callable[[Mapping[str, Any]], str]) -> float | None:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[bucket(row)][str(row["target"]["status"])] += 1
    total = sum(sum(values.values()) for values in grouped.values())
    return sum(max(values.values()) for values in grouped.values()) / total if total else None


def diagnosis_axis(row: Mapping[str, Any]) -> str:
    candidates = row["prompt_inputs"]["candidate_interventions"]
    action = candidates[0]["action"]
    return "x" if abs(float(action["lens_x_delta_mm"])) > 0 else "y"


def private_index(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        item["record"]["example_id"]: item["private_eval"]
        for master in read_jsonl(dataset_dir / "master/cases.jsonl")
        for item in master["records"]
    }


def fixed_gain(rows: list[dict[str, Any]], private: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    successes = 0
    total = 0
    for row in rows:
        if row["task_type"] != "constrained_intervention":
            continue
        inputs = row["prompt_inputs"]
        constraints = inputs["actuator_constraints"]
        actuator = constraints["active_actuator"]
        state_axis = "centroid_x_px" if "_x_" in actuator else "centroid_y_px"
        error = float(inputs["current_observation"][state_axis]) - float(inputs["target_observation"][state_axis])
        allowed = [float(value) for value in constraints["allowed_values_mm"]]
        proposed = -0.002 * error
        selected_index = min(range(len(allowed)), key=lambda index: abs(allowed[index] - proposed))
        spec = next(
            spec
            for spec in private[row["example_id"]]["replay_specs"]
            if spec["name"] == f"grid_{selected_index}"
        )
        residual = centroid_distance(spec["expected_state"], inputs["target_observation"])
        successes += residual <= float(constraints["success_tolerance_px"])
        total += 1
    return {"count": total, "successes": successes, "success_rate": successes / total if total else None}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(p * len(ordered)) - 1))]


def match_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        match_id = row.get("provenance", {}).get("match_group_id")
        if match_id:
            groups[str(match_id)].append(row)
    error_ranges: list[float] = []
    focal_ranges: list[float] = []
    for group_rows in groups.values():
        if group_rows[0]["task_type"] != "constrained_intervention":
            continue
        errors = [
            centroid_distance(row["prompt_inputs"]["current_observation"], row["prompt_inputs"]["target_observation"])
            for row in group_rows
        ]
        focals = [float(row["prompt_inputs"]["setup"]["lens_focal_length_mm"]) for row in group_rows]
        error_ranges.append(max(errors) - min(errors))
        focal_ranges.append(max(focals) - min(focals))
    return {
        "match_group_count": len(groups),
        "control_pair_count": len(error_ranges),
        "control_error_range_px_mean": statistics.fmean(error_ranges) if error_ranges else None,
        "control_error_range_px_p95": percentile(error_ranges, 0.95),
        "control_focal_range_mm_mean": statistics.fmean(focal_ranges) if focal_ranges else None,
    }


def main() -> None:
    args = parse_args()
    canonical_name = "train.jsonl" if args.split == "train" else "val.jsonl"
    rows = read_jsonl(args.dataset_dir / "canonical" / canonical_name)
    private = private_index(args.dataset_dir)
    status_tasks = ("information_sufficiency", "diagnosis", "constrained_intervention")
    status_counts = {
        task: dict(sorted(Counter(row["target"]["status"] for row in rows if row["task_type"] == task).items()))
        for task in status_tasks
    }
    task_rows = {task: [row for row in rows if row["task_type"] == task] for task in status_tasks}
    result = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split": args.split,
        "record_count": len(rows),
        "status_counts": status_counts,
        "constant_status_accuracy": {
            task: majority_accuracy(task_rows[task], lambda _: "all") for task in status_tasks
        },
        "single_nuisance_accuracy": {
            "control_active_actuator": majority_accuracy(
                task_rows["constrained_intervention"],
                lambda row: str(row["prompt_inputs"]["actuator_constraints"]["active_actuator"]),
            ),
            "diagnosis_axis": majority_accuracy(task_rows["diagnosis"], diagnosis_axis),
            "sufficiency_hidden_field": majority_accuracy(
                task_rows["information_sufficiency"],
                lambda row: str(private[row["example_id"]].get("hidden_field")),
            ),
        },
        "fixed_gain_control": fixed_gain(rows, private),
        "match_quality": match_quality(rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
