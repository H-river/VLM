#!/usr/bin/env python3
"""Measure simple prompt-feature baselines for the two failed status tasks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from .core import centroid_distance, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def best_binary_rule(rows: list[dict[str, Any]], feature: Callable[[Mapping[str, Any]], bool]) -> float | None:
    if not rows:
        return None
    labels = [row["target"]["status"] == "answerable" for row in rows]
    values = [bool(feature(row)) for row in rows]
    return max(
        sum((value ^ flip) == label for value, label in zip(values, labels)) / len(rows)
        for flip in (False, True)
    )


def majority_by_bucket(
    rows: list[dict[str, Any]], bucket: Callable[[Mapping[str, Any]], str]
) -> float | None:
    if not rows:
        return None
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        groups[bucket(row)][str(row["target"]["status"])] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(rows)


def best_threshold(values: list[tuple[float, bool]]) -> dict[str, Any]:
    if not values:
        return {"accuracy": None, "threshold": None, "low_is_positive": None}
    unique = sorted({value for value, _ in values})
    cuts = [unique[0] - 1.0]
    cuts.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    cuts.append(unique[-1] + 1.0)
    candidates = []
    for cut in cuts:
        for low_is_positive in (True, False):
            accuracy = sum(
                ((value <= cut) if low_is_positive else (value > cut)) == label
                for value, label in values
            ) / len(values)
            candidates.append((accuracy, cut, low_is_positive))
    accuracy, threshold, low_is_positive = max(candidates)
    return {
        "accuracy": accuracy,
        "threshold": threshold,
        "low_is_positive": low_is_positive,
    }


def value_list(row: Mapping[str, Any]) -> list[float]:
    return [float(value) for value in row["prompt_inputs"]["compatible_hidden_values_mm"]]


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sufficiency = [row for row in rows if row["task_type"] == "information_sufficiency"]
    controls = [row for row in rows if row["task_type"] == "constrained_intervention"]
    control_errors = [
        (
            centroid_distance(
                row["prompt_inputs"]["current_observation"],
                row["prompt_inputs"]["target_observation"],
            ),
            row["target"]["status"] == "feasible",
        )
        for row in controls
    ]
    return {
        "record_count": len(rows),
        "status_counts": {
            task: dict(
                Counter(row["target"]["status"] for row in rows if row["task_type"] == task)
            )
            for task in ("information_sufficiency", "constrained_intervention")
        },
        "information_sufficiency": {
            "count": len(sufficiency),
            "sorted_values_accuracy": best_binary_rule(
                sufficiency, lambda row: value_list(row) == sorted(value_list(row))
            ),
            "first_value_is_minimum_accuracy": best_binary_rule(
                sufficiency, lambda row: value_list(row)[0] == min(value_list(row))
            ),
            "both_signs_present_accuracy": best_binary_rule(
                sufficiency, lambda row: min(value_list(row)) < 0 < max(value_list(row))
            ),
            "exact_rank_order_majority_accuracy": majority_by_bucket(
                sufficiency,
                lambda row: str(
                    tuple(
                        {value: index for index, value in enumerate(sorted(value_list(row)))}[value]
                        for value in value_list(row)
                    )
                ),
            ),
        },
        "constrained_intervention": {
            "count": len(controls),
            "target_error_threshold": best_threshold(control_errors),
            "active_actuator_majority_accuracy": majority_by_bucket(
                controls,
                lambda row: str(
                    row["prompt_inputs"]["actuator_constraints"]["active_actuator"]
                ),
            ),
        },
    }


def main() -> None:
    args = parse_args()
    result = analyze(read_jsonl(args.records_jsonl))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
