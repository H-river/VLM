#!/usr/bin/env python3
"""Compare paired evaluator runs with scenario-group bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import read_jsonl
from .evaluate import TASKS, finite_number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DETAILS_JSONL",
        help="Named evaluator details file; repeat for every paired run.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference",
        help="Run name used as the subtraction baseline. Defaults to the first --run.",
    )
    return parser.parse_args()


def parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --run {value!r}; expected NAME=DETAILS_JSONL")
        name, raw_path = value.split("=", 1)
        if not name or name in runs:
            raise ValueError(f"empty or duplicate run name: {name!r}")
        runs[name] = Path(raw_path)
    if len(runs) < 2:
        raise ValueError("at least two --run values are required")
    return runs


def task_macro(rows: list[Mapping[str, Any]]) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("task_score")
        if row.get("task_type") in TASKS and finite_number(score):
            by_task[str(row["task_type"])].append(float(score))
    missing = [task for task in TASKS if not by_task.get(task)]
    if missing:
        raise ValueError(f"cannot calculate equal-task macro; missing tasks: {missing}")
    return statistics.fmean(statistics.fmean(by_task[task]) for task in TASKS)


def indexed_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in seen:
            raise ValueError(f"duplicate example_id: {example_id}")
        seen.add(example_id)
        result[str(row["group_id"])].append(row)
    return dict(result)


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def compare(
    runs: Mapping[str, list[dict[str, Any]]],
    *,
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    indexed = {name: indexed_rows(rows) for name, rows in runs.items()}
    reference_groups = set(indexed[reference])
    reference_examples = {str(row["example_id"]) for row in runs[reference]}
    for name, groups in indexed.items():
        if set(groups) != reference_groups:
            raise ValueError(f"group mismatch between {reference!r} and {name!r}")
        examples = {str(row["example_id"]) for row in runs[name]}
        if examples != reference_examples:
            raise ValueError(f"example mismatch between {reference!r} and {name!r}")

    group_ids = sorted(reference_groups)
    point_scores = {name: task_macro(rows) for name, rows in runs.items()}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in runs if name != reference}
    for _ in range(replicates):
        selected = [rng.choice(group_ids) for _ in group_ids]
        ref_rows = [row for group_id in selected for row in indexed[reference][group_id]]
        ref_score = task_macro(ref_rows)
        for name in samples:
            candidate_rows = [row for group_id in selected for row in indexed[name][group_id]]
            samples[name].append(task_macro(candidate_rows) - ref_score)

    comparisons = {}
    for name, differences in samples.items():
        ordered = sorted(differences)
        comparisons[name] = {
            "candidate": name,
            "reference": reference,
            "point_difference": point_scores[name] - point_scores[reference],
            "bootstrap_mean_difference": statistics.fmean(differences),
            "ci95": [percentile(ordered, 0.025), percentile(ordered, 0.975)],
            "probability_candidate_better": sum(value > 0.0 for value in differences) / replicates,
            "probability_tie": sum(value == 0.0 for value in differences) / replicates,
        }
    return {
        "method": "paired scenario-group nonparametric bootstrap",
        "equal_task_macro": True,
        "seed": seed,
        "replicates": replicates,
        "scenario_groups": len(group_ids),
        "point_scores": point_scores,
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    paths = parse_runs(args.run)
    reference = args.reference or next(iter(paths))
    if reference not in paths:
        raise ValueError(f"unknown reference {reference!r}")
    runs = {name: read_jsonl(path) for name, path in paths.items()}
    result = compare(runs, reference=reference, replicates=args.replicates, seed=args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
