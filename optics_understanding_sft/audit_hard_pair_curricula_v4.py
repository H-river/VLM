#!/usr/bin/env python3
"""Audit v4 warm-up and mixed curricula before GPU training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_corrective_curriculum import completion_target
from .core import read_jsonl


DECISION_TASKS = {"constrained_intervention", "information_sufficiency"}
ANCHOR_TASKS = {
    "setup_interpretation",
    "causal_effects",
    "forward_prediction",
    "diagnosis",
    "counterfactual_reasoning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-jsonl", type=Path, required=True)
    parser.add_argument("--mixed-jsonl", type=Path, required=True)
    parser.add_argument("--diagnostic-jsonl", type=Path, required=True)
    parser.add_argument("--holdout-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def status_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    values: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        target = row.get("target")
        if not isinstance(target, dict):
            target = completion_target(row)
        values[row["task_type"]][target["status"]] += 1
    return {task: dict(counts) for task, counts in values.items()}


def physical_groups(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row["group_id"])
        for row in rows
        if row.get("curriculum_source") != "hard_v4_preservation_anchor"
    }


def audit(
    warmup_path: Path,
    mixed_path: Path,
    diagnostic_path: Path,
    holdout_path: Path,
    dev_path: Path,
) -> dict[str, Any]:
    warmup = read_jsonl(warmup_path)
    mixed = read_jsonl(mixed_path)
    diagnostic = read_jsonl(diagnostic_path)
    holdout = read_jsonl(holdout_path)
    dev = read_jsonl(dev_path)
    failures: list[str] = []

    warmup_tasks = Counter(row["task_type"] for row in warmup)
    mixed_tasks = Counter(row["task_type"] for row in mixed)
    warmup_statuses = status_counts(warmup)
    mixed_statuses = status_counts(mixed)
    if len(warmup) != 400 or warmup_tasks != Counter(
        {"constrained_intervention": 200, "information_sufficiency": 200}
    ):
        failures.append(f"unexpected warm-up counts: {len(warmup)}, {dict(warmup_tasks)}")
    for task, labels in {
        "constrained_intervention": {"feasible": 100, "infeasible_within_limits": 100},
        "information_sufficiency": {"answerable": 100, "insufficient_information": 100},
    }.items():
        if warmup_statuses.get(task) != labels:
            failures.append(f"unbalanced warm-up {task}: {warmup_statuses.get(task)}")

    expected_mixed = Counter(
        {
            "constrained_intervention": 100,
            "information_sufficiency": 100,
            **{task: 40 for task in ANCHOR_TASKS},
        }
    )
    if len(mixed) != 400 or mixed_tasks != expected_mixed:
        failures.append(f"unexpected mixed counts: {len(mixed)}, {dict(mixed_tasks)}")
    for task, labels in {
        "constrained_intervention": {"feasible": 50, "infeasible_within_limits": 50},
        "information_sufficiency": {"answerable": 50, "insufficient_information": 50},
    }.items():
        if mixed_statuses.get(task) != labels:
            failures.append(f"unbalanced mixed {task}: {mixed_statuses.get(task)}")

    warmup_groups = physical_groups(warmup)
    mixed_groups = physical_groups(mixed)
    diagnostic_groups = {str(row["group_id"]) for row in diagnostic}
    holdout_groups = {str(row["group_id"]) for row in holdout}
    group_sets = (warmup_groups, mixed_groups, diagnostic_groups, holdout_groups)
    cross_stage_overlap = set().union(
        *(left & right for index, left in enumerate(group_sets) for right in group_sets[index + 1 :])
    )
    if (
        len(warmup_groups) != 100
        or len(mixed_groups) != 50
        or len(diagnostic_groups) != 30
        or len(holdout_groups) != 60
        or cross_stage_overlap
    ):
        failures.append("hard scenarios are missing, repeated, or shared across training stages")
    split_reports = {}
    for name, rows, scenarios in (
        ("diagnostic", diagnostic, 30),
        ("holdout", holdout, 60),
    ):
        tasks = Counter(row["task_type"] for row in rows)
        statuses = status_counts(rows)
        split_reports[name] = {"tasks": dict(tasks), "statuses": statuses}
        if len(rows) != scenarios * 4 or tasks != Counter(
            {
                "constrained_intervention": scenarios * 2,
                "information_sufficiency": scenarios * 2,
            }
        ):
            failures.append(f"unexpected {name} counts: {len(rows)}, {dict(tasks)}")
        for task, labels in {
            "constrained_intervention": {
                "feasible": scenarios,
                "infeasible_within_limits": scenarios,
            },
            "information_sufficiency": {
                "answerable": scenarios,
                "insufficient_information": scenarios,
            },
        }.items():
            if statuses.get(task) != labels:
                failures.append(f"unbalanced {name} {task}: {statuses.get(task)}")
    for name, rows, expected_groups in (("warmup", warmup, 100), ("mixed", mixed, 50)):
        positions: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            if row.get("curriculum_source") != "hard_v4_preservation_anchor":
                positions[str(row["group_id"])].append(index)
        invalid = [
            group_id
            for group_id, offsets in positions.items()
            if len(offsets) != 4 or max(offsets) - min(offsets) >= (8 if name == "mixed" else 4)
        ]
        if len(positions) != expected_groups or invalid:
            failures.append(f"{name} has invalid grouped ordering: {invalid[:3]}")

    all_rows = warmup + mixed
    ids = [row["example_id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        failures.append("curriculum example IDs overlap")
    dev_ids = {row["example_id"] for row in dev}
    source_ids = {str(row.get("source_example_id", row["example_id"])) for row in all_rows}
    overlap = source_ids & dev_ids
    if overlap:
        failures.append(f"curriculum source IDs overlap dev: {sorted(overlap)[:3]}")

    mixed_sources = Counter(row.get("curriculum_source", "hard_pairs_v4") for row in mixed)
    visual_count = sum(bool(row.get("images")) for row in mixed)
    if mixed_sources != Counter({"hard_pairs_v4": 200, "hard_v4_preservation_anchor": 200}):
        failures.append(f"unexpected mixed sources: {dict(mixed_sources)}")
    if visual_count != 10:
        failures.append(f"expected ten mixed visual anchors, got {visual_count}")
    missing_images = [image for row in mixed for image in row.get("images", []) if not Path(image).exists()]
    if missing_images:
        failures.append(f"missing images: {missing_images[:3]}")

    return {
        "passed": not failures,
        "failures": failures,
        "warmup_record_count": len(warmup),
        "mixed_record_count": len(mixed),
        "diagnostic_record_count": len(diagnostic),
        "holdout_record_count": len(holdout),
        "warmup_task_counts": dict(warmup_tasks),
        "mixed_task_counts": dict(mixed_tasks),
        "warmup_status_counts": warmup_statuses,
        "mixed_status_counts": mixed_statuses,
        "diagnostic_status_counts": split_reports["diagnostic"]["statuses"],
        "holdout_status_counts": split_reports["holdout"]["statuses"],
        "warmup_physical_group_count": len(warmup_groups),
        "mixed_physical_group_count": len(mixed_groups),
        "diagnostic_physical_group_count": len(diagnostic_groups),
        "holdout_physical_group_count": len(holdout_groups),
        "cross_stage_group_overlap_count": len(cross_stage_overlap),
        "dev_overlap_count": len(overlap),
        "mixed_source_counts": dict(mixed_sources),
        "mixed_visual_count": visual_count,
    }


def main() -> None:
    args = parse_args()
    result = audit(
        args.warmup_jsonl,
        args.mixed_jsonl,
        args.diagnostic_jsonl,
        args.holdout_jsonl,
        args.dev_jsonl,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
