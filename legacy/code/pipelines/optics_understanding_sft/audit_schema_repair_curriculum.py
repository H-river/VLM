#!/usr/bin/env python3
"""Audit the compact schema-compatible action-first repair curriculum."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .build_corrective_curriculum import completion_target
from .core import read_jsonl


ACTION_EVIDENCE_FIELDS = (
    "actuator",
    "signed_movement_mm",
    "predicted_residual_px",
    "executable_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def completion_text(row: Mapping[str, Any]) -> str:
    return str(row["completion"][0]["content"][0]["text"])


def audit(curriculum_jsonl: Path, dev_jsonl: Path) -> dict[str, Any]:
    rows = read_jsonl(curriculum_jsonl)
    dev_ids = {row["example_id"] for row in read_jsonl(dev_jsonl)}
    failures: list[str] = []
    task_counts = Counter(row["task_type"] for row in rows)
    source_counts = Counter(row["curriculum_source"] for row in rows)
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_positions: dict[str, list[int]] = defaultdict(list)
    actuator_counts: Counter[str] = Counter()
    action_order_failures = 0
    sufficiency_evidence_failures = 0

    for index, row in enumerate(rows):
        target = completion_target(row)
        status_counts[row["task_type"]][target["status"]] += 1
        if row.get("match_group_id"):
            group_positions[str(row["match_group_id"])].append(index)
        if row["task_type"] == "constrained_intervention":
            answer = target.get("answer")
            if not isinstance(answer, dict) or any(field not in answer for field in ACTION_EVIDENCE_FIELDS):
                action_order_failures += 1
                continue
            actuator_counts[str(answer["actuator"])] += 1
            text = completion_text(row)
            positions = [text.find(f'"{field}"') for field in (*ACTION_EVIDENCE_FIELDS, "status")]
            if any(position < 0 for position in positions) or positions != sorted(positions):
                action_order_failures += 1
            prompt_text = str(row["prompt"][0]["content"][-1]["text"])
            contract = prompt_text.split("Task output contract", 1)[-1]
            if '\n  "action": {' in contract or not all(field in contract for field in ACTION_EVIDENCE_FIELDS):
                action_order_failures += 1
        elif row["task_type"] == "information_sufficiency":
            answer = target.get("answer")
            compatible = answer.get("compatible_completions") if isinstance(answer, dict) else None
            changing = answer.get("answer_changing_completions") if isinstance(answer, dict) else None
            text = completion_text(row)
            if text.find('"answer"') < 0 or text.find('"answer"') > text.find('"status"'):
                sufficiency_evidence_failures += 1
            elif not isinstance(compatible, list) or len(compatible) != 5:
                sufficiency_evidence_failures += 1
            elif target["status"] == "insufficient_information" and (
                not isinstance(changing, list)
                or len({item.get("centroid_x_direction") for item in changing}) < 2
            ):
                sufficiency_evidence_failures += 1

    expected_tasks = Counter(
        {
            "constrained_intervention": 100,
            "information_sufficiency": 60,
            "setup_interpretation": 8,
            "causal_effects": 8,
            "forward_prediction": 8,
            "diagnosis": 8,
            "counterfactual_reasoning": 8,
        }
    )
    if len(rows) != 200 or task_counts != expected_tasks:
        failures.append(f"unexpected record/task counts: {len(rows)}, {dict(task_counts)}")
    if source_counts != Counter({"schema_repair_v3_1": 160, "schema_preservation_anchor": 40}):
        failures.append(f"unexpected source counts: {dict(source_counts)}")
    source_ids = [str(row["source_example_id"]) for row in rows]
    if len(set(source_ids)) != 200:
        failures.append("source examples are not unique")
    overlap = sorted(set(source_ids) & dev_ids)
    if overlap:
        failures.append(f"source examples overlap dev: {overlap[:5]}")
    if Counter(status_counts["constrained_intervention"]) != Counter(
        {"feasible": 80, "infeasible_within_limits": 20}
    ):
        failures.append(f"unexpected control statuses: {dict(status_counts['constrained_intervention'])}")
    if Counter(status_counts["information_sufficiency"]) != Counter(
        {"answerable": 30, "insufficient_information": 30}
    ):
        failures.append(f"unexpected sufficiency statuses: {dict(status_counts['information_sufficiency'])}")
    if actuator_counts != Counter({"lens_x_delta_mm": 50, "lens_y_delta_mm": 50}):
        failures.append(f"unexpected actuator counts: {dict(actuator_counts)}")
    if action_order_failures:
        failures.append(f"{action_order_failures} control rows violate compatible action-first ordering")
    if sufficiency_evidence_failures:
        failures.append(f"{sufficiency_evidence_failures} sufficiency rows violate evidence ordering")

    group_sizes = Counter(len(positions) for positions in group_positions.values())
    noncontiguous = [
        group_id
        for group_id, positions in group_positions.items()
        if max(positions) - min(positions) + 1 != len(positions)
    ]
    if group_sizes != Counter({2: 50, 3: 20}):
        failures.append(f"unexpected match-group sizes: {dict(group_sizes)}")
    if noncontiguous:
        failures.append(f"{len(noncontiguous)} match groups are noncontiguous")
    first_half = Counter(row["task_type"] for row in rows[:100])
    second_half = Counter(row["task_type"] for row in rows[100:])
    if first_half != second_half:
        failures.append("the two 100-row curriculum halves are not task-identical")
    missing_images = [image for row in rows for image in row.get("images", []) if not Path(image).exists()]
    if missing_images:
        failures.append(f"missing curriculum images: {missing_images[:3]}")

    return {
        "curriculum_jsonl": str(curriculum_jsonl.resolve()),
        "record_count": len(rows),
        "task_counts": dict(task_counts),
        "source_counts": dict(source_counts),
        "status_counts": {task: dict(values) for task, values in status_counts.items()},
        "actuator_counts": dict(actuator_counts),
        "match_group_size_counts": dict(group_sizes),
        "noncontiguous_match_group_count": len(noncontiguous),
        "action_order_failure_count": action_order_failures,
        "sufficiency_evidence_failure_count": sufficiency_evidence_failures,
        "dev_overlap_count": len(overlap),
        "visual_record_count": sum(bool(row.get("images")) for row in rows),
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.curriculum_jsonl, args.dev_jsonl)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
