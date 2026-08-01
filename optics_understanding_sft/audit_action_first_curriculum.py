#!/usr/bin/env python3
"""Audit the focused v3 curriculum before it is allowed into training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .build_corrective_curriculum import completion_target
from .core import read_jsonl


EXPECTED_ANCHORS = {
    "setup_interpretation": 32,
    "causal_effects": 32,
    "forward_prediction": 32,
    "diagnosis": 32,
    "counterfactual_reasoning": 32,
}


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
    dev_rows = read_jsonl(dev_jsonl)
    failures: list[str] = []
    source_counts = Counter(str(row.get("curriculum_source")) for row in rows)
    task_counts = Counter(str(row.get("task_type")) for row in rows)
    focused = [row for row in rows if row.get("curriculum_source") == "action_first_v3"]
    anchors = [row for row in rows if row.get("curriculum_source") == "preservation_anchor"]

    example_ids = [str(row["example_id"]) for row in rows]
    source_ids = [str(row.get("source_example_id")) for row in rows]
    if len(rows) != 800:
        failures.append(f"expected 800 curriculum rows, got {len(rows)}")
    if source_counts != Counter({"action_first_v3": 640, "preservation_anchor": 160}):
        failures.append(f"unexpected source counts: {dict(source_counts)}")
    if len(set(example_ids)) != len(example_ids):
        failures.append("curriculum example IDs are repeated")
    if len(set(source_ids)) != len(source_ids):
        failures.append("source examples are repeated")

    focused_tasks = Counter(str(row["task_type"]) for row in focused)
    if focused_tasks != Counter(
        {"constrained_intervention": 320, "information_sufficiency": 320}
    ):
        failures.append(f"unexpected focused task counts: {dict(focused_tasks)}")
    anchor_tasks = Counter(str(row["task_type"]) for row in anchors)
    if anchor_tasks != Counter(EXPECTED_ANCHORS):
        failures.append(f"unexpected preservation-anchor counts: {dict(anchor_tasks)}")

    dev_ids = {str(row["example_id"]) for row in dev_rows}
    overlap = sorted(set(source_ids) & dev_ids)
    if overlap:
        failures.append(f"curriculum source IDs overlap dev: {overlap[:5]}")

    action_order_failures = 0
    for row in focused:
        if row["task_type"] != "constrained_intervention":
            continue
        target = completion_target(row)
        if not isinstance(target.get("action"), dict):
            action_order_failures += 1
            continue
        text = completion_text(row)
        positions = [text.find(f'"{key}"') for key in ("action", "status", "answer")]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            action_order_failures += 1
    if action_order_failures:
        failures.append(f"{action_order_failures} control completions are not action-first")

    group_positions: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        if row.get("curriculum_source") == "action_first_v3" and row.get("match_group_id"):
            group_positions[str(row["match_group_id"])].append(position)
    noncontiguous = [
        group_id
        for group_id, positions in group_positions.items()
        if max(positions) - min(positions) + 1 != len(positions)
    ]
    group_sizes = Counter(len(positions) for positions in group_positions.values())
    if len(group_positions) != 280 or group_sizes != Counter({2: 200, 3: 80}):
        failures.append(
            f"unexpected focused match groups: count={len(group_positions)}, sizes={dict(group_sizes)}"
        )
    if noncontiguous:
        failures.append(f"{len(noncontiguous)} focused match groups are noncontiguous")

    return {
        "curriculum_jsonl": str(curriculum_jsonl.resolve()),
        "dev_jsonl": str(dev_jsonl.resolve()),
        "record_count": len(rows),
        "source_counts": dict(source_counts),
        "task_counts": dict(task_counts),
        "focused_task_counts": dict(focused_tasks),
        "anchor_task_counts": dict(anchor_tasks),
        "unique_example_count": len(set(example_ids)),
        "unique_source_example_count": len(set(source_ids)),
        "dev_example_overlap_count": len(overlap),
        "focused_match_group_count": len(group_positions),
        "focused_match_group_size_counts": dict(group_sizes),
        "noncontiguous_match_group_count": len(noncontiguous),
        "action_order_failure_count": action_order_failures,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.curriculum_jsonl, args.dev_jsonl)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
