#!/usr/bin/env python3
"""Audit the v7 mixed focus/preservation training curriculum."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .build_path_mapping_curriculum_v7 import TASKS, completion_target
from .core import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def image_placeholder_count(row: Mapping[str, Any]) -> int:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError(f"prompt is not prebuilt chat: {row.get('example_id')}")
    return sum(
        item.get("type") == "image"
        for message in prompt
        if isinstance(message, Mapping)
        for item in message.get("content", [])
        if isinstance(item, Mapping)
    )


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row["example_id"]) for row in rows]
    sources = Counter(str(row.get("curriculum_source")) for row in rows)
    anchor_tasks = Counter(
        str(row["task_type"])
        for row in rows
        if row.get("curriculum_source") == "seven_task_anchor"
    )
    focus_stages = Counter(
        str(row["task_type"])
        for row in rows
        if row.get("curriculum_source") == "path_mapping_v7"
    )
    visual_anchor_tasks = Counter(
        str(row["task_type"])
        for row in rows
        if row.get("curriculum_source") == "seven_task_anchor" and row.get("images")
    )
    completion_failures = []
    image_failures = []
    placeholder_failures = []
    for row in rows:
        try:
            completion_target(row)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            completion_failures.append({"example_id": row.get("example_id"), "error": str(exc)})
        images = list(row.get("images", []))
        missing = [image for image in images if not Path(image).is_file()]
        if missing:
            image_failures.append({"example_id": row.get("example_id"), "missing": missing})
        placeholders = image_placeholder_count(row)
        if placeholders != len(images):
            placeholder_failures.append(
                {
                    "example_id": row.get("example_id"),
                    "placeholders": placeholders,
                    "images": len(images),
                }
            )
    checks = {
        "record_count_620": len(rows) == 620,
        "unique_example_ids": len(ids) == len(set(ids)),
        "source_counts": sources == {"path_mapping_v7": 480, "seven_task_anchor": 140},
        "anchor_task_counts": anchor_tasks == {task: 20 for task in TASKS},
        "focus_stage_counts": focus_stages
        == {"tool_choice": 160, "tool_source_mapping": 160, "tool_result_interpretation": 160},
        "visual_anchor_counts": visual_anchor_tasks
        == {task: 2 for task in ("causal_effects", "forward_prediction", "diagnosis", "counterfactual_reasoning")},
        "all_completions_valid": not completion_failures,
        "all_images_present": not image_failures,
        "image_placeholders_match": not placeholder_failures,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "record_count": len(rows),
        "source_counts": dict(sorted(sources.items())),
        "anchor_task_counts": dict(sorted(anchor_tasks.items())),
        "focus_stage_counts": dict(sorted(focus_stages.items())),
        "visual_anchor_task_counts": dict(sorted(visual_anchor_tasks.items())),
        "completion_failures": completion_failures,
        "image_failures": image_failures,
        "placeholder_failures": placeholder_failures,
    }


def main() -> None:
    args = parse_args()
    report = audit(read_jsonl(args.curriculum_jsonl))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
