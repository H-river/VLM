#!/usr/bin/env python3
"""Audit the compact-decision v7.1 mixed training curriculum."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .audit_path_mapping_curriculum_v7 import image_placeholder_count
from .build_path_mapping_curriculum_v7 import TASKS, completion_target
from .core import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.curriculum_jsonl)
    ids = [str(row["example_id"]) for row in rows]
    sources = Counter(str(row.get("curriculum_source")) for row in rows)
    anchors = Counter(
        str(row["task_type"])
        for row in rows
        if row.get("curriculum_source") == "seven_task_anchor"
    )
    focus = Counter(
        str(row["task_type"])
        for row in rows
        if row.get("curriculum_source") == "path_mapping_v7"
    )
    failures = []
    for row in rows:
        try:
            completion_target(row)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"completion {row.get('example_id')}: {exc}")
        images = list(row.get("images", []))
        if any(not Path(image).is_file() for image in images):
            failures.append(f"missing image: {row.get('example_id')}")
        if image_placeholder_count(row) != len(images):
            failures.append(f"image placeholder mismatch: {row.get('example_id')}")
    checks = {
        "record_count_540": len(rows) == 540,
        "unique_ids": len(ids) == len(set(ids)),
        "source_counts": sources == {"path_mapping_v7": 400, "seven_task_anchor": 140},
        "anchor_counts": anchors == {task: 20 for task in TASKS},
        "focus_counts": focus
        == {"compact_tool_result_interpretation": 320, "tool_source_mapping": 80},
        "row_validation": not failures,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": failures,
        "record_count": len(rows),
        "source_counts": dict(sorted(sources.items())),
        "anchor_task_counts": dict(sorted(anchors.items())),
        "focus_task_counts": dict(sorted(focus.items())),
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
