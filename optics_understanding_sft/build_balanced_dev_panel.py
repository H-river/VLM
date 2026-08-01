#!/usr/bin/env python3
"""Build a deterministic, task/status/modality-balanced public development panel."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_path_mapping_curriculum_v7 import TASKS, VISUAL_TASKS
from .core import file_sha256, read_jsonl, stable_json_hash, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=10)
    parser.add_argument("--visual-per-supported-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    return parser.parse_args()


def target_status(row: dict[str, Any]) -> str:
    target = row.get("target")
    return str(target.get("status", "no_status")) if isinstance(target, dict) else "no_status"


def exclude_physical_groups(
    rows: list[dict[str, Any]], exclusion_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Remove every record sharing a physical group with any excluded record."""
    excluded_groups = {str(row["group_id"]) for row in exclusion_rows}
    return (
        [row for row in rows if str(row["group_id"]) not in excluded_groups],
        excluded_groups,
    )


def stratified_take(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[target_status(row)].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        values.sort(key=lambda row: str(row["example_id"]))
        rng.shuffle(values)
    result: list[dict[str, Any]] = []
    while len(result) < count:
        progressed = False
        for key in sorted(buckets):
            if buckets[key] and len(result) < count:
                result.append(buckets[key].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(result)} rows available for requested {count}")
    return result


def main() -> None:
    args = parse_args()
    source = read_jsonl(args.input_jsonl)
    exclusion_rows = [row for path in args.exclude_jsonl for row in read_jsonl(path)]
    source, excluded_groups = exclude_physical_groups(source, exclusion_rows)
    selected: list[dict[str, Any]] = []
    for task_index, task in enumerate(args.tasks):
        candidates = [row for row in source if row.get("task_type") == task]
        visual_count = args.visual_per_supported_task if task in VISUAL_TASKS else 0
        visual = [row for row in candidates if row.get("modality") == "visual"]
        text = [row for row in candidates if row.get("modality") == "text"]
        selected.extend(stratified_take(visual, visual_count, args.seed + 1000 + task_index))
        selected.extend(
            stratified_take(text, args.per_task - visual_count, args.seed + 2000 + task_index)
        )
    selected.sort(key=lambda row: (args.tasks.index(str(row["task_type"])), str(row["example_id"])))
    write_jsonl(args.output_jsonl, selected)
    counts = Counter()
    for row in selected:
        counts[f"task:{row['task_type']}"] += 1
        counts[f"status:{row['task_type']}:{target_status(row)}"] += 1
        counts[f"modality:{row['modality']}"] += 1
    manifest = {
        "name": "balanced_public_development_panel",
        "seed": args.seed,
        "tasks": args.tasks,
        "excluded_group_count": len(excluded_groups),
        "record_count": len(selected),
        "source_sha256": file_sha256(args.input_jsonl),
        "counts": dict(sorted(counts.items())),
        "example_ids_hash": stable_json_hash([row["example_id"] for row in selected]),
    }
    args.output_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
