#!/usr/bin/env python3
"""Build a bounded mapping-only repair curriculum with general-task anchors."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import make_qwen_record, read_jsonl, stable_json_hash, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-curriculum", type=Path, required=True)
    parser.add_argument("--anchor-curriculum", type=Path, required=True)
    parser.add_argument("--dev-canonical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=55)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    unique_mapping: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.orchestration_curriculum):
        if (
            row.get("stage") == "tool_source_mapping"
            and row.get("source_task_type") == "visual_pair_direction_extraction"
        ):
            unique_mapping.setdefault(str(row["example_id"]), row)
    mapping_rows = list(unique_mapping.values())
    rng.shuffle(mapping_rows)
    mapping_rows = mapping_rows[:60]

    anchors_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.anchor_curriculum):
        if "stage" not in row:
            anchors_by_task[str(row["task_type"])].append(row)
    for rows in anchors_by_task.values():
        rng.shuffle(rows)
    task_order = sorted(anchors_by_task)
    anchors: list[dict[str, Any]] = []
    cursor = 0
    while len(anchors) < 20:
        task = task_order[cursor % len(task_order)]
        candidates = anchors_by_task[task]
        if candidates:
            anchors.append(candidates.pop())
        cursor += 1

    curriculum: list[dict[str, Any]] = []
    for step in range(20):
        curriculum.extend(mapping_rows[3 * step : 3 * step + 3])
        curriculum.append(anchors[step])
    write_jsonl(args.output_dir / "train_curriculum.jsonl", curriculum)

    dev = read_jsonl(args.dev_canonical)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev:
        by_source[str(row["source_example_id"])].append(row)
    source_ids = sorted(by_source)
    failure_source = "dev_v2_030007_forward_001__v10_pair__original"
    selected_sources = [failure_source] if failure_source in by_source else []
    state_sources = [
        source for source in source_ids if by_source[source][0]["source_task_type"] == "visual_state_classification"
    ]
    pair_sources = [
        source for source in source_ids if by_source[source][0]["source_task_type"] == "visual_pair_direction_extraction"
        and source != failure_source
    ]
    rng.shuffle(state_sources)
    rng.shuffle(pair_sources)
    selected_sources.extend(state_sources[:20])
    selected_sources.extend(pair_sources[:19])
    dev_panel = [row for source in selected_sources for row in by_source[source]]
    write_jsonl(
        args.output_dir / "dev_panel120.jsonl",
        (make_qwen_record(row, include_target=True) for row in dev_panel),
    )
    write_jsonl(args.output_dir / "dev_panel120_canonical.jsonl", dev_panel)

    manifest = {
        "dataset": "visual_mapping_repair_v10_10",
        "seed": args.seed,
        "optimizer_steps": 20,
        "microbatches_per_step": 4,
        "train_records": len(curriculum),
        "train_composition_per_step": {"pair_mapping": 3, "general_anchor": 1},
        "train_task_counts": dict(sorted(Counter(row["task_type"] for row in curriculum).items())),
        "anchor_task_counts": dict(sorted(Counter(row["task_type"] for row in anchors).items())),
        "dev_records": len(dev_panel),
        "dev_sources": len(selected_sources),
        "failure_source_included": failure_source in selected_sources,
        "train_hash": stable_json_hash(curriculum),
        "dev_hash": stable_json_hash(dev_panel),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
