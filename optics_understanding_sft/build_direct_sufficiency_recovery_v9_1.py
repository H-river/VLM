#!/usr/bin/env python3
"""Build a balanced direct-sufficiency recovery curriculum with tool refresh."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .build_counterfactual_tool_curriculum_v9 import select_groups
from .build_forward_tool_curriculum_v8 import prepare
from .build_path_mapping_curriculum_v7 import select_anchors
from .core import read_jsonl, stable_json_hash, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-jsonl", type=Path, required=True)
    parser.add_argument("--counterfactual-refresh-jsonl", type=Path, required=True)
    parser.add_argument("--forward-refresh-jsonl", type=Path, required=True)
    parser.add_argument("--compact-refresh-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--per-sufficiency-status", type=int, default=80)
    parser.add_argument("--seed", type=int, default=46)
    return parser.parse_args()


def completion_status(row: dict) -> str:
    text = row["completion"][0]["content"][0]["text"]
    return str(json.loads(text)["status"])


def main() -> None:
    args = parse_args()
    direct = read_jsonl(args.direct_jsonl)
    suff_by_status: dict[str, list[dict]] = defaultdict(list)
    for row in direct:
        if row["task_type"] == "information_sufficiency":
            suff_by_status[completion_status(row)].append(row)
    expected_statuses = {"answerable", "insufficient_information"}
    if set(suff_by_status) != expected_statuses:
        raise ValueError(f"unexpected sufficiency statuses: {sorted(suff_by_status)}")
    suff = [
        row
        for status in sorted(expected_statuses)
        for row in sorted(suff_by_status[status], key=lambda item: str(item["example_id"]))[: args.per_sufficiency_status]
    ]
    counterfactual, cf_groups = select_groups(read_jsonl(args.counterfactual_refresh_jsonl), 10)
    forward, forward_groups = select_groups(read_jsonl(args.forward_refresh_jsonl), 10)
    compact, compact_groups = select_groups(read_jsonl(args.compact_refresh_jsonl), 5)
    anchors = [row for row in select_anchors(direct, 20, 2, args.seed) if row["task_type"] != "information_sufficiency"]
    prepared = [prepare(row, "direct_sufficiency_recovery", args.image_root) for row in suff]
    prepared += [prepare(row, "counterfactual_tool_v9_refresh") for row in counterfactual]
    prepared += [prepare(row, "forward_tool_v8_refresh") for row in forward]
    prepared += [prepare(row, "compact_v7_1_refresh") for row in compact]
    prepared += [prepare(row, "other_task_anchor", args.image_root) for row in anchors]
    units: dict[str, list[dict]] = defaultdict(list)
    for row in prepared:
        source = str(row["curriculum_source"])
        unit = str(row["example_id"]) if source in {"direct_sufficiency_recovery", "other_task_anchor"} else str(row["group_id"])
        units[f"{source}:{unit}"].append(row)
    ordered = sorted(units.items())
    random.Random(args.seed).shuffle(ordered)
    rows = [row for _, values in ordered for row in sorted(values, key=lambda item: str(item["example_id"]))]
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("curriculum example IDs are not unique")
    write_jsonl(args.output_jsonl, rows)
    counts = Counter()
    for row in rows:
        counts[f"source:{row['curriculum_source']}"] += 1
        counts[f"task:{row['task_type']}"] += 1
        counts[f"modality:{'visual' if row['images'] else 'text'}"] += 1
        if row["curriculum_source"] == "direct_sufficiency_recovery":
            counts[f"suff_status:{completion_status(row)}"] += 1
    manifest = {
        "name": "direct_sufficiency_recovery_v9_1",
        "seed": args.seed,
        "record_count": len(rows),
        "sampling_unit_count": len(units),
        "maximum_sampling_unit_size": max(map(len, units.values())),
        "counterfactual_refresh_group_count": len(cf_groups),
        "forward_refresh_group_count": len(forward_groups),
        "compact_refresh_group_count": len(compact_groups),
        "counts": dict(sorted(counts.items())),
        "ordered_example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
    }
    args.output_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
