#!/usr/bin/env python3
"""Mix counterfactual focus, forward/compact refresh, and seven-task anchors."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .build_forward_tool_curriculum_v8 import prepare
from .build_path_mapping_curriculum_v7 import select_anchors
from .core import file_sha256, read_jsonl, stable_json_hash, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-jsonl", type=Path, required=True)
    parser.add_argument("--forward-refresh-jsonl", type=Path, required=True)
    parser.add_argument("--compact-refresh-jsonl", type=Path, required=True)
    parser.add_argument("--anchor-jsonl", type=Path, required=True)
    parser.add_argument("--anchor-image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--forward-refresh-groups", type=int, default=20)
    parser.add_argument("--compact-refresh-groups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=45)
    return parser.parse_args()


def select_groups(rows: list[dict], count: int) -> tuple[list[dict], list[str]]:
    groups = sorted({str(row["group_id"]) for row in rows})[:count]
    selected = set(groups)
    return [row for row in rows if str(row["group_id"]) in selected], groups


def main() -> None:
    args = parse_args()
    counterfactual = read_jsonl(args.counterfactual_jsonl)
    forward, forward_groups = select_groups(read_jsonl(args.forward_refresh_jsonl), args.forward_refresh_groups)
    compact, compact_groups = select_groups(read_jsonl(args.compact_refresh_jsonl), args.compact_refresh_groups)
    anchors = select_anchors(read_jsonl(args.anchor_jsonl), 20, 2, args.seed)
    prepared = [prepare(row, "counterfactual_tool_v9") for row in counterfactual]
    prepared += [prepare(row, "forward_tool_v8_refresh") for row in forward]
    prepared += [prepare(row, "compact_v7_1_refresh") for row in compact]
    prepared += [prepare(row, "seven_task_anchor", args.anchor_image_root) for row in anchors]

    units: dict[str, list[dict]] = defaultdict(list)
    for row in prepared:
        source = str(row["curriculum_source"])
        unit = str(row["example_id"]) if source == "seven_task_anchor" else str(row["group_id"])
        units[f"{source}:{unit}"].append(row)
    ordered_units = sorted(units.items())
    random.Random(args.seed).shuffle(ordered_units)
    rows = [row for _, values in ordered_units for row in sorted(values, key=lambda item: str(item["example_id"]))]
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("curriculum example IDs are not unique")
    write_jsonl(args.output_jsonl, rows)
    counts: Counter[str] = Counter()
    for row in rows:
        counts[f"source:{row['curriculum_source']}"] += 1
        counts[f"task:{row['task_type']}"] += 1
        counts[f"modality:{'visual' if row['images'] else 'text'}"] += 1
        if row["curriculum_source"] == "seven_task_anchor":
            counts[f"anchor:{row['task_type']}"] += 1
    manifest = {
        "name": "optics_understanding_counterfactual_tool_curriculum_v9",
        "seed": args.seed,
        "record_count": len(rows),
        "counterfactual_record_count": len(counterfactual),
        "forward_refresh_record_count": len(forward),
        "forward_refresh_group_count": len(forward_groups),
        "compact_refresh_record_count": len(compact),
        "compact_refresh_group_count": len(compact_groups),
        "anchor_record_count": len(anchors),
        "sampling_unit_count": len(units),
        "maximum_sampling_unit_size": max(map(len, units.values())),
        "repetition_count": 0,
        "counts": dict(sorted(counts.items())),
        "counterfactual_sha256": file_sha256(args.counterfactual_jsonl),
        "forward_refresh_sha256": file_sha256(args.forward_refresh_jsonl),
        "compact_refresh_sha256": file_sha256(args.compact_refresh_jsonl),
        "anchor_sha256": file_sha256(args.anchor_jsonl),
        "ordered_example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
    }
    args.output_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
