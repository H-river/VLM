#!/usr/bin/env python3
"""Build the non-repeated action-first-v3 curriculum.

The curriculum keeps every focused v3 record and adds a small, balanced set of
preservation anchors from tasks that are not being corrected.  Matched control
groups remain adjacent so the signed-action contrast is visible to the trainer.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .build_corrective_curriculum import completion_target, grouped_order
from .core import file_sha256, read_jsonl, stable_json_hash, write_jsonl


ANCHOR_TASKS = (
    "setup_interpretation",
    "causal_effects",
    "forward_prediction",
    "diagnosis",
    "counterfactual_reasoning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-jsonl", type=Path, required=True)
    parser.add_argument("--focused-image-root", type=Path, required=True)
    parser.add_argument("--anchor-jsonl", type=Path, required=True)
    parser.add_argument("--anchor-image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--anchors-per-task", type=int, default=32)
    parser.add_argument("--seed", type=int, default=303)
    return parser.parse_args()


def prepare(row: Mapping[str, Any], root: Path, source: str) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    original_id = str(item["example_id"])
    # Preserve the earliest available lineage ID when anchors already came
    # through a prior curriculum builder.  This makes overlap audits compare
    # the physical source example rather than a curriculum-specific suffix.
    item["source_example_id"] = str(item.get("source_example_id", original_id))
    item["example_id"] = f"{original_id}__{source}"
    item["curriculum_source"] = source
    images = item.get("images", [])
    item["images"] = [
        str((root / value).resolve()) if not Path(value).is_absolute() else value
        for value in images
    ]
    for image in item["images"]:
        if not Path(image).exists():
            raise FileNotFoundError(image)
    return item


def stratified_anchor_sample(
    rows: list[dict[str, Any]], *, per_task: int, seed: int
) -> list[dict[str, Any]]:
    """Select exact per-task quotas while cycling through target statuses."""
    selected: list[dict[str, Any]] = []
    for task_index, task in enumerate(ANCHOR_TASKS):
        by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("task_type") == task:
                by_status[str(completion_target(row)["status"])].append(row)
        if not by_status:
            raise ValueError(f"no anchor rows for task {task}")
        rng = random.Random(seed + task_index * 1009)
        for values in by_status.values():
            values.sort(key=lambda value: str(value["example_id"]))
            rng.shuffle(values)
        status_names = sorted(by_status)
        offsets = {status: 0 for status in status_names}
        task_selected: list[dict[str, Any]] = []
        while len(task_selected) < per_task:
            made_progress = False
            for status in status_names:
                offset = offsets[status]
                values = by_status[status]
                if offset < len(values) and len(task_selected) < per_task:
                    task_selected.append(values[offset])
                    offsets[status] += 1
                    made_progress = True
            if not made_progress:
                raise ValueError(f"need {per_task} unique anchors for {task}")
        selected.extend(task_selected)
    return selected


def main() -> None:
    args = parse_args()
    focused = read_jsonl(args.focused_jsonl)
    if len(focused) != 640:
        raise ValueError(f"expected 640 focused records, got {len(focused)}")
    anchor_all = read_jsonl(args.anchor_jsonl)
    anchors = stratified_anchor_sample(
        anchor_all, per_task=args.anchors_per_task, seed=args.seed
    )

    prepared = [
        prepare(row, args.focused_image_root, "action_first_v3") for row in focused
    ]
    prepared += [
        prepare(row, args.anchor_image_root, "preservation_anchor") for row in anchors
    ]
    rows, sampling_unit_count, maximum_sampling_unit_size = grouped_order(
        prepared, args.seed
    )
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("curriculum example IDs are not unique")
    write_jsonl(args.output_jsonl, rows)

    counts: Counter[str] = Counter()
    for row in rows:
        target = completion_target(row)
        task = str(row["task_type"])
        status = str(target["status"])
        counts[f"source:{row['curriculum_source']}"] += 1
        counts[f"task:{task}"] += 1
        counts[f"status:{task}:{status}"] += 1
        counts[f"modality:{'visual' if row['images'] else 'text'}"] += 1

    manifest = {
        "name": "optics_understanding_action_first_curriculum_v3",
        "seed": args.seed,
        "record_count": len(rows),
        "unique_source_examples": len(rows),
        "repetition_count": 0,
        "sampling_unit_count": sampling_unit_count,
        "maximum_sampling_unit_size": maximum_sampling_unit_size,
        "focused_count": len(focused),
        "anchor_count": len(anchors),
        "anchors_per_task": args.anchors_per_task,
        "focused_sha256": file_sha256(args.focused_jsonl),
        "anchor_pool_sha256": file_sha256(args.anchor_jsonl),
        "counts": dict(sorted(counts.items())),
        "example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
    }
    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
