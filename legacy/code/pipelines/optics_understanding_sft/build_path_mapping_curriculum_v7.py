#!/usr/bin/env python3
"""Mix focused v7 supervision with balanced seven-task preservation anchors."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import file_sha256, read_jsonl, stable_json_hash, write_jsonl


TASKS = (
    "setup_interpretation",
    "information_sufficiency",
    "causal_effects",
    "forward_prediction",
    "diagnosis",
    "constrained_intervention",
    "counterfactual_reasoning",
)
VISUAL_TASKS = frozenset(
    {"causal_effects", "forward_prediction", "diagnosis", "counterfactual_reasoning"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus-jsonl", type=Path, required=True)
    parser.add_argument("--focus-image-root", type=Path, required=True)
    parser.add_argument("--anchor-jsonl", type=Path, required=True)
    parser.add_argument("--anchor-image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--anchors-per-task", type=int, default=20)
    parser.add_argument("--visual-anchors-per-supported-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def completion_target(row: Mapping[str, Any]) -> dict[str, Any]:
    completion = row.get("completion")
    if not isinstance(completion, list) or not completion:
        raise ValueError(f"missing completion: {row.get('example_id')}")
    content = completion[0].get("content")
    if not isinstance(content, list):
        raise ValueError(f"invalid completion: {row.get('example_id')}")
    text = next(
        (
            item.get("text")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ),
        None,
    )
    target = json.loads(str(text))
    if not isinstance(target, dict):
        raise ValueError(f"completion is not a JSON object: {row.get('example_id')}")
    return target


def prepare(row: Mapping[str, Any], root: Path, source: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(row))
    source_id = str(result["example_id"])
    result["source_example_id"] = source_id
    result["example_id"] = f"{source_id}__curriculum_{source}"
    result["curriculum_source"] = source
    result["images"] = [
        str((root / image).resolve()) if not Path(image).is_absolute() else str(Path(image))
        for image in result.get("images", [])
    ]
    for image in result["images"]:
        if not Path(image).is_file():
            raise FileNotFoundError(image)
    return result


def status(row: Mapping[str, Any]) -> str:
    return str(completion_target(row).get("status", "no_status"))


def stratified_take(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin status strata so preservation does not reinforce a majority label."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[status(row)].append(row)
    randomizer = random.Random(seed)
    for key in sorted(buckets):
        buckets[key].sort(key=lambda value: str(value["example_id"]))
        randomizer.shuffle(buckets[key])
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        advanced = False
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < count:
                chosen.append(buckets[key].pop())
                advanced = True
        if not advanced:
            raise ValueError(f"only {len(chosen)} rows available for requested count {count}")
    return chosen


def select_anchors(
    rows: list[dict[str, Any]],
    per_task: int,
    visual_per_supported_task: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        task_rows = [row for row in rows if row.get("task_type") == task]
        visual_count = visual_per_supported_task if task in VISUAL_TASKS else 0
        visual = [row for row in task_rows if row.get("images")]
        text = [row for row in task_rows if not row.get("images")]
        if len(visual) < visual_count:
            raise ValueError(f"{task} has {len(visual)} visual rows, need {visual_count}")
        chosen_visual = stratified_take(visual, visual_count, seed + 1000 + task_index)
        chosen_text = stratified_take(text, per_task - visual_count, seed + 2000 + task_index)
        selected.extend(chosen_visual + chosen_text)
    return selected


def grouped_order(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], int, int]:
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["curriculum_source"] == "path_mapping_v7":
            unit_id = "focus:" + str(row["group_id"])
        else:
            unit_id = "anchor:" + str(row["example_id"])
        units[unit_id].append(row)
    ordered_units = sorted(units.items())
    random.Random(seed).shuffle(ordered_units)
    ordered = [
        row
        for _, unit_rows in ordered_units
        for row in sorted(unit_rows, key=lambda value: str(value["example_id"]))
    ]
    return ordered, len(units), max(map(len, units.values()), default=0)


def build(args: argparse.Namespace) -> dict[str, Any]:
    focus_source = read_jsonl(args.focus_jsonl)
    anchor_source = read_jsonl(args.anchor_jsonl)
    anchors = select_anchors(
        anchor_source,
        args.anchors_per_task,
        args.visual_anchors_per_supported_task,
        args.seed,
    )
    prepared = [prepare(row, args.focus_image_root, "path_mapping_v7") for row in focus_source]
    prepared += [prepare(row, args.anchor_image_root, "seven_task_anchor") for row in anchors]
    rows, unit_count, maximum_unit_size = grouped_order(prepared, args.seed)
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("curriculum example IDs are not unique")
    write_jsonl(args.output_jsonl, rows)

    counts: Counter[str] = Counter()
    for row in rows:
        task = str(row["task_type"])
        counts[f"source:{row['curriculum_source']}"] += 1
        counts[f"task:{task}"] += 1
        counts[f"modality:{'visual' if row['images'] else 'text'}"] += 1
        if row["curriculum_source"] == "seven_task_anchor":
            counts[f"anchor_status:{task}:{status(row)}"] += 1
    manifest = {
        "name": "optics_understanding_path_mapping_curriculum_v7",
        "seed": args.seed,
        "record_count": len(rows),
        "focus_record_count": len(focus_source),
        "anchor_record_count": len(anchors),
        "anchors_per_task": args.anchors_per_task,
        "visual_anchors_per_supported_task": args.visual_anchors_per_supported_task,
        "sampling_unit_count": unit_count,
        "maximum_sampling_unit_size": maximum_unit_size,
        "repetition_count": 0,
        "focus_sha256": file_sha256(args.focus_jsonl),
        "anchor_source_sha256": file_sha256(args.anchor_jsonl),
        "counts": dict(sorted(counts.items())),
        "ordered_example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
    }
    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
