#!/usr/bin/env python3
"""Build the non-repeated corrective-v2 Qwen curriculum."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import file_sha256, read_jsonl, stable_json_hash, write_jsonl


ANCHOR_TASKS = frozenset(
    {"setup_interpretation", "causal_effects", "forward_prediction", "counterfactual_reasoning"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--base-image-root", type=Path, required=True)
    parser.add_argument("--corrective-jsonl", type=Path, required=True)
    parser.add_argument("--corrective-image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202)
    return parser.parse_args()


def completion_target(row: Mapping[str, Any]) -> dict[str, Any]:
    completion = row.get("completion")
    if not isinstance(completion, list) or not completion:
        raise ValueError(f"missing completion: {row.get('example_id')}")
    content = completion[0].get("content")
    if not isinstance(content, list):
        raise ValueError(f"invalid completion: {row.get('example_id')}")
    text = next(
        (item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"),
        None,
    )
    target = json.loads(str(text))
    if not isinstance(target, dict):
        raise ValueError(f"completion is not an object: {row.get('example_id')}")
    return target


def prepare(row: Mapping[str, Any], root: Path, source: str) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    original_id = str(item["example_id"])
    item["source_example_id"] = original_id
    item["example_id"] = f"{original_id}__{source}"
    item["curriculum_source"] = source
    images = item.get("images", [])
    item["images"] = [str((root / value).resolve()) if not Path(value).is_absolute() else value for value in images]
    for image in item["images"]:
        if not Path(image).exists():
            raise FileNotFoundError(image)
    return item


def grouped_order(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], int, int]:
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        unit_id = str(row.get("match_group_id") or f"singleton:{row['example_id']}")
        units[unit_id].append(row)
    ordered_units = sorted(units.items())
    random.Random(seed).shuffle(ordered_units)
    ordered = [row for _, unit_rows in ordered_units for row in sorted(unit_rows, key=lambda value: value["example_id"])]
    return ordered, len(units), max(map(len, units.values()), default=0)


def main() -> None:
    args = parse_args()
    base_all = read_jsonl(args.base_jsonl)
    base = [row for row in base_all if row["task_type"] in ANCHOR_TASKS]
    corrective = read_jsonl(args.corrective_jsonl)
    prepared = [prepare(row, args.base_image_root, "base_anchor") for row in base]
    prepared += [prepare(row, args.corrective_image_root, "corrective_v2") for row in corrective]
    rows, sampling_unit_count, maximum_sampling_unit_size = grouped_order(prepared, args.seed)
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
        "name": "optics_understanding_corrective_curriculum_v2",
        "seed": args.seed,
        "record_count": len(rows),
        "unique_source_examples": len(rows),
        "repetition_count": 0,
        "sampling_unit_count": sampling_unit_count,
        "maximum_sampling_unit_size": maximum_sampling_unit_size,
        "base_anchor_count": len(base),
        "corrective_count": len(corrective),
        "base_sha256": file_sha256(args.base_jsonl),
        "corrective_sha256": file_sha256(args.corrective_jsonl),
        "counts": dict(sorted(counts.items())),
        "example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
    }
    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
