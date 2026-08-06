#!/usr/bin/env python3
"""Merge pilot and fresh augmentation rows into a deterministic targeted curriculum."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .core import file_sha256, read_jsonl, stable_json_hash, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--base-image-root", type=Path, required=True)
    parser.add_argument("--augmentation-jsonl", type=Path, required=True)
    parser.add_argument("--augmentation-image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202)
    return parser.parse_args()


def target_from_qwen(row: Mapping[str, Any]) -> dict[str, Any]:
    completion = row.get("completion")
    if not isinstance(completion, list) or not completion:
        raise ValueError(f"missing completion: {row.get('example_id')}")
    content = completion[0].get("content")
    if not isinstance(content, list):
        raise ValueError(f"invalid completion content: {row.get('example_id')}")
    text = next((item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"), None)
    parsed = json.loads(str(text))
    if not isinstance(parsed, dict):
        raise ValueError(f"completion target is not an object: {row.get('example_id')}")
    return parsed


def repetitions(task: str, status: str, source: str) -> int:
    """Training-only sampling weights derived before the v1.1 ablations."""
    if task == "setup_interpretation":
        return 2
    if task == "information_sufficiency":
        return 3 if status == "insufficient_information" else 1
    if task == "constrained_intervention":
        return 4 if status == "feasible" else 1
    if task == "diagnosis":
        return 2 if status in {"unique", "unsupported"} else 1
    if task == "counterfactual_reasoning":
        return 2 if source == "augmentation" else 1
    return 1


def absolute_images(row: dict[str, Any], root: Path) -> dict[str, Any]:
    result = copy.deepcopy(row)
    images = result.get("images", [])
    result["images"] = [str((root / value).resolve()) if not Path(value).is_absolute() else value for value in images]
    return result


def expand(rows: list[dict[str, Any]], image_root: Path, source: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        target = target_from_qwen(row)
        status = str(target.get("status"))
        task = str(row.get("task_type"))
        repeat = repetitions(task, status, source)
        for repetition in range(repeat):
            item = absolute_images(row, image_root)
            original_id = str(row["example_id"])
            item["source_example_id"] = original_id
            item["example_id"] = f"{original_id}__{source[:3]}r{repetition}"
            item["curriculum_source"] = source
            item["curriculum_repetition"] = repetition
            output.append(item)
            counts[f"source:{source}"] += 1
            counts[f"task:{task}"] += 1
            counts[f"status:{task}:{status}"] += 1
            counts[f"modality:{'visual' if item['images'] else 'text'}"] += 1
    return output, counts


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.base_jsonl)
    augmentation_rows = read_jsonl(args.augmentation_jsonl)
    base_expanded, base_counts = expand(base_rows, args.base_image_root, "base")
    augmentation_expanded, augmentation_counts = expand(
        augmentation_rows, args.augmentation_image_root, "augmentation"
    )
    rows = base_expanded + augmentation_expanded
    random.Random(args.seed).shuffle(rows)
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("curriculum example IDs are not unique")
    for row in rows:
        for image in row["images"]:
            if not Path(image).exists():
                raise FileNotFoundError(image)
    write_jsonl(args.output_jsonl, rows)
    counts = base_counts + augmentation_counts
    manifest = {
        "name": "optics_understanding_targeted_curriculum_v1_1",
        "seed": args.seed,
        "record_count": len(rows),
        "unique_source_examples": len(base_rows) + len(augmentation_rows),
        "base_jsonl": str(args.base_jsonl.resolve()),
        "augmentation_jsonl": str(args.augmentation_jsonl.resolve()),
        "base_sha256": file_sha256(args.base_jsonl),
        "augmentation_sha256": file_sha256(args.augmentation_jsonl),
        "counts": dict(sorted(counts.items())),
        "example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
    }
    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
