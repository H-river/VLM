#!/usr/bin/env python3
"""Audit calibrated visual-evidence data and its mixed training curriculum."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .audit_path_mapping_curriculum_v7 import image_placeholder_count
from .core import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--curriculum-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_target(row: Mapping[str, Any]) -> bool:
    target = row.get("target")
    if not isinstance(target, Mapping) or target.get("status") != "answerable":
        return False
    answer = target.get("answer")
    if not isinstance(answer, Mapping):
        return False
    if row["task_type"] == "visual_state_classification":
        allowed = {
            "centroid_horizontal_region": {"left_of_center", "centered", "right_of_center"},
            "centroid_vertical_region": {"above_center", "centered", "below_center"},
            "sigma_x_band": {"narrow", "medium", "wide"},
            "sigma_y_band": {"narrow", "medium", "wide"},
        }
        return all(answer.get(field) in values for field, values in allowed.items())
    values = answer.get("observed_direction_set")
    return isinstance(values, Mapping) and set(values) == {
        "centroid_x",
        "centroid_y",
        "sigma_x",
        "sigma_y",
        "peak_intensity",
    } and all(value in {"increase", "decrease", "no_change"} for value in values.values())


def main() -> None:
    args = parse_args()
    train = read_jsonl(args.dataset_root / "canonical" / "train.jsonl")
    dev = read_jsonl(args.dataset_root / "canonical" / "dev.jsonl")
    curriculum = read_jsonl(args.curriculum_jsonl)
    failures: list[str] = []

    train_groups = {row["group_id"] for row in train}
    dev_groups = {row["group_id"] for row in dev}
    if train_groups & dev_groups:
        failures.append("train/dev group overlap")
    for split, rows in (("train", train), ("dev", dev)):
        if len({row["example_id"] for row in rows}) != len(rows):
            failures.append(f"duplicate {split} example IDs")
        for row in rows:
            images = row.get("prompt_inputs", {}).get("images", [])
            if row.get("modality") != "visual" or not images:
                failures.append(f"nonvisual or image-free record: {row.get('example_id')}")
            prompt = str(row.get("prompt", "")).lower()
            if any(token in prompt for token in ('"action"', "candidate_interventions", "source_example_id")):
                failures.append(f"prompt shortcut field: {row.get('example_id')}")
            if not validate_target(row):
                failures.append(f"invalid target: {row.get('example_id')}")
            calibration = row.get("prompt_inputs", {}).get("render_calibration", {})
            if not calibration.get("pair_shared_calibration"):
                failures.append(f"uncalibrated record: {row.get('example_id')}")
            for relative in images:
                path = args.dataset_root / relative
                if not path.exists():
                    failures.append(f"missing image: {relative}")
                elif Image.open(path).size != (384, 384):
                    failures.append(f"wrong image size: {relative}")

    direction_counts: Counter[str] = Counter()
    for row in train:
        values = row["target"]["answer"].get("observed_direction_set", {})
        for field, value in values.items():
            direction_counts[f"{field}:{value}"] += 1
    for field in ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity"):
        for value in ("increase", "decrease"):
            if direction_counts[f"{field}:{value}"] < 3:
                failures.append(f"too few {field}:{value} training labels")

    for row in curriculum:
        images = list(row.get("images", []))
        if image_placeholder_count(row) != len(images):
            failures.append(f"curriculum placeholder mismatch: {row.get('example_id')}")
        for relative in images:
            if not (args.dataset_root / relative).exists():
                failures.append(f"curriculum missing image: {relative}")
    source_counts = Counter(row.get("v10_curriculum_source") for row in curriculum)
    allowed_source_counts = (
        {"visual_state_v10": 220, "visual_pair_v10": 180},
        {
            "visual_state_v10": 220,
            "visual_pair_v10": 180,
            "preservation_anchor": 180,
        },
    )
    if dict(source_counts) not in allowed_source_counts:
        failures.append(f"curriculum source counts differ: {dict(source_counts)}")

    report = {
        "passed": not failures,
        "checks": {
            "scenario_disjoint": not bool(train_groups & dev_groups),
            "all_records_visual_and_calibrated": not any(
                "nonvisual" in value or "uncalibrated" in value for value in failures
            ),
            "images_present_and_384": not any(
                "image" in value and ("missing" in value or "size" in value)
                for value in failures
            ),
            "pixel_shortcuts_absent": not any("shortcut" in value for value in failures),
            "targets_valid": not any("invalid target" in value for value in failures),
            "curriculum_valid": not any("curriculum" in value for value in failures),
        },
        "counts": {
            "train": len(train),
            "dev": len(dev),
            "curriculum": len(curriculum),
            "train_groups": len(train_groups),
            "dev_groups": len(dev_groups),
        },
        "curriculum_source_counts": dict(source_counts),
        "failures": failures[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
