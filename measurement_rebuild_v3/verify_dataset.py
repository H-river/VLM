#!/usr/bin/env python3
"""Verify corrected measurement dataset counts, images, transforms and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measurement_rebuild_v3.common import (
    analytic_measurement,
    iter_jsonl,
    measurement_tolerance,
    read_json,
)
from measurement_rebuild_v3.train import load_linear_image, metric_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def verify_checksums(data_dir: Path) -> int:
    checksum_path = data_dir / "checksums.sha256"
    lines = [line for line in checksum_path.read_text().splitlines() if line]
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = data_dir / relative
        if not path.is_file():
            raise RuntimeError(f"missing checksummed file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {relative}")
    return len(lines)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    config = read_json(data_dir / "config.json")
    manifest = read_json(data_dir / "manifest.json")
    if manifest["dataset_version"] != config["version"]:
        raise RuntimeError("dataset version mismatch")
    conditions = list(config["conditions"])
    seen_groups: set[str] = set()
    checked_images = 0
    baseline_targets, baseline_predictions = [], []
    split_report: dict[str, Any] = {}
    for split, expected in manifest["split_summary"].items():
        states = list(iter_jsonl(data_dir / "states" / f"{split}.jsonl"))
        views = list(iter_jsonl(data_dir / "views" / f"{split}.jsonl"))
        groups = {row["group_id"] for row in states}
        if seen_groups & groups:
            raise RuntimeError(f"group overlap in split {split}")
        seen_groups |= groups
        if len(groups) != int(expected["groups"]):
            raise RuntimeError(f"{split}: group count mismatch")
        if len(states) != int(expected["base_states"]):
            raise RuntimeError(f"{split}: state count mismatch")
        if len(views) != int(expected["views"]):
            raise RuntimeError(f"{split}: view count mismatch")
        condition_counts = Counter(row["condition"] for row in views)
        if condition_counts != Counter(expected["conditions"]):
            raise RuntimeError(f"{split}: condition counts mismatch")
        state_ids = {row["state_id"] for row in states}
        if len(state_ids) != len(states):
            raise RuntimeError(f"{split}: duplicate state_id")
        view_ids = {row["view_id"] for row in views}
        if len(view_ids) != len(views):
            raise RuntimeError(f"{split}: duplicate view_id")
        for row in states:
            path = data_dir / row["base_image"]
            if not path.is_file():
                raise RuntimeError(f"missing image: {row['base_image']}")
            with Image.open(path) as image:
                if image.size != (
                    int(config["stored_resolution_px"]),
                    int(config["stored_resolution_px"]),
                ):
                    raise RuntimeError(f"wrong image size: {row['base_image']}")
                if image.mode not in {"I;16", "I"}:
                    raise RuntimeError(
                        f"wrong image mode {image.mode}: {row['base_image']}"
                    )
            checked_images += 1
            if not args.skip_baseline:
                base = load_linear_image(path)
                calibration = row["image_calibration"]
                baseline, _ = analytic_measurement(
                    base,
                    np.ones_like(base, dtype=np.float32),
                    float(calibration["linear_intensity_high"]),
                    tuple(calibration["source_sensor_resolution_px"]),
                )
                target = np.asarray(
                    [
                        float(row["target_state"][field])
                        for field in (
                            "centroid_x_px",
                            "centroid_y_px",
                            "sigma_x_px",
                            "sigma_y_px",
                            "peak_intensity",
                        )
                    ],
                    dtype=np.float32,
                )
                baseline_targets.append(target)
                baseline_predictions.append(baseline)
        for row in views:
            if row["state_id"] not in state_ids:
                raise RuntimeError(f"{split}: view references unknown state")
            expected_transform = config["condition_parameters"][row["condition"]]
            if row["transform"] != expected_transform:
                raise RuntimeError(f"{split}: transform metadata mismatch")
        split_report[split] = {
            "groups": len(groups),
            "base_states": len(states),
            "views": len(views),
            "conditions": dict(condition_counts),
        }
    if checked_images != int(manifest["total_base_images"]):
        raise RuntimeError("total image count mismatch")
    checksum_count = None
    if not args.skip_checksums:
        checksum_count = verify_checksums(data_dir)
        if checksum_count != int(manifest["checksum_file_count"]):
            raise RuntimeError("checksum file count mismatch")
    baseline_metrics = None
    if baseline_targets:
        baseline_metrics = metric_block(
            np.asarray(baseline_targets), np.asarray(baseline_predictions)
        )
    report = {
        "passed": True,
        "dataset_version": manifest["dataset_version"],
        "groups": len(seen_groups),
        "base_images": checked_images,
        "views": int(manifest["total_views"]),
        "checksummed_files": checksum_count,
        "clean_analytic_baseline": baseline_metrics,
        "split_summary": split_report,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

