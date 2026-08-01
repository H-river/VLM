#!/usr/bin/env python3
"""Fit train-only thresholds for the deterministic visual state tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import read_jsonl
from .visual_state_tool_v10_1 import extract_features, fit_ordered_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor-crop-px", type=int, default=512)
    parser.add_argument(
        "--overlay-handling",
        choices=("mask_zero", "interpolate_colored_aids"),
        default="mask_zero",
    )
    parser.add_argument("--noise-floor-sigma", type=float, default=0.0)
    parser.add_argument("--relative-floor", type=float, default=0.0)
    parser.add_argument("--denoise-passes", type=int, default=0)
    args = parser.parse_args()
    rows = [
        row
        for row in read_jsonl(args.records_jsonl)
        if row["task_type"] == "visual_state_classification"
        and row["provenance"].get("transform") == "original"
    ]
    features = {
        row["example_id"]: extract_features(
            args.image_root / row["prompt_inputs"]["images"][0],
            sensor_crop_px=args.sensor_crop_px,
            overlay_handling=args.overlay_handling,
            noise_floor_sigma=args.noise_floor_sigma,
            relative_floor=args.relative_floor,
            denoise_passes=args.denoise_passes,
        )
        for row in rows
    }
    specs = {
        "centroid_x": (
            "centroid_x_px",
            "centroid_horizontal_region",
            ["left_of_center", "centered", "right_of_center"],
        ),
        "centroid_y": (
            "centroid_y_px",
            "centroid_vertical_region",
            ["above_center", "centered", "below_center"],
        ),
        "sigma_x": (
            "rendered_sigma_x_px",
            "sigma_x_band",
            ["narrow", "medium", "wide"],
        ),
        "sigma_y": (
            "rendered_sigma_y_px",
            "sigma_y_band",
            ["narrow", "medium", "wide"],
        ),
    }
    calibration = {
        "dataset_split": "train_only",
        "record_count": len(rows),
        "feature_extractor": "instrumented_rgb_min_moment_v1",
        "sensor_crop_px": args.sensor_crop_px,
        "overlay_handling": args.overlay_handling,
        "noise_floor_sigma": args.noise_floor_sigma,
        "relative_floor": args.relative_floor,
        "denoise_passes": args.denoise_passes,
    }
    for name, (feature_field, target_field, labels) in specs.items():
        fitted = fit_ordered_thresholds(
            [
                (features[row["example_id"]][feature_field], row["target"]["answer"][target_field])
                for row in rows
            ],
            labels,
        )
        calibration[name] = {
            "labels": labels,
            "thresholds": [fitted[1], fitted[2]],
            "train_accuracy": fitted[0],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(calibration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
