#!/usr/bin/env python3
"""Run deterministic paired visual evidence and emit evaluator records."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import read_jsonl, write_jsonl
from .visual_state_tool_v10_1 import (
    classify_pair,
    extract_pair_features,
    load_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    calibration = load_calibration(args.calibration)
    records = [
        row
        for row in read_jsonl(args.records_jsonl)
        if row["task_type"] == "visual_pair_direction_extraction"
    ]
    predictions = []
    for row in records:
        images = row["prompt_inputs"]["images"]
        features = extract_pair_features(
            args.image_root / images[0],
            args.image_root / images[1],
            sensor_crop_px=int(calibration.get("sensor_crop_px", 512)),
            peak_feature=str(calibration.get("peak_feature", "energy_squared_ratio")),
            overlay_handling=str(calibration.get("overlay_handling", "mask_zero")),
            sigma_x_power=float(calibration.get("sigma_x_power", 1.0)),
            sigma_y_power=float(calibration.get("sigma_y_power", 1.0)),
            noise_floor_sigma=float(calibration.get("noise_floor_sigma", 0.0)),
            relative_floor=float(calibration.get("relative_floor", 0.0)),
        )
        predictions.append(
            {
                "example_id": row["example_id"],
                "group_id": row["group_id"],
                "task_type": row["task_type"],
                "modality": "visual_tool",
                "parsed_json": {
                    "status": "answerable",
                    "answer": {
                        "observed_direction_set": classify_pair(features, calibration)
                    },
                },
                "parse_error": None,
                "tool_features": features,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "records.jsonl", records)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    print(f"wrote {len(predictions)} deterministic predictions")


if __name__ == "__main__":
    main()
