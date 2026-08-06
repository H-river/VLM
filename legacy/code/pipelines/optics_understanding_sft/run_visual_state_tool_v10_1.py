#!/usr/bin/env python3
"""Run the deterministic state tool and emit evaluator-compatible predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import read_jsonl, write_jsonl
from .visual_state_tool_v10_1 import classify, extract_features, load_calibration


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
        if row["task_type"] == "visual_state_classification"
    ]
    predictions = []
    for row in records:
        features = extract_features(
            args.image_root / row["prompt_inputs"]["images"][0],
            sensor_crop_px=int(calibration.get("sensor_crop_px", 512)),
            overlay_handling=str(calibration.get("overlay_handling", "mask_zero")),
            noise_floor_sigma=float(calibration.get("noise_floor_sigma", 0.0)),
            relative_floor=float(calibration.get("relative_floor", 0.0)),
        )
        predictions.append(
            {
                "example_id": row["example_id"],
                "group_id": row["group_id"],
                "task_type": row["task_type"],
                "modality": "visual_tool",
                "parsed_json": {"status": "answerable", "answer": classify(features, calibration)},
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
