#!/usr/bin/env python3
"""Fit train-only thresholds for deterministic paired visual evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import read_jsonl
from .visual_state_tool_v10_1 import extract_pair_features, fit_ordered_thresholds


FIELDS = ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for row in read_jsonl(args.records_jsonl)
        if row["task_type"] == "visual_pair_direction_extraction"
        and row["provenance"].get("transform") == "original"
    ]
    features = {
        row["example_id"]: extract_pair_features(
            args.image_root / row["prompt_inputs"]["images"][0],
            args.image_root / row["prompt_inputs"]["images"][1],
        )
        for row in rows
    }
    calibration = {
        "dataset_split": "train_only",
        "record_count": len(rows),
        "feature_extractor": "paired_instrumented_moments_and_energy_v1",
    }
    for field in FIELDS:
        fitted = fit_ordered_thresholds(
            [
                (
                    features[row["example_id"]][field],
                    row["target"]["answer"]["observed_direction_set"][field],
                )
                for row in rows
            ],
            ["decrease", "no_change", "increase"],
        )
        calibration[field] = {
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
