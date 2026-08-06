#!/usr/bin/env python3
"""Fit a group-CV-selected kNN calibrator from original training images only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import read_jsonl
from .visual_state_tool_v10_1 import STATE_FIELDS, extract_features, neighbor_indices_with_ties


FEATURE_ORDER = (
    "centroid_x_px",
    "centroid_y_px",
    "rendered_sigma_x_px",
    "rendered_sigma_y_px",
)
K_CANDIDATES = (1, 3, 5, 7, 9, 15, 25)


def fold_for_group(group_id: str, folds: int = 5) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % folds


def majority(labels: list[str]) -> str:
    counts = {label: labels.count(label) for label in set(labels)}
    return min(counts, key=lambda label: (-counts[label], label))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor-crop-px", type=int, default=512)
    args = parser.parse_args()
    rows = [
        row
        for row in read_jsonl(args.records_jsonl)
        if row["task_type"] == "visual_state_classification"
        and row["provenance"].get("transform") == "original"
    ]
    feature_maps = [
        extract_features(
            args.image_root / row["prompt_inputs"]["images"][0],
            sensor_crop_px=args.sensor_crop_px,
        )
        for row in rows
    ]
    matrix = np.asarray(
        [[features[key] for key in FEATURE_ORDER] for features in feature_maps],
        dtype=np.float64,
    )
    folds = np.asarray([fold_for_group(str(row["group_id"])) for row in rows])
    field_specs: dict[str, Any] = {}
    for field in STATE_FIELDS:
        labels = np.asarray([row["target"]["answer"][field] for row in rows])
        scores: dict[int, float] = {}
        for k in K_CANDIDATES:
            correct: list[bool] = []
            for fold in range(5):
                train_mask = folds != fold
                val_mask = ~train_mask
                center = matrix[train_mask].mean(axis=0)
                scale = matrix[train_mask].std(axis=0)
                scale[scale == 0.0] = 1.0
                distances = np.square(
                    (matrix[val_mask, None, :] - matrix[None, train_mask, :]) / scale
                ).sum(axis=2)
                train_labels = labels[train_mask]
                for row_distances, target in zip(distances, labels[val_mask]):
                    indices = neighbor_indices_with_ties(row_distances, k)
                    prediction = majority([str(value) for value in train_labels[indices]])
                    correct.append(prediction == target)
            scores[k] = float(np.mean(correct))
        chosen = max(K_CANDIDATES, key=lambda value: (scores[value], -value))
        field_specs[field] = {
            "k": chosen,
            "cross_validation_accuracy": scores[chosen],
            "candidate_accuracies": {str(k): scores[k] for k in K_CANDIDATES},
        }
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0.0] = 1.0
    calibration = {
        "classifier": "group_cv_knn_v1",
        "dataset_split": "train_only",
        "record_count": len(rows),
        "group_count": len({row["group_id"] for row in rows}),
        "feature_extractor": "instrumented_moments_v1",
        "sensor_crop_px": args.sensor_crop_px,
        "feature_order": list(FEATURE_ORDER),
        "normalization": {"mean": center.tolist(), "scale": scale.tolist()},
        "fields": field_specs,
        "prototypes": [
            {
                "features": matrix[index].tolist(),
                "labels": {field: row["target"]["answer"][field] for field in STATE_FIELDS},
                "group_id": row["group_id"],
            }
            for index, row in enumerate(rows)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"record_count": len(rows), "fields": field_specs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
