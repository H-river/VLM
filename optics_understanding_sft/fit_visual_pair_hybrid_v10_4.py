#!/usr/bin/env python3
"""Fit a group-CV-selected threshold/kNN hybrid for paired visual evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import read_jsonl
from .visual_state_tool_v10_1 import (
    extract_pair_features,
    fit_ordered_thresholds,
    neighbor_indices_with_ties,
)


FIELDS = ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")
ABSOLUTE_FEATURES = (
    "centroid_x_first",
    "centroid_x_second",
    "centroid_y_first",
    "centroid_y_second",
    "sigma_x_first",
    "sigma_x_second",
    "sigma_y_first",
    "sigma_y_second",
    "peak_proxy_first",
    "peak_proxy_second",
)
DIFFERENTIAL_FEATURES = (
    "diff_positive_total_fraction",
    "diff_positive_centroid_x",
    "diff_positive_centroid_y",
    "diff_positive_sigma_x",
    "diff_positive_sigma_y",
    "diff_negative_centroid_x",
    "diff_negative_centroid_y",
    "diff_negative_sigma_x",
    "diff_negative_sigma_y",
    "diff_sigma_x_contrast",
    "diff_sigma_y_contrast",
)
K_CANDIDATES = (1, 3, 5, 7, 9, 15, 25)


def fold_for_group(group_id: str) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 5


def majority(labels: list[str]) -> str:
    counts = {label: labels.count(label) for label in set(labels)}
    return min(counts, key=lambda label: (-counts[label], label))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor-crop-px", type=int, default=512)
    parser.add_argument("--objective", choices=("accuracy", "macro_f1"), default="accuracy")
    parser.add_argument(
        "--peak-feature",
        choices=("energy_squared_ratio", "top10_mean_ratio"),
        default="energy_squared_ratio",
    )
    parser.add_argument(
        "--overlay-handling",
        choices=("mask_zero", "interpolate_colored_aids"),
        default="mask_zero",
    )
    parser.add_argument("--sigma-x-power", type=float, default=1.0)
    parser.add_argument("--sigma-y-power", type=float, default=1.0)
    parser.add_argument("--noise-floor-sigma", type=float, default=0.0)
    parser.add_argument("--relative-floor", type=float, default=0.0)
    parser.add_argument(
        "--width-estimator",
        choices=("moments_2d", "projected_1d"),
        default="moments_2d",
    )
    parser.add_argument(
        "--feature-set",
        choices=("delta", "delta_absolute", "delta_differential"),
        default="delta",
        help="Features used by kNN fields; scalar threshold fields always use their named delta.",
    )
    parser.add_argument("--differential-floor-sigma", type=float, default=3.0)
    args = parser.parse_args()
    rows = [
        row
        for row in read_jsonl(args.records_jsonl)
        if row["task_type"] == "visual_pair_direction_extraction"
        and row["provenance"].get("transform") == "original"
    ]
    feature_maps = [
        extract_pair_features(
            args.image_root / row["prompt_inputs"]["images"][0],
            args.image_root / row["prompt_inputs"]["images"][1],
            sensor_crop_px=args.sensor_crop_px,
            peak_feature=args.peak_feature,
            overlay_handling=args.overlay_handling,
            sigma_x_power=args.sigma_x_power,
            sigma_y_power=args.sigma_y_power,
            noise_floor_sigma=args.noise_floor_sigma,
            relative_floor=args.relative_floor,
            width_estimator=args.width_estimator,
            include_differential_features=args.feature_set == "delta_differential",
            differential_floor_sigma=args.differential_floor_sigma,
        )
        for row in rows
    ]
    feature_order = list(FIELDS)
    if args.feature_set == "delta_absolute":
        feature_order.extend(ABSOLUTE_FEATURES)
    elif args.feature_set == "delta_differential":
        feature_order.extend(DIFFERENTIAL_FEATURES)
    matrix = np.asarray(
        [[features[field] for field in feature_order] for features in feature_maps],
        dtype=np.float64,
    )
    folds = np.asarray([fold_for_group(str(row["group_id"])) for row in rows])
    field_specs = {}
    for field in FIELDS:
        column = feature_order.index(field)
        labels = np.asarray(
            [row["target"]["answer"]["observed_direction_set"][field] for row in rows]
        )
        cv_targets: list[str] = []
        threshold_predictions: list[str] = []
        knn_predictions: dict[int, list[str]] = {k: [] for k in K_CANDIDATES}
        for fold in range(5):
            train_mask = folds != fold
            val_mask = ~train_mask
            _, low, high = fit_ordered_thresholds(
                list(zip(matrix[train_mask, column], labels[train_mask])),
                ["decrease", "no_change", "increase"],
                objective=args.objective,
            )
            fold_threshold_predictions = np.where(
                matrix[val_mask, column] < low,
                "decrease",
                np.where(matrix[val_mask, column] > high, "increase", "no_change"),
            )
            cv_targets.extend(str(value) for value in labels[val_mask])
            threshold_predictions.extend(str(value) for value in fold_threshold_predictions)
            center = matrix[train_mask].mean(axis=0)
            scale = matrix[train_mask].std(axis=0)
            scale[scale == 0.0] = 1.0
            distances = np.square(
                (matrix[val_mask, None, :] - matrix[None, train_mask, :]) / scale
            ).sum(axis=2)
            train_labels = labels[train_mask]
            for row_distances, target in zip(distances, labels[val_mask]):
                for k in K_CANDIDATES:
                    indices = neighbor_indices_with_ties(row_distances, k)
                    prediction = majority([str(value) for value in train_labels[indices]])
                    knn_predictions[k].append(prediction)

        def score(predictions: list[str]) -> float:
            if args.objective == "accuracy":
                return float(np.mean(np.asarray(predictions) == np.asarray(cv_targets)))
            values = []
            for label in ("decrease", "no_change", "increase"):
                tp = sum(t == label and p == label for t, p in zip(cv_targets, predictions))
                fp = sum(t != label and p == label for t, p in zip(cv_targets, predictions))
                fn = sum(t == label and p != label for t, p in zip(cv_targets, predictions))
                denominator = 2 * tp + fp + fn
                values.append(2 * tp / denominator if denominator else 0.0)
            return float(np.mean(values))

        threshold_cv = score(threshold_predictions)
        knn_scores = {k: score(values) for k, values in knn_predictions.items()}
        best_k = max(K_CANDIDATES, key=lambda value: (knn_scores[value], -value))
        _, low, high = fit_ordered_thresholds(
            list(zip(matrix[:, column], labels)),
            ["decrease", "no_change", "increase"],
            objective=args.objective,
        )
        if threshold_cv >= knn_scores[best_k]:
            field_specs[field] = {
                "method": "threshold",
                "thresholds": [low, high],
                "cross_validation_accuracy": threshold_cv,
                "knn_best_cross_validation_accuracy": knn_scores[best_k],
            }
        else:
            field_specs[field] = {
                "method": "knn",
                "k": best_k,
                "cross_validation_accuracy": knn_scores[best_k],
                "threshold_cross_validation_accuracy": threshold_cv,
            }
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0.0] = 1.0
    calibration = {
        "classifier": "group_cv_hybrid_v1",
        "dataset_split": "train_only",
        "selection_objective": args.objective,
        "record_count": len(rows),
        "group_count": len({row["group_id"] for row in rows}),
        "feature_extractor": "paired_instrumented_moments_and_energy_v1",
        "sensor_crop_px": args.sensor_crop_px,
        "peak_feature": args.peak_feature,
        "overlay_handling": args.overlay_handling,
        "sigma_x_power": args.sigma_x_power,
        "sigma_y_power": args.sigma_y_power,
        "noise_floor_sigma": args.noise_floor_sigma,
        "relative_floor": args.relative_floor,
        "width_estimator": args.width_estimator,
        "include_differential_features": args.feature_set == "delta_differential",
        "differential_floor_sigma": args.differential_floor_sigma,
        "feature_set": args.feature_set,
        "feature_order": feature_order,
        "normalization": {"mean": center.tolist(), "scale": scale.tolist()},
        "fields": field_specs,
        "prototypes": [
            {
                "features": matrix[index].tolist(),
                "labels": {
                    field: row["target"]["answer"]["observed_direction_set"][field]
                    for field in FIELDS
                },
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
