"""Streaming data preparation and exact direction metrics."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from Qwen_orchestration.runtime.numerics import model_feature_vector
from specialist_rebuild_v2.common import (
    DIRECTION_FIELDS,
    STATE_FIELDS,
    forward_feature,
)

CLASSES = ("decrease", "no_change", "increase")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
DISTANCE_BIN_NAMES = ("near", "middle", "far")
DISTANCE_BIN_EDGES = (0.25, 0.75)


@dataclass(frozen=True)
class DirectionArrays:
    features: np.ndarray
    labels: np.ndarray
    distance_bins: np.ndarray
    normalized_changes: np.ndarray
    legacy_features: np.ndarray | None
    category_indices: np.ndarray
    category_names: tuple[str, ...]
    group_count: int

    @property
    def transition_count(self) -> int:
        return int(len(self.features))


def count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def distance_bins(normalized_changes: np.ndarray) -> np.ndarray:
    """Bin absolute distance from the -1/+1 direction boundaries."""

    distance = np.abs(np.abs(np.asarray(normalized_changes)) - 1.0)
    return np.digitize(distance, DISTANCE_BIN_EDGES).astype(np.int64)


def labels_from_normalized_change(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.where(values < -1.0, 0, np.where(values > 1.0, 2, 1)).astype(
        np.int64
    )


def load_grid_arrays(
    path: Path,
    *,
    include_legacy_features: bool,
    max_groups: int | None = None,
) -> DirectionArrays:
    """Flatten grid groups without retaining the large decoded JSON objects."""

    available_groups = count_nonempty_lines(path)
    group_count = (
        available_groups
        if max_groups is None
        else min(int(max_groups), available_groups)
    )
    if group_count < 1:
        raise ValueError(f"no groups selected from {path}")
    transition_count = group_count * len(ACTION_GRID)
    feature_dim = 46
    features = np.empty((transition_count, feature_dim), dtype=np.float32)
    labels = np.empty(
        (transition_count, len(DIRECTION_FIELDS)),
        dtype=np.int64,
    )
    normalized = np.empty_like(labels, dtype=np.float32)
    legacy = (
        np.empty((transition_count, 21), dtype=np.float32)
        if include_legacy_features
        else None
    )
    categories: list[str] = []
    category_values = np.empty(transition_count, dtype=np.int16)
    category_to_index: dict[str, int] = {}

    cursor = 0
    groups_seen = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if groups_seen >= group_count:
                break
            row = json.loads(line)
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != len(ACTION_GRID):
                raise ValueError(
                    f"{path}: {row.get('group_id')} does not have 81 candidates"
                )
            setup = row["setup"]
            current = row["current_beam_state"]
            tolerance = tolerance_from_current(current)
            category = str(row.get("source_category", "old_iid"))
            if category not in category_to_index:
                category_to_index[category] = len(categories)
                categories.append(category)
            category_index = category_to_index[category]

            for action_index, candidate in enumerate(candidates):
                expected_action = ACTION_GRID[action_index]
                action = candidate["action"]
                if any(
                    float(action[field]) != float(expected_action[field])
                    for field in expected_action
                ):
                    raise ValueError(
                        f"{path}: {row.get('group_id')} action order differs"
                    )
                feature = forward_feature(setup, current, action)
                if feature.shape != (feature_dim,):
                    raise ValueError(
                        f"engineered direction feature shape differs: {feature.shape}"
                    )
                features[cursor] = feature
                change = np.asarray(
                    [float(candidate["change"][field]) for field in STATE_FIELDS],
                    dtype=np.float32,
                )
                normalized[cursor] = change / tolerance
                declared = np.asarray(
                    [
                        CLASS_TO_INDEX[candidate["directions"][field]]
                        for field in DIRECTION_FIELDS
                    ],
                    dtype=np.int64,
                )
                derived = labels_from_normalized_change(normalized[cursor])
                if not np.array_equal(declared, derived):
                    raise ValueError(
                        f"{path}: {row.get('group_id')} direction labels differ "
                        "from the registered thresholds"
                    )
                labels[cursor] = declared
                if legacy is not None:
                    legacy[cursor] = model_feature_vector(setup, current, action)
                category_values[cursor] = category_index
                cursor += 1
            groups_seen += 1

    if cursor != transition_count or groups_seen != group_count:
        raise ValueError(
            f"{path}: expected {transition_count} transitions, read {cursor}"
        )
    if not np.isfinite(features).all() or not np.isfinite(normalized).all():
        raise ValueError(f"{path}: direction arrays contain non-finite values")
    return DirectionArrays(
        features=features,
        labels=labels,
        distance_bins=distance_bins(normalized),
        normalized_changes=normalized,
        legacy_features=legacy,
        category_indices=category_values,
        category_names=tuple(categories),
        group_count=group_count,
    )


def concatenate_direction_arrays(
    parts: list[DirectionArrays],
) -> DirectionArrays:
    if not parts:
        raise ValueError("at least one direction array block is required")
    category_names: list[str] = []
    category_lookup: dict[str, int] = {}
    remapped_categories = []
    for part in parts:
        local_to_combined = {}
        for local_index, name in enumerate(part.category_names):
            if name not in category_lookup:
                category_lookup[name] = len(category_names)
                category_names.append(name)
            local_to_combined[local_index] = category_lookup[name]
        remapped_categories.append(
            np.asarray(
                [local_to_combined[int(value)] for value in part.category_indices],
                dtype=np.int16,
            )
        )
    legacy_parts = [part.legacy_features for part in parts]
    if all(value is None for value in legacy_parts):
        legacy = None
    elif all(value is not None for value in legacy_parts):
        legacy = np.concatenate(legacy_parts)  # type: ignore[arg-type]
    else:
        raise ValueError("legacy feature presence differs across blocks")
    return DirectionArrays(
        features=np.concatenate([part.features for part in parts]),
        labels=np.concatenate([part.labels for part in parts]),
        distance_bins=np.concatenate([part.distance_bins for part in parts]),
        normalized_changes=np.concatenate(
            [part.normalized_changes for part in parts]
        ),
        legacy_features=legacy,
        category_indices=np.concatenate(remapped_categories),
        category_names=tuple(category_names),
        group_count=sum(part.group_count for part in parts),
    )


def balance_table(
    labels: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return normalized inverse-sqrt weights for field/class/distance strata."""

    labels = np.asarray(labels, dtype=np.int64)
    bins = np.asarray(bins, dtype=np.int64)
    if labels.shape != bins.shape or labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError("direction labels and distance bins must have shape N x 5")
    counts = np.zeros((5, 3, 3), dtype=np.int64)
    for field in range(5):
        np.add.at(counts[field], (labels[:, field], bins[:, field]), 1)
    weights = np.zeros_like(counts, dtype=np.float32)
    for field in range(5):
        populated = counts[field] > 0
        weights[field][populated] = np.sqrt(
            labels.shape[0] / (9.0 * counts[field][populated])
        )
        sample_weights = weights[field, labels[:, field], bins[:, field]]
        weights[field] /= max(float(sample_weights.mean()), 1e-12)
        weights[field] = np.clip(weights[field], 0.20, 8.0)
        sample_weights = weights[field, labels[:, field], bins[:, field]]
        weights[field] /= max(float(sample_weights.mean()), 1e-12)

    count_report = {
        DIRECTION_FIELDS[field]: {
            CLASSES[class_index]: {
                DISTANCE_BIN_NAMES[bin_index]: int(
                    counts[field, class_index, bin_index]
                )
                for bin_index in range(3)
            }
            for class_index in range(3)
        }
        for field in range(5)
    }
    weight_report = {
        DIRECTION_FIELDS[field]: {
            CLASSES[class_index]: {
                DISTANCE_BIN_NAMES[bin_index]: float(
                    weights[field, class_index, bin_index]
                )
                for bin_index in range(3)
            }
            for class_index in range(3)
        }
        for field in range(5)
    }
    return weights, {
        "distance_definition": "abs(abs(change/tolerance)-1)",
        "distance_bin_edges": list(DISTANCE_BIN_EDGES),
        "distance_bin_names": list(DISTANCE_BIN_NAMES),
        "counts": count_report,
        "weights": weight_report,
    }


def _class_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (target, predicted), 1)
    f1_values = []
    for class_index in range(3):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        false_negative = int(confusion[class_index, :].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
    return {
        "accuracy": float(np.mean(target == predicted)),
        "macro_f1": float(np.mean(f1_values)),
        "class_f1": {
            name: float(f1_values[index]) for index, name in enumerate(CLASSES)
        },
        "confusion_target_rows_predicted_columns": confusion.tolist(),
        "target_distribution": {
            CLASSES[index]: int(value)
            for index, value in enumerate(
                np.bincount(target, minlength=len(CLASSES))
            )
        },
        "predicted_distribution": {
            CLASSES[index]: int(value)
            for index, value in enumerate(
                np.bincount(predicted, minlength=len(CLASSES))
            )
        },
    }


def direction_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    bins: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 2:
        raise ValueError("direction metric arrays must have the same N x 5 shape")
    per_field = {
        field: _class_metrics(labels[:, index], predictions[:, index])
        for index, field in enumerate(DIRECTION_FIELDS)
    }
    result: dict[str, Any] = {
        "count": int(len(labels)),
        "joint_exact_count": int(np.all(labels == predictions, axis=1).sum()),
        "joint_exact": float(np.mean(np.all(labels == predictions, axis=1))),
        "mean_field_accuracy": float(
            np.mean([value["accuracy"] for value in per_field.values()])
        ),
        "equal_field_macro_f1": float(
            np.mean([value["macro_f1"] for value in per_field.values()])
        ),
        "per_field": per_field,
    }
    if bins is not None:
        bins = np.asarray(bins, dtype=np.int64)
        if bins.shape != labels.shape:
            raise ValueError("direction distance-bin shape differs")
        result["accuracy_by_boundary_distance"] = {
            DIRECTION_FIELDS[field]: {
                DISTANCE_BIN_NAMES[bin_index]: (
                    None
                    if not bool((bins[:, field] == bin_index).any())
                    else float(
                        np.mean(
                            labels[bins[:, field] == bin_index, field]
                            == predictions[bins[:, field] == bin_index, field]
                        )
                    )
                )
                for bin_index in range(3)
            }
            for field in range(5)
        }
    return result


def category_metrics(
    arrays: DirectionArrays,
    predictions: np.ndarray,
) -> dict[str, Any]:
    output = {}
    for category_index, category in enumerate(arrays.category_names):
        selected = arrays.category_indices == category_index
        output[category] = direction_metrics(
            arrays.labels[selected],
            predictions[selected],
            arrays.distance_bins[selected],
        )
    return output


def class_count_summary(labels: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        field: {
            CLASSES[index]: int(count)
            for index, count in enumerate(
                np.bincount(labels[:, field_index], minlength=3)
            )
        }
        for field_index, field in enumerate(DIRECTION_FIELDS)
    }
