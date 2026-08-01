#!/usr/bin/env python3
"""Fit a group-local forward/direction interpolator on fixed-action grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from direction_rebuild_v4.data import (
    concatenate_direction_arrays,
    labels_from_normalized_change,
    load_grid_arrays,
)

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-train",
        type=Path,
        default=DEFAULT_DATA / "specialist_rebuild_v2/grids/train.jsonl",
    )
    parser.add_argument(
        "--difficult-train",
        type=Path,
        default=DEFAULT_DATA / "control_rebuild_v4_quickcheck/grids/train.jsonl",
    )
    parser.add_argument(
        "--targeted-train",
        type=Path,
        default=DEFAULT_DATA / "physics_structured_rebuild_v9/targeted_bundle/grids/train.jsonl",
    )
    parser.add_argument(
        "--old-validation",
        type=Path,
        default=DEFAULT_DATA / "specialist_rebuild_v2/grids/val.jsonl",
    )
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DATA / "control_rebuild_v4_quickcheck/grids/val.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN / "local_physics_interpolator_v9.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_RUN / "local_physics_interpolator_validation.json",
    )
    parser.add_argument("--neighbors", type=int, nargs="+", default=(2, 4, 8, 16, 32, 64))
    parser.add_argument("--powers", type=float, nargs="+", default=(1.0, 2.0))
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_view(values: np.ndarray, group_count: int) -> np.ndarray:
    return np.asarray(values).reshape(group_count, 81, *values.shape[1:])


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target_labels = labels_from_normalized_change(target)
    predicted_labels = labels_from_normalized_change(prediction)
    field_direction = predicted_labels == target_labels
    direction_exact = np.all(field_direction, axis=2)
    field_forward = np.abs(prediction - target) <= 1.0
    forward_exact = np.all(field_forward, axis=2)
    return {
        "count": int(target.shape[0] * target.shape[1]),
        "direction_all_five_count": int(direction_exact.sum()),
        "direction_all_five_exact": float(direction_exact.mean()),
        "direction_per_field": field_direction.mean(axis=(0, 1)).tolist(),
        "forward_strict_count": int(forward_exact.sum()),
        "forward_strict_success": float(forward_exact.mean()),
        "forward_per_field": field_forward.mean(axis=(0, 1)).tolist(),
        "mae_normalized": float(np.abs(prediction - target).mean()),
    }


def interpolate(
    train_targets: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
    neighbors: int,
    power: float,
) -> np.ndarray:
    local_indices = indices[:, :neighbors]
    local_distances = distances[:, :neighbors]
    weights = 1.0 / np.maximum(local_distances, 1e-6) ** float(power)
    weights /= weights.sum(axis=1, keepdims=True)
    gathered = train_targets[local_indices]
    return np.einsum("vk,vkaj->vaj", weights, gathered).astype(np.float32)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    report_path = args.report.resolve()
    for path in (output, report_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite output: {path}")
    neighbors = sorted(set(int(value) for value in args.neighbors))
    powers = sorted(set(float(value) for value in args.powers))
    if not neighbors or neighbors[0] < 1 or not powers or powers[0] <= 0.0:
        raise ValueError("neighbor counts and powers must be positive")

    train_paths = [
        args.old_train.resolve(),
        args.difficult_train.resolve(),
        args.targeted_train.resolve(),
    ]
    train_parts = [
        load_grid_arrays(path, include_legacy_features=False)
        for path in train_paths
    ]
    train = concatenate_direction_arrays(train_parts)
    del train_parts
    old_path = args.old_validation.resolve()
    difficult_path = args.difficult_validation.resolve()
    old = load_grid_arrays(old_path, include_legacy_features=False)
    difficult = load_grid_arrays(difficult_path, include_legacy_features=False)

    train_context = group_view(train.features, train.group_count)[:, 0, :17]
    old_context = group_view(old.features, old.group_count)[:, 0, :17]
    difficult_context = group_view(
        difficult.features,
        difficult.group_count,
    )[:, 0, :17]
    train_targets = group_view(train.normalized_changes, train.group_count)
    old_targets = group_view(old.normalized_changes, old.group_count)
    difficult_targets = group_view(
        difficult.normalized_changes,
        difficult.group_count,
    )

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_context).astype(np.float32)
    scaled_validation = scaler.transform(
        np.concatenate([old_context, difficult_context], axis=0)
    ).astype(np.float32)
    search = NearestNeighbors(
        n_neighbors=max(neighbors),
        algorithm="auto",
        metric="euclidean",
        n_jobs=2,
    )
    search.fit(scaled_train)
    distances, indices = search.kneighbors(
        scaled_validation,
        return_distance=True,
    )
    old_count = old.group_count
    candidates = []
    for neighbor_count in neighbors:
        for power in powers:
            prediction = interpolate(
                train_targets,
                indices,
                distances,
                neighbor_count,
                power,
            )
            old_prediction = prediction[:old_count]
            difficult_prediction = prediction[old_count:]
            old_metrics = metrics(old_targets, old_prediction)
            difficult_metrics = metrics(
                difficult_targets,
                difficult_prediction,
            )
            candidate = {
                "neighbors": int(neighbor_count),
                "distance_power": float(power),
                "old_iid": old_metrics,
                "difficult": difficult_metrics,
            }
            candidate["selection_key"] = [
                min(
                    old_metrics["direction_all_five_exact"],
                    difficult_metrics["direction_all_five_exact"],
                ),
                (
                    old_metrics["direction_all_five_exact"]
                    + difficult_metrics["direction_all_five_exact"]
                ),
                min(
                    old_metrics["forward_strict_success"],
                    difficult_metrics["forward_strict_success"],
                ),
                (
                    old_metrics["forward_strict_success"]
                    + difficult_metrics["forward_strict_success"]
                ),
                -float(neighbor_count),
                -float(power),
            ]
            candidates.append(candidate)
            print(
                json.dumps(
                    {
                        "neighbors": neighbor_count,
                        "power": power,
                        "old_direction": old_metrics["direction_all_five_exact"],
                        "difficult_direction": difficult_metrics[
                            "direction_all_five_exact"
                        ],
                        "old_forward": old_metrics["forward_strict_success"],
                        "difficult_forward": difficult_metrics[
                            "forward_strict_success"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    selected = max(candidates, key=lambda item: tuple(item["selection_key"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_context=train_context.astype(np.float32),
        train_targets=train_targets.astype(np.float32),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
        neighbors=np.asarray(int(selected["neighbors"]), dtype=np.int32),
        distance_power=np.asarray(
            float(selected["distance_power"]),
            dtype=np.float32,
        ),
    )
    report = {
        "version": "local_physics_interpolator_v9_one_seed",
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "seed": int(args.seed),
        "feature_contract": (
            "first 17 engineered features: setup 12 plus log-peak state 5; "
            "one neighbor search per setup, fixed action index preserved"
        ),
        "training_group_count": int(train.group_count),
        "training_transition_count": int(train.transition_count),
        "candidates": candidates,
        "selected": selected,
        "seconds": time.perf_counter() - started,
        "source_contract": {
            "training_files": [str(path) for path in train_paths],
            "old_validation": str(old_path),
            "difficult_validation": str(difficult_path),
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
