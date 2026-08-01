#!/usr/bin/env python3
"""Build setup-group five-fold out-of-fold forward-v5 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import forward_metrics
from control_rebuild_v5.forward_runtime import ZERO_ACTION_INDEX
from direction_rebuild_v4.data import (
    concatenate_direction_arrays,
    load_grid_arrays,
)
from joint_forward_direction_v6.train import source_sample_weights
from specialist_rebuild_v2.common import STATE_FIELDS

DEFAULT_OLD_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_DIFFICULT_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
)
DEFAULT_ADDITIONAL_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
)
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_data/joint_forward_direction_v7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", type=Path, default=DEFAULT_OLD_DATA)
    parser.add_argument(
        "--difficult-data", type=Path, default=DEFAULT_DIFFICULT_DATA
    )
    parser.add_argument(
        "--additional-data", type=Path, default=DEFAULT_ADDITIONAL_DATA
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--max-leaf-nodes", type=int, default=127)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--l2-regularization", type=float, default=0.10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_fold_assignments(
    counts: list[int],
    folds: int,
    seed: int,
) -> np.ndarray:
    assignments = []
    for source, count in enumerate(counts):
        rng = np.random.default_rng(seed + 1009 * source)
        order = rng.permutation(count)
        source_assignments = np.empty(count, dtype=np.int8)
        source_assignments[order] = np.arange(count, dtype=np.int64) % folds
        assignments.append(source_assignments)
    return np.concatenate(assignments)


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    args = parse_args()
    if args.folds < 2:
        raise ValueError("at least two folds are required")
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "oof_forward_v5_predictions.npy"
    metadata_path = output_dir / "oof_forward_v5_metadata.json"
    if prediction_path.exists() or metadata_path.exists():
        raise RuntimeError(
            "refusing to overwrite an existing out-of-fold artifact"
        )

    paths = [
        args.old_data.resolve() / "grids/train.jsonl",
        args.difficult_data.resolve() / "grids/train.jsonl",
        args.additional_data.resolve() / "grids/train.jsonl",
    ]
    parts = [
        load_grid_arrays(path, include_legacy_features=False)
        for path in paths
    ]
    group_counts = [part.group_count for part in parts]
    arrays = concatenate_direction_arrays(parts)
    del parts
    group_folds = group_fold_assignments(
        group_counts,
        int(args.folds),
        int(args.seed),
    )
    transition_folds = np.repeat(group_folds, len(ACTION_GRID))
    base_weights = source_sample_weights(
        group_counts,
        [0.30, 0.30, 0.40],
    )
    predictions = np.empty_like(
        arrays.normalized_changes,
        dtype=np.float32,
    )
    trace = []
    for fold in range(int(args.folds)):
        fold_started = time.perf_counter()
        train_mask = transition_folds != fold
        validation_mask = ~train_mask
        train_features = arrays.features[train_mask]
        train_target = arrays.normalized_changes[train_mask]
        train_weight = base_weights[train_mask]
        train_weight *= len(train_weight) / train_weight.sum()
        validation_features = arrays.features[validation_mask]
        field_trace = []
        for field_index, field in enumerate(STATE_FIELDS):
            field_started = time.perf_counter()
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=float(args.learning_rate),
                max_iter=int(args.max_iter),
                max_leaf_nodes=int(args.max_leaf_nodes),
                min_samples_leaf=int(args.min_samples_leaf),
                l2_regularization=float(args.l2_regularization),
                early_stopping=False,
                random_state=int(args.seed) + 101 * fold + field_index,
            )
            model.fit(
                train_features,
                train_target[:, field_index],
                sample_weight=train_weight,
            )
            predictions[validation_mask, field_index] = model.predict(
                validation_features
            ).astype(np.float32)
            record = {
                "field": field,
                "iterations": int(model.n_iter_),
                "seconds": time.perf_counter() - field_started,
            }
            field_trace.append(record)
            print(
                json.dumps({"fold": fold, **record}, sort_keys=True),
                flush=True,
            )
        trace.append(
            {
                "fold": fold,
                "train_groups": int((group_folds != fold).sum()),
                "validation_groups": int((group_folds == fold).sum()),
                "fields": field_trace,
                "seconds": time.perf_counter() - fold_started,
            }
        )
        del train_features, train_target, train_weight, validation_features

    zero = np.arange(len(predictions)) % len(ACTION_GRID) == ZERO_ACTION_INDEX
    predictions[zero] = 0.0
    if not np.isfinite(predictions).all():
        raise RuntimeError("out-of-fold predictions contain non-finite values")
    target_grouped = arrays.normalized_changes.reshape(
        arrays.group_count,
        len(ACTION_GRID),
        len(STATE_FIELDS),
    )
    prediction_grouped = predictions.reshape(target_grouped.shape)
    metrics = forward_metrics(target_grouped, prediction_grouped)
    np.save(prediction_path, predictions, allow_pickle=False)
    metadata = {
        "version": "joint_forward_direction_v7_oof_forward_v5",
        "complete": True,
        "seed": int(args.seed),
        "folds": int(args.folds),
        "split_unit": "complete optical setup group",
        "group_counts": {
            "old_iid": group_counts[0],
            "difficult": group_counts[1],
            "additional_v5": group_counts[2],
            "total": arrays.group_count,
        },
        "transition_count": arrays.transition_count,
        "prediction_shape": list(predictions.shape),
        "prediction_dtype": str(predictions.dtype),
        "prediction_file": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "source_files": [
            {
                "path": str(path),
                "sha256": sha256(path),
            }
            for path in paths
        ],
        "hyperparameters": {
            "max_iter": int(args.max_iter),
            "max_leaf_nodes": int(args.max_leaf_nodes),
            "min_samples_leaf": int(args.min_samples_leaf),
            "learning_rate": float(args.learning_rate),
            "l2_regularization": float(args.l2_regularization),
        },
        "oof_metrics": metrics,
        "trace": trace,
        "held_out_test_files_opened": [],
        "seconds": time.perf_counter() - started,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "predictions": str(prediction_path),
                "metadata": str(metadata_path),
                "oof_strict_all_five": metrics[
                    "strict_all_five_success"
                ],
                "seconds": metadata["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

