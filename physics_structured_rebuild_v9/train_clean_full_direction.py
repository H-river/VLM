#!/usr/bin/env python3
"""Train five clean LightGBM direction heads with group-held confirmation."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v9.train_boundary_direction_correction import (
    group_partitions,
    rows,
)
from physics_structured_rebuild_v9.train_forward_tree_residual import sha256
from specialist_rebuild_v2.common import DIRECTION_FIELDS

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
DEFAULT_CACHE = DEFAULT_RUN / "clean_full_direction_features_v9.npz"
DEFAULT_OUTPUT = DEFAULT_RUN / "clean_full_direction_v9.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--estimators", type=int, default=400)
    return parser.parse_args()


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    correct = prediction == target
    exact = np.all(correct, axis=1)
    return {
        "count": int(len(exact)),
        "all_five_count": int(exact.sum()),
        "all_five_exact": float(exact.mean()),
        "per_field_accuracy": correct.mean(axis=0).tolist(),
    }


def predict(models: list[Any], features: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [model.predict(features) for model in models]
    ).astype(np.int8)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from lightgbm import LGBMClassifier

    started = time.perf_counter()
    with np.load(args.cache.resolve(), allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float32)
        labels = np.asarray(cache["labels"], dtype=np.int8)
        normalized = np.asarray(
            cache["normalized_changes"], dtype=np.float32
        )
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    training_groups, calibration_groups, confirmation_groups = group_partitions(
        group_ids, int(args.seed)
    )
    training = rows(training_groups)
    calibration = rows(calibration_groups)
    confirmation = rows(confirmation_groups)
    models = []
    trace = []
    for field, name in enumerate(DIRECTION_FIELDS):
        target = labels[training, field].astype(np.int64)
        class_count = np.bincount(target, minlength=3).astype(np.float64)
        class_weight = (len(target) / (3.0 * class_count))[target]
        boundary_distance = np.abs(
            np.abs(normalized[training, field]) - 1.0
        )
        sample_weight = class_weight * (
            1.0 + 1.5 * (boundary_distance < 0.30)
        )
        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=int(args.estimators),
            learning_rate=0.035,
            num_leaves=63,
            min_child_samples=45,
            reg_lambda=3.0,
            feature_fraction=0.90,
            bagging_fraction=0.90,
            bagging_freq=1,
            n_jobs=2,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=int(args.seed) + 37 * field,
        )
        field_started = time.perf_counter()
        model.fit(
            features[training],
            target,
            sample_weight=sample_weight,
        )
        record = {
            "field": name,
            "seconds": time.perf_counter() - field_started,
            "calibration_accuracy": float(
                np.mean(
                    model.predict(features[calibration])
                    == labels[calibration, field]
                )
            ),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        trace.append(record)
        models.append(model)
    confirmation_prediction = predict(models, features[confirmation])
    baseline = np.where(
        normalized[confirmation] < -1.0,
        0,
        np.where(normalized[confirmation] > 1.0, 2, 1),
    ).astype(np.int8)
    artifact = {
        "version": "clean_full_direction_v9_one_seed",
        "model": "five_head_clean_full_lightgbm_direction_v9",
        "seed": int(args.seed),
        "models": models,
        "feature_count": int(features.shape[1]),
        "feature_mode": (
            "engineered_46_plus_v7_forward_5_plus_full_basis_forward_5"
        ),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "group_split": {
            "training": int(len(training_groups)),
            "calibration": int(len(calibration_groups)),
            "internal_confirmation": int(len(confirmation_groups)),
        },
        "internal_confirmation": {
            "truth_threshold_sanity": metric(
                baseline, labels[confirmation]
            ),
            "candidate": metric(
                confirmation_prediction, labels[confirmation]
            ),
        },
        "trace": trace,
        "source_contract": {
            "protected_validation_used": False,
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
