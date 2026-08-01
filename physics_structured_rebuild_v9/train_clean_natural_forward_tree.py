#!/usr/bin/env python3
"""Train residual trees only on non-overlapping existing natural groups."""

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
from specialist_rebuild_v2.common import STATE_FIELDS, read_jsonl

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "clean_nonoverlap_forward_training_features.npz"
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_EXCLUSION = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "clean_nonoverlap_forward_tree_v9.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--exclusion-data", type=Path, default=DEFAULT_EXCLUSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=30)
    return parser.parse_args()


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.abs(prediction - target)
    passed = error <= 1.0
    exact = np.all(passed, axis=1)
    return {
        "count": int(len(exact)),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "mae_in_tolerance_units": float(error.mean()),
        "per_field_tolerance_pass": passed.mean(axis=0).tolist(),
    }


def predictions(models: list[Any], features: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [model.predict(features) for model in models]
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from lightgbm import LGBMRegressor

    started = time.perf_counter()
    cache_path = args.cache.resolve()
    with np.load(cache_path, allow_pickle=False) as cache:
        features = np.asarray(cache["grid_features"], dtype=np.float32)
        prior = np.asarray(cache["grid_base_prediction"], dtype=np.float32)
        target = np.asarray(
            cache["grid_target_normalized"], dtype=np.float32
        )
        residual_target = np.asarray(
            cache["grid_residual_target"], dtype=np.float32
        )
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    exclusion_path = args.exclusion_data.resolve()
    excluded_groups = {
        str(row["group_id"]) for row in read_jsonl(exclusion_path)
    }
    overlap = excluded_groups & set(map(str, group_ids))
    if overlap:
        raise ValueError(
            f"clean natural cache overlaps {len(overlap)} excluded groups"
        )
    train_groups, _, confirmation_groups = group_partitions(
        group_ids, int(args.seed)
    )
    train_indices = rows(
        np.sort(
            np.concatenate(
                [
                    train_groups,
                    group_partitions(group_ids, int(args.seed))[1],
                ]
            )
        )
    )
    confirmation_indices = rows(confirmation_groups)
    joint_residual = np.max(np.abs(residual_target[train_indices]), axis=1)
    models = []
    trace = []
    for field, name in enumerate(STATE_FIELDS):
        field_residual = np.abs(residual_target[train_indices, field])
        sample_weight = (
            1.0
            + 2.0 * (joint_residual > 1.0).astype(np.float32)
            + 0.5 * np.minimum(field_residual, 4.0)
        )
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=int(args.estimators),
            learning_rate=float(args.learning_rate),
            num_leaves=int(args.num_leaves),
            min_child_samples=int(args.min_child_samples),
            reg_lambda=2.0,
            max_bin=255,
            feature_fraction=0.90,
            bagging_fraction=0.90,
            bagging_freq=1,
            n_jobs=2,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=int(args.seed) + 43 * field,
        )
        field_started = time.perf_counter()
        model.fit(
            features[train_indices],
            residual_target[train_indices, field],
            sample_weight=sample_weight,
        )
        record = {
            "field": name,
            "seconds": time.perf_counter() - field_started,
            "train_residual_mae": float(
                np.mean(
                    np.abs(
                        model.predict(features[train_indices])
                        - residual_target[train_indices, field]
                    )
                )
            ),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        trace.append(record)
        models.append(model)
    confirmation_prediction = (
        prior[confirmation_indices]
        + predictions(models, features[confirmation_indices])
    )
    artifact = {
        "version": "clean_nonoverlap_forward_tree_v9_one_seed",
        "model": "v7_stacked_five_head_lightgbm_forward_v9",
        "seed": int(args.seed),
        "input_dim": int(features.shape[1]),
        "state_fields": list(STATE_FIELDS),
        "models": models,
        "feature_mode": "engineered_46_plus_forward_v7_change_5",
        "forward_artifact": str(args.forward_artifact.resolve()),
        "forward_artifact_sha256": sha256(
            args.forward_artifact.resolve()
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
        "training": {
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "group_count": int(len(group_ids)),
            "train_group_count": int(len(train_indices) // 81),
            "internal_confirmation_group_count": int(
                len(confirmation_groups)
            ),
            "qwen_direct_group_overlap": 0,
            "trace": trace,
        },
        "internal_confirmation": {
            "base": metric(
                prior[confirmation_indices], target[confirmation_indices]
            ),
            "candidate": metric(
                confirmation_prediction, target[confirmation_indices]
            ),
        },
        "source_contract": {
            "generated_setups": 0,
            "generated_images": 0,
            "exclusion_data": str(exclusion_path),
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
