#!/usr/bin/env python3
"""Train five boosted residual heads on top of the frozen v7 forward model."""

from __future__ import annotations

import argparse
import hashlib
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

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import (
    concatenate_direction_arrays,
    load_grid_arrays,
)
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from joint_forward_direction_v7.train import forward_metric_bundle
from physics_structured_rebuild_v9.train_direction_tree_targeted import (
    stream_forward_changes,
)
from specialist_rebuild_v2.common import STATE_FIELDS

DEFAULT_OLD_TRAIN = (
    REPO_ROOT.parent
    / "VLM_data/specialist_rebuild_v2/grids/train.jsonl"
)
DEFAULT_DIFFICULT_TRAIN = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/train.jsonl"
)
DEFAULT_TARGETED_TRAIN = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9"
    / "targeted_bundle/grids/train.jsonl"
)
DEFAULT_ADDITIONAL_TRAIN = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v5_numerical/grids/train.jsonl"
)
DEFAULT_OLD_VAL = (
    REPO_ROOT.parent
    / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT_VAL = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_tree_residual"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-train", type=Path, default=DEFAULT_OLD_TRAIN)
    parser.add_argument(
        "--difficult-train",
        type=Path,
        default=DEFAULT_DIFFICULT_TRAIN,
    )
    parser.add_argument(
        "--targeted-train",
        type=Path,
        default=DEFAULT_TARGETED_TRAIN,
    )
    parser.add_argument(
        "--additional-train",
        type=Path,
        help=(
            "Optional fourth training-only grid file. It is omitted by default "
            "to preserve compatibility with completed v9 experiments."
        ),
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD_VAL)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT_VAL,
    )
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument(
        "--natural-adaptation-cache",
        type=Path,
        help=(
            "Optional training-only cache containing state/image natural-request "
            "features and residual targets."
        ),
    )
    parser.add_argument(
        "--natural-adaptation-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--estimator",
        choices=("hist_gradient_boosting", "extra_trees", "lightgbm"),
        default="hist_gradient_boosting",
    )
    parser.add_argument(
        "--loss",
        choices=("squared_error", "absolute_error"),
        default="squared_error",
        help="Loss for hist-gradient-boosting heads; ignored by extra trees.",
    )
    parser.add_argument(
        "--version",
        default="forward_tree_residual_v9_one_seed",
    )
    parser.add_argument(
        "--artifact-name",
        default="forward_tree_residual_v9.pkl",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-iter", type=int, default=320)
    parser.add_argument("--max-leaf-nodes", type=int, default=127)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.20)
    parser.add_argument("--hard-error-weight", type=float, default=1.5)
    parser.add_argument("--extra-tree-count", type=int, default=64)
    parser.add_argument("--extra-tree-max-leaves", type=int, default=8192)
    parser.add_argument(
        "--extra-tree-min-samples-leaf",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--extra-tree-max-features",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_predictions(
    models: list[Any],
    features: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [model.predict(features) for model in models],
        axis=1,
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.max_iter < 1 or args.max_leaf_nodes < 2:
        raise ValueError("tree iteration and leaf counts must be positive")
    if args.natural_adaptation_weight <= 0.0:
        raise ValueError("natural-adaptation-weight must be positive")
    if (
        args.extra_tree_count < 1
        or args.extra_tree_max_leaves < 2
        or args.extra_tree_min_samples_leaf < 1
        or not 0.0 < args.extra_tree_max_features <= 1.0
    ):
        raise ValueError("extra-tree hyperparameters must be positive")
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
    )
    from lightgbm import LGBMRegressor

    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    artifact_path = output_dir / str(args.artifact_name)
    summary_path = output_dir / (
        Path(str(args.artifact_name)).stem + "_summary.json"
    )
    if artifact_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, device = configure(int(args.seed), args.device)
    forward_path = args.forward_artifact.resolve()
    forward, _ = load_forward_direction_runtime_v7(
        forward_path,
        torch,
        device,
    )
    train_paths = [
        args.old_train.resolve(),
        args.difficult_train.resolve(),
        args.targeted_train.resolve(),
    ]
    source_names = ["old_iid", "difficult", "targeted"]
    if args.additional_train is not None:
        train_paths.append(args.additional_train.resolve())
        source_names.append("additional")
    train_parts = [
        load_grid_arrays(path, include_legacy_features=False)
        for path in train_paths
    ]
    group_counts = [int(part.group_count) for part in train_parts]
    train = concatenate_direction_arrays(train_parts)
    del train_parts
    train_prior = np.concatenate(
        [stream_forward_changes(path, forward) for path in train_paths]
    )
    if train_prior.shape != train.normalized_changes.shape:
        raise ValueError("training prior shape differs from forward targets")
    train_features = np.concatenate(
        [train.features, train_prior],
        axis=1,
    ).astype(np.float32)
    residual_target = (
        train.normalized_changes - train_prior
    ).astype(np.float32)
    domain_weight = np.ones(len(train_features), dtype=np.float32)
    natural_count = 0
    natural_path = (
        None
        if args.natural_adaptation_cache is None
        else args.natural_adaptation_cache.resolve()
    )
    if natural_path is not None:
        natural = np.load(natural_path, allow_pickle=False)
        if "grid_features" in natural.files:
            natural_features = np.asarray(
                natural["grid_features"],
                dtype=np.float32,
            )
            natural_targets = np.asarray(
                natural["grid_residual_target"],
                dtype=np.float32,
            )
        else:
            natural_features = np.concatenate(
                [
                    np.asarray(natural["state_features"], dtype=np.float32),
                    np.asarray(natural["image_features"], dtype=np.float32),
                ],
                axis=0,
            )
            natural_targets = np.concatenate(
                [
                    np.asarray(
                        natural["state_residual_target"],
                        dtype=np.float32,
                    ),
                    np.asarray(
                        natural["image_residual_target"],
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            )
        if natural_features.shape[1:] != train_features.shape[1:]:
            raise ValueError("natural adaptation feature shape differs")
        if natural_targets.shape[1:] != residual_target.shape[1:]:
            raise ValueError("natural adaptation target shape differs")
        natural_count = len(natural_features)
        train_features = np.concatenate(
            [train_features, natural_features],
            axis=0,
        )
        residual_target = np.concatenate(
            [residual_target, natural_targets],
            axis=0,
        )
        domain_weight = np.concatenate(
            [
                domain_weight,
                np.full(
                    natural_count,
                    float(args.natural_adaptation_weight),
                    dtype=np.float32,
                ),
            ]
        )
    joint_error = np.max(np.abs(residual_target), axis=1)

    old_path = args.old_validation.resolve()
    difficult_path = args.difficult_validation.resolve()
    old_arrays = load_grid_arrays(
        old_path,
        include_legacy_features=False,
    )
    difficult_arrays = load_grid_arrays(
        difficult_path,
        include_legacy_features=False,
    )
    old_prior = stream_forward_changes(old_path, forward)
    difficult_prior = stream_forward_changes(difficult_path, forward)
    old_features = np.concatenate(
        [old_arrays.features, old_prior],
        axis=1,
    )
    difficult_features = np.concatenate(
        [difficult_arrays.features, difficult_prior],
        axis=1,
    )

    models = []
    trace = []
    for field_index, field in enumerate(STATE_FIELDS):
        field_error = np.abs(residual_target[:, field_index])
        sample_weight = (
            1.0
            + float(args.hard_error_weight) * (joint_error > 1.0)
            + 0.35 * np.minimum(field_error, 4.0)
        ).astype(np.float32) * domain_weight
        if args.estimator == "hist_gradient_boosting":
            model = HistGradientBoostingRegressor(
                loss=str(args.loss),
                learning_rate=float(args.learning_rate),
                max_iter=int(args.max_iter),
                max_leaf_nodes=int(args.max_leaf_nodes),
                min_samples_leaf=int(args.min_samples_leaf),
                l2_regularization=float(args.l2_regularization),
                early_stopping=True,
                validation_fraction=0.08,
                n_iter_no_change=24,
                tol=1e-6,
                random_state=int(args.seed) + 41 * field_index,
            )
        elif args.estimator == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=int(args.extra_tree_count),
                max_leaf_nodes=int(args.extra_tree_max_leaves),
                min_samples_leaf=int(args.extra_tree_min_samples_leaf),
                max_features=float(args.extra_tree_max_features),
                n_jobs=2,
                random_state=int(args.seed) + 41 * field_index,
            )
        else:
            model = LGBMRegressor(
                objective="regression",
                n_estimators=int(args.max_iter),
                learning_rate=float(args.learning_rate),
                num_leaves=int(args.max_leaf_nodes),
                min_child_samples=int(args.min_samples_leaf),
                reg_lambda=float(args.l2_regularization),
                max_bin=255,
                feature_fraction=0.90,
                bagging_fraction=0.90,
                bagging_freq=1,
                n_jobs=2,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
                random_state=int(args.seed) + 41 * field_index,
            )
        field_started = time.perf_counter()
        model.fit(
            train_features,
            residual_target[:, field_index],
            sample_weight=sample_weight,
        )
        predicted = model.predict(train_features)
        record = {
            "field": field,
            "iterations": int(
                model.n_iter_
                if hasattr(model, "n_iter_")
                else model.n_estimators_
                if hasattr(model, "n_estimators_")
                else len(model.estimators_)
            ),
            "seconds": time.perf_counter() - field_started,
            "train_residual_mae": float(
                np.mean(
                    np.abs(
                        predicted - residual_target[:, field_index]
                    )
                )
            ),
        }
        trace.append(record)
        models.append(model)
        print(json.dumps(record, sort_keys=True), flush=True)

    old_residual = tree_predictions(models, old_features)
    difficult_residual = tree_predictions(models, difficult_features)
    blends = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
    validation_candidates = []
    for blend in blends:
        old_prediction = old_prior + float(blend) * old_residual
        difficult_prediction = (
            difficult_prior + float(blend) * difficult_residual
        )
        validation_candidates.append(
            {
                "blend": float(blend),
                "old_iid": forward_metric_bundle(
                    old_arrays,
                    old_prediction,
                ),
                "difficult": forward_metric_bundle(
                    difficult_arrays,
                    difficult_prediction,
                ),
            }
        )

    model_name = (
        "v7_stacked_five_head_hist_gradient_boosting_forward_v9"
        if args.estimator == "hist_gradient_boosting"
        else "v7_stacked_five_head_extra_trees_forward_v9"
        if args.estimator == "extra_trees"
        else "v7_stacked_five_head_lightgbm_forward_v9"
    )
    artifact = {
        "version": str(args.version),
        "model": model_name,
        "seed": int(args.seed),
        "input_dim": int(train_features.shape[1]),
        "state_fields": list(STATE_FIELDS),
        "models": models,
        "feature_mode": "engineered_46_plus_forward_v7_change_5",
        "forward_artifact": str(forward_path),
        "forward_artifact_sha256": sha256(forward_path),
        "held_out_test_used": False,
    }
    with artifact_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "model": artifact["model"],
        "input_dim": artifact["input_dim"],
        "feature_mode": artifact["feature_mode"],
        "seed": int(args.seed),
        "training": {
            "source_group_counts": dict(
                zip(
                    source_names,
                    group_counts,
                    strict=True,
                )
            ),
            "total_groups": int(train.group_count),
            "total_transitions": int(train.transition_count),
            "natural_adaptation_cache": (
                None if natural_path is None else str(natural_path)
            ),
            "natural_adaptation_view_count": int(natural_count),
            "natural_adaptation_weight": float(
                args.natural_adaptation_weight
            ),
            "hard_error_weight": float(args.hard_error_weight),
            "trace": trace,
        },
        "validation_candidates": validation_candidates,
        "hyperparameters": {
            "estimator": str(args.estimator),
            "loss": str(args.loss),
            "max_iter": int(args.max_iter),
            "max_leaf_nodes": int(args.max_leaf_nodes),
            "min_samples_leaf": int(args.min_samples_leaf),
            "learning_rate": float(args.learning_rate),
            "l2_regularization": float(args.l2_regularization),
            "extra_tree_count": int(args.extra_tree_count),
            "extra_tree_max_leaves": int(args.extra_tree_max_leaves),
            "extra_tree_min_samples_leaf": int(
                args.extra_tree_min_samples_leaf
            ),
            "extra_tree_max_features": float(
                args.extra_tree_max_features
            ),
        },
        "source_contract": {
            "training_files": [str(path) for path in train_paths],
            "natural_adaptation_cache": (
                None if natural_path is None else str(natural_path)
            ),
            "old_validation": str(old_path),
            "difficult_validation": str(difficult_path),
            "forward_artifact": str(forward_path),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "summary": str(summary_path),
                "validation": {
                    str(row["blend"]): {
                        "old_iid": row["old_iid"]["overall"][
                            "strict_all_five_success"
                        ],
                        "difficult": row["difficult"]["overall"][
                            "strict_all_five_success"
                        ],
                        "old_high": row["old_iid"][
                            "by_action_complexity"
                        ]["three_or_four"]["strict_all_five_success"],
                        "difficult_high": row["difficult"][
                            "by_action_complexity"
                        ]["three_or_four"]["strict_all_five_success"],
                    }
                    for row in validation_candidates
                },
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
