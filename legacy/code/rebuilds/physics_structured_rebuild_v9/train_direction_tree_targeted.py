#!/usr/bin/env python3
"""Train a v7-change-aware direction tree on existing plus targeted data."""

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
    DirectionArrays,
    balance_table,
    category_metrics,
    class_count_summary,
    concatenate_direction_arrays,
    distance_bins,
    direction_metrics,
    labels_from_normalized_change,
    load_grid_arrays,
)
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS

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
    / "direction_tree_targeted"
)
DEFAULT_RESIDUAL_FORWARD_TREE = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_tree_residual/forward_tree_residual_v9.pkl"
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
        help="Optional training-only 81-action natural-grid feature cache.",
    )
    parser.add_argument(
        "--natural-adaptation-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--residual-forward-tree",
        type=Path,
        help="Optional residual-forward tree supplying five extra features.",
    )
    parser.add_argument(
        "--version",
        default="direction_tree_targeted_v9_one_seed",
    )
    parser.add_argument(
        "--artifact-name",
        default="direction_tree_targeted_v9.pkl",
    )
    parser.add_argument(
        "--estimator",
        choices=("hist_gradient_boosting", "extra_trees", "lightgbm"),
        default="hist_gradient_boosting",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-iter", type=int, default=360)
    parser.add_argument("--max-leaf-nodes", type=int, default=127)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2-regularization", type=float, default=0.20)
    parser.add_argument("--balance-strength", type=float, default=0.15)
    parser.add_argument("--extra-tree-count", type=int, default=96)
    parser.add_argument("--extra-tree-max-leaves", type=int, default=8192)
    parser.add_argument(
        "--extra-tree-min-samples-leaf",
        type=int,
        default=2,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_forward_changes(
    path: Path,
    runtime: Any,
    *,
    batch_groups: int = 32,
) -> np.ndarray:
    outputs = []
    batch = []

    def flush() -> None:
        if not batch:
            return
        predicted = runtime.predict_changes(batch)
        outputs.append(
            predicted.reshape(-1, predicted.shape[-1]).astype(np.float32)
        )
        batch.clear()

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) == batch_groups:
                flush()
    flush()
    if not outputs:
        raise ValueError(f"no groups found in {path}")
    return np.concatenate(outputs)


def with_forward_features(
    arrays: DirectionArrays,
    changes: np.ndarray,
) -> np.ndarray:
    if changes.shape != (arrays.transition_count, 5):
        raise ValueError(
            "v7 forward-change features do not match transition count"
        )
    return np.concatenate(
        [
            arrays.features,
            np.asarray(changes, dtype=np.float32),
        ],
        axis=1,
    )


def model_predictions(
    models: list[Any],
    features: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [model.predict(features) for model in models],
        axis=1,
    ).astype(np.int64)


def regression_predictions(
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
    if (
        args.extra_tree_count < 1
        or args.extra_tree_max_leaves < 2
        or args.extra_tree_min_samples_leaf < 1
    ):
        raise ValueError("extra-tree count and leaf limit must be positive")
    if not 0.0 <= args.balance_strength <= 1.0:
        raise ValueError("--balance-strength must be between zero and one")
    if args.natural_adaptation_weight <= 0.0:
        raise ValueError("natural-adaptation-weight must be positive")
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
    )
    from lightgbm import LGBMClassifier

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
    forward_runtime, _ = load_forward_direction_runtime_v7(
        forward_path,
        torch,
        device,
    )

    train_paths = [
        args.old_train.resolve(),
        args.difficult_train.resolve(),
        args.targeted_train.resolve(),
    ]
    train_parts = [
        load_grid_arrays(path, include_legacy_features=False)
        for path in train_paths
    ]
    train_group_counts = [int(part.group_count) for part in train_parts]
    train = concatenate_direction_arrays(train_parts)
    del train_parts
    train_changes = np.concatenate(
        [
            stream_forward_changes(path, forward_runtime)
            for path in train_paths
        ]
    )
    train_features = with_forward_features(train, train_changes)
    del train_changes
    train_labels = np.asarray(train.labels, dtype=np.int64)
    train_distance_bins = np.asarray(train.distance_bins, dtype=np.int64)
    domain_weight = np.ones(len(train_features), dtype=np.float32)
    natural_path = (
        None
        if args.natural_adaptation_cache is None
        else args.natural_adaptation_cache.resolve()
    )
    natural_count = 0
    if natural_path is not None:
        natural = np.load(natural_path, allow_pickle=False)
        natural_features = np.asarray(
            natural["grid_features"],
            dtype=np.float32,
        )
        natural_normalized = np.asarray(
            natural["grid_target_normalized"],
            dtype=np.float32,
        )
        if natural_features.shape[1:] != train_features.shape[1:]:
            raise ValueError("natural direction feature shape differs")
        natural_labels = labels_from_normalized_change(natural_normalized)
        natural_bins = distance_bins(natural_normalized)
        natural_count = len(natural_features)
        train_features = np.concatenate(
            [train_features, natural_features],
            axis=0,
        )
        train_labels = np.concatenate(
            [train_labels, natural_labels],
            axis=0,
        )
        train_distance_bins = np.concatenate(
            [train_distance_bins, natural_bins],
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
    residual_forward_path = (
        None
        if args.residual_forward_tree is None
        else args.residual_forward_tree.resolve()
    )
    residual_forward_models = None
    if residual_forward_path is not None:
        with residual_forward_path.open("rb") as stream:
            residual_forward_artifact = pickle.load(stream)
        if (
            residual_forward_artifact.get("model")
            != "v7_stacked_five_head_hist_gradient_boosting_forward_v9"
        ):
            raise ValueError("unexpected residual-forward tree artifact")
        residual_forward_models = list(
            residual_forward_artifact["models"]
        )
        train_features = np.concatenate(
            [
                train_features,
                regression_predictions(
                    residual_forward_models,
                    train_features,
                ),
            ],
            axis=1,
        )

    old_val_path = args.old_validation.resolve()
    difficult_val_path = args.difficult_validation.resolve()
    old_val = load_grid_arrays(
        old_val_path,
        include_legacy_features=False,
    )
    difficult_val = load_grid_arrays(
        difficult_val_path,
        include_legacy_features=False,
    )
    old_val_features = with_forward_features(
        old_val,
        stream_forward_changes(old_val_path, forward_runtime),
    )
    difficult_val_features = with_forward_features(
        difficult_val,
        stream_forward_changes(difficult_val_path, forward_runtime),
    )
    if residual_forward_models is not None:
        old_val_features = np.concatenate(
            [
                old_val_features,
                regression_predictions(
                    residual_forward_models,
                    old_val_features,
                ),
            ],
            axis=1,
        )
        difficult_val_features = np.concatenate(
            [
                difficult_val_features,
                regression_predictions(
                    residual_forward_models,
                    difficult_val_features,
                ),
            ],
            axis=1,
        )

    balance, balance_report = balance_table(
        train_labels,
        train_distance_bins,
    )
    models = []
    trace = []
    for field_index, field in enumerate(DIRECTION_FIELDS):
        stratum_weight = balance[
            field_index,
            train_labels[:, field_index],
            train_distance_bins[:, field_index],
        ]
        sample_weight = (
            float(args.balance_strength) * stratum_weight
            + (1.0 - float(args.balance_strength))
        ) * domain_weight
        if args.estimator == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=float(args.learning_rate),
                max_iter=int(args.max_iter),
                max_leaf_nodes=int(args.max_leaf_nodes),
                min_samples_leaf=int(args.min_samples_leaf),
                l2_regularization=float(args.l2_regularization),
                early_stopping=True,
                validation_fraction=0.08,
                n_iter_no_change=24,
                tol=1e-6,
                random_state=int(args.seed) + 31 * field_index,
            )
        elif args.estimator == "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=int(args.extra_tree_count),
                max_leaf_nodes=int(args.extra_tree_max_leaves),
                min_samples_leaf=int(args.extra_tree_min_samples_leaf),
                max_features=1.0,
                n_jobs=2,
                random_state=int(args.seed) + 31 * field_index,
            )
        else:
            model = LGBMClassifier(
                objective="multiclass",
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
                random_state=int(args.seed) + 31 * field_index,
            )
        field_started = time.perf_counter()
        model.fit(
            train_features,
            train_labels[:, field_index],
            sample_weight=sample_weight,
        )
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
            "train_accuracy": float(
                np.mean(
                    model.predict(train_features)
                    == train_labels[:, field_index]
                )
            ),
        }
        trace.append(record)
        models.append(model)
        print(json.dumps(record, sort_keys=True), flush=True)

    old_prediction = model_predictions(models, old_val_features)
    difficult_prediction = model_predictions(
        models,
        difficult_val_features,
    )
    old_metrics = direction_metrics(
        old_val.labels,
        old_prediction,
        old_val.distance_bins,
    )
    difficult_metrics = direction_metrics(
        difficult_val.labels,
        difficult_prediction,
        difficult_val.distance_bins,
    )
    model_name = (
        "balanced_five_head_hist_gradient_boosting_direction_v4"
        if args.estimator == "hist_gradient_boosting"
        else "balanced_five_head_extra_trees_direction_v9"
        if args.estimator == "extra_trees"
        else "balanced_five_head_lightgbm_direction_v9"
    )
    artifact = {
        "version": str(args.version),
        "model": model_name,
        "seed": int(args.seed),
        "input_dim": int(train_features.shape[1]),
        "direction_fields": list(DIRECTION_FIELDS),
        "classes": ["decrease", "no_change", "increase"],
        "models": models,
        "feature_mode": (
            "engineered_46_plus_forward_v7_change_5"
            if residual_forward_path is None
            else (
                "engineered_46_plus_forward_v7_change_5"
                "_plus_residual_forward_5"
            )
        ),
        "forward_artifact": str(forward_path),
        "forward_artifact_sha256": sha256(forward_path),
        "residual_forward_tree_artifact": (
            None
            if residual_forward_path is None
            else str(residual_forward_path)
        ),
        "residual_forward_tree_artifact_sha256": (
            None
            if residual_forward_path is None
            else sha256(residual_forward_path)
        ),
        "balance_contract": balance_report,
        "held_out_test_used": False,
    }
    with artifact_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "model": artifact["model"],
        "feature_mode": artifact["feature_mode"],
        "input_dim": artifact["input_dim"],
        "seed": int(args.seed),
        "training": {
            "source_group_counts": dict(
                zip(
                    ("old_iid", "difficult", "targeted"),
                    train_group_counts,
                    strict=True,
                )
            ),
            "total_groups": int(train.group_count),
            "total_transitions": int(train.transition_count),
            "natural_adaptation_cache": (
                None if natural_path is None else str(natural_path)
            ),
            "natural_adaptation_transition_count": int(natural_count),
            "natural_adaptation_weight": float(
                args.natural_adaptation_weight
            ),
            "class_counts": class_count_summary(train_labels),
            "balance_strength": float(args.balance_strength),
            "trace": trace,
        },
        "validation": {
            "old_iid": old_metrics,
            "difficult": difficult_metrics,
            "difficult_by_category": category_metrics(
                difficult_val,
                difficult_prediction,
            ),
        },
        "hyperparameters": {
            "estimator": str(args.estimator),
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
        },
        "source_contract": {
            "training_files": [str(path) for path in train_paths],
            "natural_adaptation_cache": (
                None if natural_path is None else str(natural_path)
            ),
            "old_validation": str(old_val_path),
            "difficult_validation": str(difficult_val_path),
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
                "old_iid_joint_exact": old_metrics["joint_exact"],
                "difficult_joint_exact": difficult_metrics["joint_exact"],
                "old_iid_macro_f1": old_metrics[
                    "equal_field_macro_f1"
                ],
                "difficult_macro_f1": difficult_metrics[
                    "equal_field_macro_f1"
                ],
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
