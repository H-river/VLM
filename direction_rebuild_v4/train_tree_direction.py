#!/usr/bin/env python3
"""Train five balanced histogram-gradient direction classifiers."""

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

from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from direction_rebuild_v4.data import (
    DirectionArrays,
    balance_table,
    category_metrics,
    class_count_summary,
    concatenate_direction_arrays,
    direction_metrics,
    labels_from_normalized_change,
    load_grid_arrays,
)
from direction_rebuild_v4.train_direction import (
    DEFAULT_CONTROL_RUN,
    DEFAULT_NEW_DATA,
    DEFAULT_OLD_DATA,
    forward_changes,
    frozen_v1_predictions,
    gate_summary,
    sha256,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS

DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_runs/direction_rebuild_v4_tree_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", type=Path, default=DEFAULT_OLD_DATA)
    parser.add_argument("--new-data", type=Path, default=DEFAULT_NEW_DATA)
    parser.add_argument("--control-run", type=Path, default=DEFAULT_CONTROL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-iter", type=int, default=240)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2-regularization", type=float, default=0.10)
    parser.add_argument("--balance-strength", type=float, default=0.20)
    parser.add_argument("--max-old-train-groups", type=int)
    parser.add_argument("--max-new-train-groups", type=int)
    parser.add_argument("--max-old-val-groups", type=int)
    parser.add_argument("--max-new-val-groups", type=int)
    return parser.parse_args()


def model_predictions(
    models: list[Any],
    arrays: DirectionArrays,
) -> np.ndarray:
    return np.stack(
        [model.predict(arrays.features) for model in models],
        axis=1,
    ).astype(np.int64)


def main() -> None:
    args = parse_args()
    if args.max_iter < 1 or args.max_leaf_nodes < 2:
        raise ValueError("tree iteration and leaf counts must be positive")
    if not 0.0 <= args.balance_strength <= 1.0:
        raise ValueError("--balance-strength must be between zero and one")
    from sklearn.ensemble import HistGradientBoostingClassifier

    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device = configure(args.seed, args.device)

    old_train = load_grid_arrays(
        args.old_data.resolve() / "grids/train.jsonl",
        include_legacy_features=False,
        max_groups=args.max_old_train_groups,
    )
    new_train = load_grid_arrays(
        args.new_data.resolve() / "grids/train.jsonl",
        include_legacy_features=False,
        max_groups=args.max_new_train_groups,
    )
    old_val = load_grid_arrays(
        args.old_data.resolve() / "grids/val.jsonl",
        include_legacy_features=True,
        max_groups=args.max_old_val_groups,
    )
    new_val = load_grid_arrays(
        args.new_data.resolve() / "grids/val.jsonl",
        include_legacy_features=True,
        max_groups=args.max_new_val_groups,
    )
    old_train_group_count = old_train.group_count
    new_train_group_count = new_train.group_count
    train = concatenate_direction_arrays([old_train, new_train])
    del old_train, new_train
    balance, balance_report = balance_table(
        train.labels,
        train.distance_bins,
    )

    models = []
    training_trace = []
    for field_index, field in enumerate(DIRECTION_FIELDS):
        stratum_weight = balance[
            field_index,
            train.labels[:, field_index],
            train.distance_bins[:, field_index],
        ]
        sample_weight = (
            args.balance_strength * stratum_weight
            + (1.0 - args.balance_strength)
        )
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            early_stopping=True,
            validation_fraction=0.08,
            n_iter_no_change=20,
            tol=1e-6,
            random_state=args.seed + 31 * field_index,
        )
        field_started = time.perf_counter()
        model.fit(
            train.features,
            train.labels[:, field_index],
            sample_weight=sample_weight,
        )
        record = {
            "field": field,
            "iterations": int(model.n_iter_),
            "seconds": time.perf_counter() - field_started,
            "train_accuracy": float(
                np.mean(
                    model.predict(train.features)
                    == train.labels[:, field_index]
                )
            ),
        }
        training_trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        models.append(model)

    old_predictions = model_predictions(models, old_val)
    new_predictions = model_predictions(models, new_val)
    old_metrics = direction_metrics(
        old_val.labels,
        old_predictions,
        old_val.distance_bins,
    )
    new_metrics = direction_metrics(
        new_val.labels,
        new_predictions,
        new_val.distance_bins,
    )
    frozen_old_metrics = direction_metrics(
        old_val.labels,
        frozen_v1_predictions(old_val),
        old_val.distance_bins,
    )
    frozen_new_metrics = direction_metrics(
        new_val.labels,
        frozen_v1_predictions(new_val),
        new_val.distance_bins,
    )
    forward_runtime, _ = load_forward_runtime_v4(
        args.control_run.resolve() / "forward_physics_residual_v4.pt",
        torch,
        device,
    )
    forward_old_metrics = direction_metrics(
        old_val.labels,
        labels_from_normalized_change(
            forward_changes(
                args.old_data.resolve() / "grids/val.jsonl",
                args.max_old_val_groups,
                forward_runtime,
            )
        ),
        old_val.distance_bins,
    )
    forward_new_metrics = direction_metrics(
        new_val.labels,
        labels_from_normalized_change(
            forward_changes(
                args.new_data.resolve() / "grids/val.jsonl",
                args.max_new_val_groups,
                forward_runtime,
            )
        ),
        new_val.distance_bins,
    )
    gates = gate_summary(
        old_metrics,
        new_metrics,
        frozen_old_metrics,
        frozen_new_metrics,
    )
    artifact = {
        "version": "direction_rebuild_v4_tree_one_seed",
        "model": "balanced_five_head_hist_gradient_boosting_direction_v4",
        "seed": int(args.seed),
        "input_dim": int(train.features.shape[1]),
        "direction_fields": list(DIRECTION_FIELDS),
        "classes": ["decrease", "no_change", "increase"],
        "models": models,
        "feature_mode": "engineered_46",
        "balance_contract": balance_report,
        "held_out_test_used": False,
    }
    artifact_path = output_dir / "direction_tree_v4.pkl"
    with artifact_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "seed": int(args.seed),
        "model": artifact["model"],
        "input_dim": artifact["input_dim"],
        "tree_count": int(sum(model.n_iter_ for model in models)),
        "training": {
            "old_iid_groups": old_train_group_count,
            "difficult_groups": new_train_group_count,
            "total_groups": train.group_count,
            "total_transitions": train.transition_count,
            "class_counts": class_count_summary(train.labels),
            "balance_strength": float(args.balance_strength),
            "balance_contract": balance_report,
            "trace": training_trace,
        },
        "validation": {
            "frozen_v1": {
                "old_iid": frozen_old_metrics,
                "difficult": frozen_new_metrics,
            },
            "thresholded_forward_v4": {
                "old_iid": forward_old_metrics,
                "difficult": forward_new_metrics,
            },
            "direction_tree_v4": {
                "old_iid": old_metrics,
                "difficult": new_metrics,
                "difficult_by_category": category_metrics(
                    new_val,
                    new_predictions,
                ),
            },
            "acceptance_gates": gates,
        },
        "hyperparameters": {
            "max_iter": int(args.max_iter),
            "max_leaf_nodes": int(args.max_leaf_nodes),
            "min_samples_leaf": int(args.min_samples_leaf),
            "learning_rate": float(args.learning_rate),
            "l2_regularization": float(args.l2_regularization),
        },
        "source_contract": {
            "old_train": str(
                args.old_data.resolve() / "grids/train.jsonl"
            ),
            "old_val": str(args.old_data.resolve() / "grids/val.jsonl"),
            "new_train": str(
                args.new_data.resolve() / "grids/train.jsonl"
            ),
            "new_val": str(args.new_data.resolve() / "grids/val.jsonl"),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "direction_tree_v4_summary.json"
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
                "difficult_joint_exact": new_metrics["joint_exact"],
                "old_iid_macro_f1": old_metrics["equal_field_macro_f1"],
                "difficult_macro_f1": new_metrics[
                    "equal_field_macro_f1"
                ],
                "all_gates_passed": gates["all_passed"],
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
