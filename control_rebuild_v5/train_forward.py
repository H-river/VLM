#!/usr/bin/env python3
"""Train and validate the five-head continuous forward tree v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID, group_arrays, read_jsonl
from control_rebuild_v3.train_forward import forward_metrics
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from control_rebuild_v5.forward_runtime import (
    ForwardTreeRuntimeV5,
    ZERO_ACTION_INDEX,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    STATE_FIELDS,
    forward_feature,
)

DEFAULT_OLD_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_DIFFICULT_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
)
DEFAULT_ADDITIONAL_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
)
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/control_rebuild_v5_one_seed"
DEFAULT_V4_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v4_quickcheck_12h"
    / "forward_physics_residual_v4.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", type=Path, default=DEFAULT_OLD_DATA)
    parser.add_argument(
        "--difficult-data",
        type=Path,
        default=DEFAULT_DIFFICULT_DATA,
    )
    parser.add_argument(
        "--additional-data",
        type=Path,
        default=DEFAULT_ADDITIONAL_DATA,
    )
    parser.add_argument("--include-additional-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--max-leaf-nodes", type=int, default=127)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--l2-regularization", type=float, default=0.10)
    parser.add_argument("--v4-forward", type=Path, default=DEFAULT_V4_FORWARD)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_array(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                forward_feature(
                    row["setup"],
                    row["current_beam_state"],
                    action,
                )
                for action in ACTION_GRID
            ]
            for row in rows
        ],
        dtype=np.float32,
    )


def action_complexity() -> np.ndarray:
    actions = np.asarray(
        [
            [float(action[field]) for field in ACTION_FIELDS]
            for action in ACTION_GRID
        ],
        dtype=np.float32,
    )
    return np.count_nonzero(np.abs(actions) > 0.0, axis=1)


def training_weights(
    old_group_count: int,
    difficult_group_count: int,
    additional_group_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Balance action complexity while preserving all three data sources."""

    complexity = action_complexity()
    counts = np.bincount(complexity, minlength=5)
    per_action = np.asarray(
        [
            0.02 if value == 0 else 0.245 / counts[value]
            for value in complexity
        ],
        dtype=np.float32,
    )
    if additional_group_count:
        source_mass = {
            "old_iid": 0.30,
            "difficult_v4": 0.30,
            "additional_v5": 0.40,
        }
    else:
        total = old_group_count + difficult_group_count
        source_mass = {
            "old_iid": old_group_count / total,
            "difficult_v4": difficult_group_count / total,
            "additional_v5": 0.0,
        }
    per_group = np.concatenate(
        [
            np.full(
                old_group_count,
                source_mass["old_iid"] / old_group_count,
                dtype=np.float32,
            ),
            np.full(
                difficult_group_count,
                source_mass["difficult_v4"] / difficult_group_count,
                dtype=np.float32,
            ),
            (
                np.full(
                    additional_group_count,
                    source_mass["additional_v5"] / additional_group_count,
                    dtype=np.float32,
                )
                if additional_group_count
                else np.empty(0, dtype=np.float32)
            ),
        ]
    )
    weights = (per_group[:, None] * per_action[None, :]).reshape(-1)
    weights *= len(weights) / weights.sum()
    return weights, source_mass


def metrics_by_complexity(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    complexity = action_complexity()
    output = {}
    for value in range(5):
        selected = complexity == value
        error = np.abs(predicted[:, selected] - truth[:, selected])
        output[str(value)] = {
            "transitions": int(error.shape[0] * error.shape[1]),
            "strict_all_five_success": float(
                np.all(error <= 1.0, axis=-1).mean()
            ),
            "mae_in_tolerance_units": float(error.mean()),
            "per_field_mae": {
                field: float(error[..., index].mean())
                for index, field in enumerate(STATE_FIELDS)
            },
        }
    return output


def category_metrics(
    rows: Sequence[Mapping[str, Any]],
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    if not rows or "source_category" not in rows[0]:
        return {}
    categories = sorted({str(row["source_category"]) for row in rows})
    return {
        category: forward_metrics(
            truth[
                np.asarray(
                    [
                        str(row["source_category"]) == category
                        for row in rows
                    ],
                    dtype=np.bool_,
                )
            ],
            predicted[
                np.asarray(
                    [
                        str(row["source_category"]) == category
                        for row in rows
                    ],
                    dtype=np.bool_,
                )
            ],
        )
        for category in categories
    }


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    return {
        "overall": forward_metrics(truth, predicted),
        "by_action_complexity": metrics_by_complexity(truth, predicted),
        "by_source_category": category_metrics(rows, truth, predicted),
    }


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    old_train = read_jsonl(args.old_data.resolve() / "grids/train.jsonl")
    old_val = read_jsonl(args.old_data.resolve() / "grids/val.jsonl")
    difficult_train = read_jsonl(
        args.difficult_data.resolve() / "grids/train.jsonl"
    )
    difficult_val = read_jsonl(
        args.difficult_data.resolve() / "grids/val.jsonl"
    )
    additional_train: list[dict[str, Any]] = []
    if args.include_additional_data:
        additional_train = read_jsonl(
            args.additional_data.resolve() / "grids/train.jsonl"
        )
    train_rows = [*old_train, *difficult_train, *additional_train]

    features_train = feature_array(train_rows)
    features_old = feature_array(old_val)
    features_difficult = feature_array(difficult_val)
    _, _, _, target_train, _ = group_arrays(train_rows)
    _, _, _, target_old, _ = group_arrays(old_val)
    _, _, _, target_difficult, _ = group_arrays(difficult_val)
    flat_train = features_train.reshape(-1, features_train.shape[-1])
    flat_target = target_train.reshape(-1, len(STATE_FIELDS))
    weights, source_mass = training_weights(
        len(old_train),
        len(difficult_train),
        len(additional_train),
    )

    models = []
    trace = []
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
            random_state=int(args.seed) + field_index,
        )
        model.fit(
            flat_train,
            flat_target[:, field_index],
            sample_weight=weights,
        )
        record = {
            "field": field,
            "iterations": int(model.n_iter_),
            "seconds": time.perf_counter() - field_started,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        trace.append(record)
        models.append(model)

    artifact = {
        "version": "control_rebuild_v5_one_seed",
        "model": "five_head_hist_gradient_boosting_forward_v5",
        "seed": int(args.seed),
        "input_dim": int(features_train.shape[-1]),
        "state_fields": list(STATE_FIELDS),
        "models": models,
        "action_grid": ACTION_GRID,
        "zero_action_index": ZERO_ACTION_INDEX,
        "zero_action_exact": True,
        "feature_mode": "engineered_46",
        "action_complexity_weighting": {
            "zero": 0.02,
            "one": 0.245,
            "two": 0.245,
            "three": 0.245,
            "four": 0.245,
        },
        "training_source_weighting": source_mass,
        "held_out_test_used": False,
    }
    runtime = ForwardTreeRuntimeV5(artifact)
    predicted_train = runtime.predict_changes(train_rows)
    predicted_old = runtime.predict_changes(old_val)
    predicted_difficult = runtime.predict_changes(difficult_val)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v4_runtime, _ = load_forward_runtime_v4(
        args.v4_forward.resolve(),
        torch,
        device,
    )
    v4_old = v4_runtime.predict_changes(old_val)
    v4_difficult = v4_runtime.predict_changes(difficult_val)

    artifact_path = output_dir / "forward_tree_v5.pkl"
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
            "old_groups": len(old_train),
            "difficult_groups": len(difficult_train),
            "additional_groups": len(additional_train),
            "unique_groups": len(train_rows),
            "transitions": len(train_rows) * len(ACTION_GRID),
            "source_weighting": source_mass,
            "trace": trace,
            "diagnostic": evaluate(
                train_rows,
                target_train,
                predicted_train,
            ),
        },
        "validation": {
            "forward_v4_reference": {
                "old_iid": evaluate(old_val, target_old, v4_old),
                "difficult": evaluate(
                    difficult_val,
                    target_difficult,
                    v4_difficult,
                ),
            },
            "forward_tree_v5": {
                "old_iid": evaluate(
                    old_val,
                    target_old,
                    predicted_old,
                ),
                "difficult": evaluate(
                    difficult_val,
                    target_difficult,
                    predicted_difficult,
                ),
            },
        },
        "hyperparameters": {
            "max_iter": int(args.max_iter),
            "max_leaf_nodes": int(args.max_leaf_nodes),
            "min_samples_leaf": int(args.min_samples_leaf),
            "learning_rate": float(args.learning_rate),
            "l2_regularization": float(args.l2_regularization),
        },
        "source_contract": {
            "old_train": str(args.old_data.resolve() / "grids/train.jsonl"),
            "old_val": str(args.old_data.resolve() / "grids/val.jsonl"),
            "difficult_train": str(
                args.difficult_data.resolve() / "grids/train.jsonl"
            ),
            "difficult_val": str(
                args.difficult_data.resolve() / "grids/val.jsonl"
            ),
            "additional_train": (
                str(args.additional_data.resolve() / "grids/train.jsonl")
                if args.include_additional_data
                else None
            ),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "forward_tree_v5_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected = summary["validation"]["forward_tree_v5"]
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "summary": str(summary_path),
                "old_iid_strict_all_five": selected["old_iid"]["overall"][
                    "strict_all_five_success"
                ],
                "difficult_strict_all_five": selected["difficult"]["overall"][
                    "strict_all_five_success"
                ],
                "difficult_four_component": selected["difficult"][
                    "by_action_complexity"
                ]["4"]["strict_all_five_success"],
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
