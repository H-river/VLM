#!/usr/bin/env python3
"""Train a coupled 81-action inverse classifier and select it on validation."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

from optics_understanding_sft.core import write_jsonl
from optics_understanding_sft.direction_inverse_v1.build_inverse import state_matches
from optics_understanding_sft.direction_inverse_v1.train_inverse_direct import (
    STATUSES,
    action_from_prediction,
    action_index,
    augmented_training,
    deployment_arrays,
    make_classifier,
    public_numeric,
    replay_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def joint_labels(data_dir: Path, axis_labels: np.ndarray) -> np.ndarray:
    actions = public_numeric(data_dir, "train")[0]["prompt_inputs"]["action_grid"]
    return np.asarray(
        [action_index(actions, action_from_prediction(labels)) for labels in axis_labels],
        dtype=np.int64,
    )


def make_joint(seed: int, min_samples_leaf: int, max_features: float) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def evaluate(
    records: list[dict[str, Any]],
    private: Mapping[str, Mapping[str, Any]],
    status_model: Any,
    action_model: Any,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = deployment_arrays(records)
    status_pred = status_model.predict(x)
    index_pred = action_model.predict(x).astype(int)
    status_true = np.asarray([STATUSES.index(row["target"]["status"]) for row in records])
    acceptable: list[bool] = []
    exact: list[bool] = []
    achieved: list[bool] = []
    details: list[dict[str, Any]] = []
    for n, row in enumerate(records):
        inputs = row["prompt_inputs"]
        index = int(index_pred[n])
        replay = private[row["match_group_id"]]
        matches = set(replay["matching_indices"])
        selected = replay["selected_index"]
        acceptable.append(index in matches)
        exact.append(selected is not None and index == selected)
        achieved.append(
            state_matches(
                replay["candidate_states"][index],
                inputs["desired_beam_state_B"],
                inputs["matching_tolerance"],
            )
        )
        details.append(
            {
                "example_id": row["example_id"],
                "split": split,
                "target_status": row["target"]["status"],
                "predicted_status": STATUSES[int(status_pred[n])],
                "selected_index": index,
                "selected_action": inputs["action_grid"][index],
                "selected_is_acceptable": acceptable[-1],
                "selected_exact": exact[-1],
                "selected_action_reaches_target": achieved[-1],
            }
        )
    feasible = status_true != STATUSES.index("infeasible_within_limits")
    return {
        "count": len(records),
        "status_accuracy": float(np.mean(status_pred == status_true)),
        "status_macro_f1": float(
            f1_score(status_true, status_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "selected_exact_feasible": float(np.mean(np.asarray(exact)[feasible])),
        "selected_acceptable_feasible": float(np.mean(np.asarray(acceptable)[feasible])),
        "selected_action_target_success_feasible": float(np.mean(np.asarray(achieved)[feasible])),
    }, details


def main() -> None:
    args = parse_args()
    x, y_status, y_axis = augmented_training(args.data_dir)
    y_joint = joint_labels(args.data_dir, y_axis)
    status_model = make_classifier(42).fit(x, y_status)
    val_records = public_numeric(args.data_dir, "val")
    val_private = replay_map(args.data_dir, "val")
    search: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for leaf in (1, 2, 4, 8):
        for features in (0.6, 1.0):
            name = f"extra_trees_leaf{leaf}_features{features:.1f}"
            model = make_joint(173 + leaf, leaf, features).fit(x, y_joint)
            metrics, _ = evaluate(val_records, val_private, status_model, model, "val")
            search.append({"name": name, "min_samples_leaf": leaf, "max_features": features,
                           "validation_metrics": metrics})
            models[name] = model
    selected = max(
        search,
        key=lambda row: (
            row["validation_metrics"]["selected_action_target_success_feasible"],
            row["validation_metrics"]["selected_exact_feasible"],
            -row["min_samples_leaf"],
            -row["max_features"],
        ),
    )
    action_model = models[selected["name"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "inverse_joint.pkl").open("wb") as stream:
        pickle.dump(
            {
                "version": "direction_inverse_v1_joint_inverse",
                "status_model": status_model,
                "action_model": action_model,
                "selected_hyperparameters": selected,
                "simulator_at_inference": False,
            },
            stream,
        )
    summary: dict[str, Any] = {
        "version": "direction_inverse_v1_joint_inverse",
        "simulator_at_inference": False,
        "training_records": len(x),
        "joint_action_classes": int(len(np.unique(y_joint))),
        "selection_policy": "highest feasible target success on validation, then exact success",
        "validation_search": search,
        "selected_model": selected,
    }
    details: list[dict[str, Any]] = []
    for split in ("val", "eval_iid", "eval_ood"):
        metrics, rows = evaluate(
            public_numeric(args.data_dir, split), replay_map(args.data_dir, split),
            status_model, action_model, split,
        )
        summary[split] = metrics
        details.extend(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(args.output_dir / "details.jsonl", details)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
