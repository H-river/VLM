#!/usr/bin/env python3
"""Train a direct local inverse classifier from exhaustive train-grid replay."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from optics_understanding_sft.core import read_jsonl, write_jsonl
from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    ACTION_FIELDS, matching_indices, select_minimum_motion, state_matches,
)

STATUSES = ("unique", "ambiguous", "infeasible_within_limits")
SETUP_FIELDS = ("wavelength_nm", "beam_waist_mm", "power_w", "lens_focal_length_mm",
                "lens_aperture_mm", "source_to_lens_mm", "lens_to_camera_mm",
                "lens_x_offset_mm", "lens_y_offset_mm", "camera_x_offset_mm",
                "camera_y_offset_mm", "pixel_size_um")
STATE_FIELDS = ("centroid_x_px", "centroid_y_px", "sigma_x_px", "sigma_y_px", "peak_intensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def action_values(field: str) -> tuple[float, float, float]:
    return (-0.05, 0.0, 0.05) if field.startswith("lens") else (-0.02, 0.0, 0.02)


def action_labels(action: Mapping[str, Any]) -> list[int]:
    return [min(range(3), key=lambda i: abs(float(action[field]) - action_values(field)[i]))
            for field in ACTION_FIELDS]


def feature(setup: Mapping[str, Any], current: Mapping[str, Any], desired: Mapping[str, Any]) -> list[float]:
    setup_values = [float(setup[key]) for key in SETUP_FIELDS]
    current_values = [float(current[key]) for key in STATE_FIELDS]
    desired_values = [float(desired[key]) for key in STATE_FIELDS]
    difference = [after - before for after, before in zip(desired_values, current_values, strict=True)]
    peak = max(abs(current_values[-1]), 1e-9)
    derived = [difference[-1] / peak, np.hypot(difference[0], difference[1]),
               np.hypot(float(setup["lens_x_offset_mm"]), float(setup["lens_y_offset_mm"])),
               np.hypot(float(setup["camera_x_offset_mm"]), float(setup["camera_y_offset_mm"]))]
    return setup_values + current_values + desired_values + difference + derived


def public_numeric(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(data_dir / "inverse/canonical" / f"{split}.jsonl")
            if row["task_type"] == "inverse_action_numeric"]


def replay_map(data_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    return {row["pair_id"]: row for row in read_jsonl(data_dir / "inverse/private" / f"{split}_replay.jsonl")}


def augmented_training(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    public = public_numeric(data_dir, "train")
    by_pair = {row["match_group_id"]: row for row in public}
    replay = replay_map(data_dir, "train")
    representative: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for pair_id, private in replay.items():
        representative.setdefault(private["group_id"], (private, by_pair[pair_id]))
    features, statuses, actions_out = [], [], []
    for private, record in representative.values():
        inputs = record["prompt_inputs"]; actions = inputs["action_grid"]
        states, tolerance = private["candidate_states"], inputs["matching_tolerance"]
        for target in states:
            matches = matching_indices(states, target, tolerance)
            selected = select_minimum_motion(actions, states, target, matches, tolerance)
            if selected is None:
                continue
            status = "unique" if len(matches) == 1 else "ambiguous"
            features.append(feature(inputs["setup"], inputs["current_beam_state_A"], target))
            statuses.append(STATUSES.index(status)); actions_out.append(action_labels(actions[selected]))
    # Add the certified off-grid infeasible targets. Balanced class weights stop
    # their smaller count from being ignored by the status head.
    for record in public:
        if record["target"]["status"] != "infeasible_within_limits":
            continue
        inputs = record["prompt_inputs"]
        features.append(feature(inputs["setup"], inputs["current_beam_state_A"],
                                inputs["desired_beam_state_B"]))
        statuses.append(STATUSES.index("infeasible_within_limits"))
        actions_out.append(action_labels(record["target"]["answer"]["best_grid_action"]))
    return np.asarray(features, dtype=np.float32), np.asarray(statuses), np.asarray(actions_out)


def deployment_arrays(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([feature(row["prompt_inputs"]["setup"],
                               row["prompt_inputs"]["current_beam_state_A"],
                               row["prompt_inputs"]["desired_beam_state_B"]) for row in records],
                      dtype=np.float32)


def make_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=400, max_leaf_nodes=31, learning_rate=0.08,
                                          l2_regularization=1.0, class_weight="balanced",
                                          early_stopping=True, random_state=seed)


def action_from_prediction(labels: Sequence[int]) -> dict[str, float]:
    return {field: action_values(field)[int(label)] for field, label in zip(ACTION_FIELDS, labels, strict=True)}


def action_index(actions: Sequence[Mapping[str, Any]], predicted: Mapping[str, Any]) -> int:
    return min(range(len(actions)), key=lambda i: (sum(abs(float(actions[i][field]) - float(predicted[field]))
                                                       for field in ACTION_FIELDS), i))


def evaluate(records: list[dict[str, Any]], private: Mapping[str, Mapping[str, Any]],
             status_model: Any, action_models: Sequence[Any], split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = deployment_arrays(records)
    status_pred = status_model.predict(x)
    axis_pred = np.column_stack([model.predict(x) for model in action_models])
    status_true = np.asarray([STATUSES.index(row["target"]["status"]) for row in records])
    acceptable, exact, achieved, details = [], [], [], []
    for n, row in enumerate(records):
        action = action_from_prediction(axis_pred[n]); inputs = row["prompt_inputs"]
        index = action_index(inputs["action_grid"], action); replay = private[row["match_group_id"]]
        matches = set(replay["matching_indices"]); selected = replay["selected_index"]
        acceptable.append(index in matches); exact.append(selected is not None and index == selected)
        achieved.append(state_matches(replay["candidate_states"][index], inputs["desired_beam_state_B"],
                                      inputs["matching_tolerance"]))
        details.append({"example_id": row["example_id"], "split": split,
                        "target_status": row["target"]["status"],
                        "predicted_status": STATUSES[int(status_pred[n])], "selected_index": index,
                        "selected_action": action, "selected_is_acceptable": acceptable[-1],
                        "selected_exact": exact[-1], "selected_action_reaches_target": achieved[-1]})
    feasible = status_true != STATUSES.index("infeasible_within_limits")
    return {"count": len(records), "status_accuracy": float(np.mean(status_pred == status_true)),
            "status_macro_f1": float(f1_score(status_true, status_pred, labels=[0, 1, 2],
                                               average="macro", zero_division=0)),
            "selected_exact_feasible": float(np.mean(np.asarray(exact)[feasible])),
            "selected_acceptable_feasible": float(np.mean(np.asarray(acceptable)[feasible])),
            "selected_action_target_success_feasible": float(np.mean(np.asarray(achieved)[feasible]))}, details


def main() -> None:
    args = parse_args(); x, y_status, y_action = augmented_training(args.data_dir)
    status_model = make_classifier(42).fit(x, y_status)
    action_models = [make_classifier(100 + i).fit(x, y_action[:, i]) for i in range(len(ACTION_FIELDS))]
    bundle = {"version": "direction_inverse_v1_direct_inverse", "feature_fields": list(SETUP_FIELDS),
              "status_model": status_model, "action_models": action_models,
              "simulator_at_inference": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "inverse_direct.pkl").open("wb") as stream: pickle.dump(bundle, stream)
    summary: dict[str, Any] = {"version": bundle["version"], "simulator_at_inference": False,
                               "augmented_training_records": len(x),
                               "training_status_counts": {status: int(np.sum(y_status == i))
                                                          for i, status in enumerate(STATUSES)}}
    details = []
    for split in ("val", "eval_iid", "eval_ood"):
        records = public_numeric(args.data_dir, split)
        summary[split], rows = evaluate(records, replay_map(args.data_dir, split), status_model,
                                        action_models, split); details.extend(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "details.jsonl", details)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
