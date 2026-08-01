#!/usr/bin/env python3
"""Validation-tune a direct-probability plus forward-residual inverse ensemble."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score

from optics_understanding_sft.core import write_jsonl
from optics_understanding_sft.direction_inverse_v1.build_inverse import state_matches
from optics_understanding_sft.direction_inverse_v1.evaluate_inverse_controller import (
    classify, numeric_records, predicted_grid_states, replay_map, score_grid, tune_calibration,
)
from optics_understanding_sft.direction_inverse_v1.train_inverse_direct import (
    STATUSES, action_labels, deployment_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--direct-bundle", type=Path, required=True)
    parser.add_argument("--general-forward", type=Path, required=True)
    parser.add_argument("--grid-forward", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def direct_costs(records: Sequence[Mapping[str, Any]], direct: Mapping[str, Any]) -> list[np.ndarray]:
    x = deployment_arrays(records)
    probabilities = [model.predict_proba(x) for model in direct["action_models"]]
    output = []
    for row_index, record in enumerate(records):
        costs = []
        for action in record["prompt_inputs"]["action_grid"]:
            labels = action_labels(action)
            costs.append(-sum(math.log(max(float(probabilities[axis][row_index, label]), 1e-12))
                              for axis, label in enumerate(labels)))
        output.append(np.asarray(costs, dtype=np.float64))
    return output


def forward_scores(records: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]) -> list[np.ndarray]:
    return [score_grid(row, predicted_grid_states(row, bundle)) for row in records]


def blended_index(direct: np.ndarray, residual: np.ndarray, residual_weight: float) -> int:
    scaled = (residual - residual.min()) / max(float(residual.std()), 1e-9)
    cost = direct + residual_weight * scaled
    return int(np.argmin(cost))


def tune_action(
    records: Sequence[Mapping[str, Any]], private: Mapping[str, Mapping[str, Any]],
    direct: Sequence[np.ndarray], forward_by_name: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, Any]:
    candidates = []
    for name, scores in forward_by_name.items():
        for weight in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            acceptable, exact = [], []
            for record, direct_score, residual in zip(records, direct, scores, strict=True):
                if record["target"]["status"] == "infeasible_within_limits":
                    continue
                index = blended_index(direct_score, residual, weight)
                replay = private[record["match_group_id"]]
                acceptable.append(index in set(replay["matching_indices"]))
                exact.append(replay["selected_index"] is not None and index == replay["selected_index"])
            candidates.append({"forward_model": name, "residual_weight": weight,
                               "validation_acceptable": float(np.mean(acceptable)),
                               "validation_exact": float(np.mean(exact))})
    return max(candidates, key=lambda row: (row["validation_acceptable"], row["validation_exact"],
                                            -row["residual_weight"], row["forward_model"])), candidates


def choose_status_source(records: Sequence[Mapping[str, Any]], direct: Mapping[str, Any],
                         forward_by_name: Mapping[str, Sequence[np.ndarray]]) -> tuple[dict[str, Any], dict[str, Any]]:
    true = np.asarray([STATUSES.index(row["target"]["status"]) for row in records])
    direct_pred = direct["status_model"].predict(deployment_arrays(records))
    choices: dict[str, Any] = {"direct": {"macro_f1": float(f1_score(
        true, direct_pred, labels=[0, 1, 2], average="macro", zero_division=0))}}
    for name, scores in forward_by_name.items():
        calibration = tune_calibration(records, scores)
        choices[name] = {"macro_f1": calibration["validation_status_macro_f1"],
                         "calibration": calibration}
    selected = max(choices, key=lambda name: (choices[name]["macro_f1"], name))
    return {"source": selected, **choices[selected]}, choices


def evaluate(
    records: Sequence[Mapping[str, Any]], private: Mapping[str, Mapping[str, Any]],
    direct_bundle: Mapping[str, Any], direct_scores: Sequence[np.ndarray],
    forward_by_name: Mapping[str, Sequence[np.ndarray]], action_choice: Mapping[str, Any],
    status_choice: Mapping[str, Any], split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if status_choice["source"] == "direct":
        status_pred = direct_bundle["status_model"].predict(deployment_arrays(records))
    else:
        calibration = status_choice["calibration"]
        status_pred = np.asarray([STATUSES.index(classify(score, calibration["feasibility_cutoff"],
                                                         calibration["ambiguity_margin"]))
                                  for score in forward_by_name[status_choice["source"]]])
    true = np.asarray([STATUSES.index(row["target"]["status"]) for row in records])
    acceptable, exact, achieved, details = [], [], [], []
    selected_forward = forward_by_name[action_choice["forward_model"]]
    for n, (record, direct_score, residual) in enumerate(zip(records, direct_scores, selected_forward, strict=True)):
        index = blended_index(direct_score, residual, float(action_choice["residual_weight"]))
        replay = private[record["match_group_id"]]; matches = set(replay["matching_indices"])
        acceptable.append(index in matches)
        exact.append(replay["selected_index"] is not None and index == replay["selected_index"])
        inputs = record["prompt_inputs"]
        achieved.append(state_matches(replay["candidate_states"][index], inputs["desired_beam_state_B"],
                                      inputs["matching_tolerance"]))
        details.append({"example_id": record["example_id"], "split": split,
                        "target_status": record["target"]["status"],
                        "predicted_status": STATUSES[int(status_pred[n])], "selected_index": index,
                        "selected_action": inputs["action_grid"][index],
                        "selected_is_acceptable": acceptable[-1], "selected_exact": exact[-1],
                        "selected_action_reaches_target": achieved[-1]})
    feasible = true != STATUSES.index("infeasible_within_limits")
    return {"count": len(records), "status_accuracy": float(np.mean(status_pred == true)),
            "status_macro_f1": float(f1_score(true, status_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "selected_exact_feasible": float(np.mean(np.asarray(exact)[feasible])),
            "selected_acceptable_feasible": float(np.mean(np.asarray(acceptable)[feasible])),
            "selected_action_target_success_feasible": float(np.mean(np.asarray(achieved)[feasible]))}, details


def main() -> None:
    args = parse_args(); direct = load_pickle(args.direct_bundle)
    forward = {"general_forward": load_pickle(args.general_forward),
               "grid_forward": load_pickle(args.grid_forward)}
    records = {split: numeric_records(args.data_dir, split) for split in ("val", "eval_iid", "eval_ood")}
    private = {split: replay_map(args.data_dir, split) for split in records}
    direct_by_split = {split: direct_costs(rows, direct) for split, rows in records.items()}
    forward_by_split = {split: {name: forward_scores(rows, bundle) for name, bundle in forward.items()}
                        for split, rows in records.items()}
    action_choice, action_search = tune_action(records["val"], private["val"], direct_by_split["val"],
                                                forward_by_split["val"])
    status_choice, status_search = choose_status_source(records["val"], direct, forward_by_split["val"])
    summary: dict[str, Any] = {"version": "direction_inverse_v1_inverse_ensemble",
                               "simulator_at_inference": False, "selected_action_routing": action_choice,
                               "action_validation_search": action_search,
                               "selected_status_routing": status_choice,
                               "status_validation_search": status_search}
    details = []
    for split in ("val", "eval_iid", "eval_ood"):
        summary[split], rows = evaluate(records[split], private[split], direct, direct_by_split[split],
                                        forward_by_split[split], action_choice, status_choice, split)
        details.extend(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "details.jsonl", details)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
