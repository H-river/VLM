#!/usr/bin/env python3
"""Evaluate forward-model grid search on numeric inverse A-to-B records."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score

from optics_understanding_sft.core import read_jsonl, write_jsonl
from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    ACTION_FIELDS, movement_mm, normalized_residual, state_matches,
)
from optics_understanding_sft.direction_inverse_v1.train_forward_small import (
    BASE_TOLERANCES, input_arrays, peak_tolerance, predict_bundle,
)

STATUSES = ("unique", "ambiguous", "infeasible_within_limits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--forward-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def numeric_records(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(data_dir / "inverse/canonical" / f"{split}.jsonl")
            if row["task_type"] == "inverse_action_numeric"]


def predicted_grid_states(record: Mapping[str, Any], bundle: Mapping[str, Any]) -> list[dict[str, float]]:
    inputs = record["prompt_inputs"]
    current = inputs["current_beam_state_A"]
    synthetic = [{"inputs": {"setup": inputs["setup"], "current_beam_state": current,
                              "action": action}} for action in inputs["action_grid"]]
    predicted_scaled, _ = predict_bundle(bundle, input_arrays(synthetic))
    tolerance = BASE_TOLERANCES.copy()
    tolerance[4] = peak_tolerance(synthetic[0])
    change = predicted_scaled * tolerance
    keys = ("centroid_x_px", "centroid_y_px", "sigma_x_px", "sigma_y_px", "peak_intensity")
    return [
        {key: float(current[key]) + float(change[index, field_index])
         for field_index, key in enumerate(keys)}
        for index in range(len(change))
    ]


def score_grid(record: Mapping[str, Any], states: Sequence[Mapping[str, Any]]) -> np.ndarray:
    inputs = record["prompt_inputs"]
    return np.asarray([
        normalized_residual(state, inputs["desired_beam_state_B"], inputs["matching_tolerance"])
        for state in states
    ], dtype=np.float64)


def classify(scores: np.ndarray, feasibility_cutoff: float, ambiguity_margin: float) -> str:
    ordered = np.sort(scores)
    if ordered[0] > feasibility_cutoff:
        return "infeasible_within_limits"
    if len(ordered) > 1 and ordered[1] <= ordered[0] + ambiguity_margin:
        return "ambiguous"
    return "unique"


def selected_index(record: Mapping[str, Any], scores: np.ndarray) -> int:
    actions = record["prompt_inputs"]["action_grid"]
    return min(range(len(scores)), key=lambda i: (float(scores[i]), movement_mm(actions[i]), i))


def tune_calibration(records: Sequence[Mapping[str, Any]], grids: Sequence[np.ndarray]) -> dict[str, float]:
    target = [STATUSES.index(row["target"]["status"]) for row in records]
    best: tuple[float, float, float] | None = None
    for cutoff in np.linspace(0.25, 6.0, 24):
        for margin in np.linspace(0.0, 1.5, 16):
            predicted = [STATUSES.index(classify(scores, float(cutoff), float(margin))) for scores in grids]
            score = float(f1_score(target, predicted, labels=[0, 1, 2], average="macro", zero_division=0))
            candidate = (score, -float(cutoff), -float(margin))
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    return {"validation_status_macro_f1": best[0], "feasibility_cutoff": -best[1],
            "ambiguity_margin": -best[2]}


def replay_map(data_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    return {row["pair_id"]: row for row in read_jsonl(data_dir / "inverse/private" / f"{split}_replay.jsonl")}


def evaluate(
    records: Sequence[Mapping[str, Any]], grids: Sequence[np.ndarray], replay: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, float], split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    targets, predictions, details = [], [], []
    selected_exact, acceptable, achieved, zero_achieved = [], [], [], []
    for record, scores in zip(records, grids, strict=True):
        true_status = record["target"]["status"]
        predicted_status = classify(scores, calibration["feasibility_cutoff"], calibration["ambiguity_margin"])
        index = selected_index(record, scores)
        private = replay[record["match_group_id"]]
        true_selected = private["selected_index"]
        true_matches = set(private["matching_indices"])
        target_state = record["prompt_inputs"]["desired_beam_state_B"]
        tolerance = record["prompt_inputs"]["matching_tolerance"]
        actual_state = private["candidate_states"][index]
        zero_index = next(i for i, action in enumerate(record["prompt_inputs"]["action_grid"])
                          if all(float(action[field]) == 0.0 for field in ACTION_FIELDS))
        is_achieved = state_matches(actual_state, target_state, tolerance)
        zero_is_achieved = state_matches(private["candidate_states"][zero_index], target_state, tolerance)
        targets.append(STATUSES.index(true_status)); predictions.append(STATUSES.index(predicted_status))
        selected_exact.append(true_selected is not None and index == int(true_selected))
        acceptable.append(index in true_matches)
        achieved.append(is_achieved); zero_achieved.append(zero_is_achieved)
        details.append({
            "example_id": record["example_id"], "split": split, "target_status": true_status,
            "predicted_status": predicted_status, "selected_index": index,
            "selected_action": record["prompt_inputs"]["action_grid"][index],
            "true_selected_index": true_selected, "true_matching_indices": sorted(true_matches),
            "selected_exact": selected_exact[-1], "selected_is_acceptable": acceptable[-1],
            "selected_action_reaches_target": is_achieved, "zero_action_reaches_target": zero_is_achieved,
            "best_predicted_normalized_residual": float(scores[index]),
        })
    status_f1 = float(f1_score(targets, predictions, labels=[0, 1, 2], average="macro", zero_division=0))
    feasible_mask = np.asarray([value != STATUSES.index("infeasible_within_limits") for value in targets])
    achieved_array, exact_array, acceptable_array = map(np.asarray, (achieved, selected_exact, acceptable))
    return {
        "count": len(records), "status_accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))),
        "status_macro_f1": status_f1, "selected_exact_all": float(np.mean(exact_array)),
        "selected_exact_feasible": float(np.mean(exact_array[feasible_mask])) if feasible_mask.any() else 0.0,
        "selected_acceptable_feasible": float(np.mean(acceptable_array[feasible_mask])) if feasible_mask.any() else 0.0,
        "selected_action_target_success_feasible": float(np.mean(achieved_array[feasible_mask])) if feasible_mask.any() else 0.0,
        "zero_action_target_success_feasible": float(np.mean(np.asarray(zero_achieved)[feasible_mask])) if feasible_mask.any() else 0.0,
    }, details


def report_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Numeric inverse controller", "",
             "The controller evaluates all 81 declared actions with the learned forward model. Simulator-cached states are used only after prediction for private scoring.", "",
             "| Split | Status macro-F1 | Exact selected action | Any acceptable action | Target reached | Zero-action baseline |",
             "|---|---:|---:|---:|---:|---:|"]
    for split in ("val", "eval_iid", "eval_ood"):
        m = summary[split]
        lines.append(f"| {split} | {m['status_macro_f1']:.3f} | {m['selected_exact_feasible']:.3f} | {m['selected_acceptable_feasible']:.3f} | {m['selected_action_target_success_feasible']:.3f} | {m['zero_action_target_success_feasible']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    with args.forward_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    records = {split: numeric_records(args.data_dir, split) for split in ("val", "eval_iid", "eval_ood")}
    grids = {split: [score_grid(row, predicted_grid_states(row, bundle)) for row in rows]
             for split, rows in records.items()}
    calibration = tune_calibration(records["val"], grids["val"])
    summary: dict[str, Any] = {"version": "direction_inverse_v1_inverse_controller",
                               "simulator_at_inference": False,
                               "private_cached_simulator_states_used_for_scoring_only": True,
                               "calibration_selected_on": "val", "calibration": calibration}
    details = []
    for split in ("val", "eval_iid", "eval_ood"):
        summary[split], rows = evaluate(records[split], grids[split], replay_map(args.data_dir, split), calibration, split)
        details.extend(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "details.jsonl", details)
    (args.output_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
