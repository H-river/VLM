#!/usr/bin/env python3
"""Measure calibrated beam images and evaluate visual A-to-B inverse control."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from optics_understanding_sft.core import read_jsonl, write_jsonl
from optics_understanding_sft.direction_inverse_v1.evaluate_inverse_controller import (
    evaluate, predicted_grid_states, score_grid, tune_calibration,
)
from optics_understanding_sft.direction_inverse_v1.repair_visual_sensor_frame import sensor_state, zero_action

STATE_KEYS = ("centroid_x_px", "centroid_y_px", "sigma_x_px", "sigma_y_px", "peak_intensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--forward-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def measure_image(path: Path, calibration: Mapping[str, Any]) -> dict[str, float]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    gamma = float(calibration.get("gamma", 1.0))
    if gamma != 1.0:
        gray = np.power(gray, 1.0 / gamma)
    low = float(calibration["linear_intensity_low"])
    high = float(calibration["linear_intensity_high"])
    intensity = low + gray * (high - low)
    weights = np.maximum(intensity - low, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"image has no positive intensity: {path}")
    height, width = weights.shape
    source_width, source_height = map(float, calibration["source_sensor_resolution_px"])
    x = (np.arange(width, dtype=np.float64) + 0.5) * source_width / width - 0.5
    y = (np.arange(height, dtype=np.float64) + 0.5) * source_height / height - 0.5
    marginal_x, marginal_y = weights.sum(axis=0), weights.sum(axis=1)
    cx = float(np.dot(marginal_x, x) / total)
    cy = float(np.dot(marginal_y, y) / total)
    sx = float(np.sqrt(np.dot(marginal_x, np.square(x - cx)) / total))
    sy = float(np.sqrt(np.dot(marginal_y, np.square(y - cy)) / total))
    return {"centroid_x_px": cx, "centroid_y_px": cy, "sigma_x_px": sx,
            "sigma_y_px": sy, "peak_intensity": float(intensity.max())}


def paired_records(data_dir: Path, split: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = read_jsonl(data_dir / "inverse/canonical" / f"{split}.jsonl")
    numeric = {row["match_group_id"]: row for row in rows if row["task_type"] == "inverse_action_numeric"}
    visual = {row["match_group_id"]: row for row in rows if row["task_type"] == "inverse_action_visual"}
    if set(numeric) != set(visual):
        raise RuntimeError(f"numeric/visual pair mismatch in {split}")
    return [(numeric[key], visual[key]) for key in sorted(numeric)]


def measured_numeric_record(
    data_dir: Path, numeric: Mapping[str, Any], visual: Mapping[str, Any],
    private: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = visual["prompt_inputs"]["image_calibration"]
    paths = [data_dir / value for value in visual["prompt_inputs"]["images"]]
    measured_a, measured_b = (measure_image(path, calibration) for path in paths)
    shadow = copy.deepcopy(dict(numeric))
    shadow["example_id"] = visual["example_id"]
    shadow["target"] = copy.deepcopy(visual["target"])
    # Forward models use the legacy lab frame for their initial state. Convert
    # only image A back to that frame; image B remains in the sensor frame.
    setup = numeric["prompt_inputs"]["setup"]
    pitch_mm = float(setup["pixel_size_um"]) / 1000.0
    measured_a_legacy = dict(measured_a)
    measured_a_legacy["centroid_x_px"] += float(setup["camera_x_offset_mm"]) / pitch_mm
    measured_a_legacy["centroid_y_px"] += float(setup["camera_y_offset_mm"]) / pitch_mm
    shadow["prompt_inputs"]["current_beam_state_A"] = measured_a_legacy
    shadow["prompt_inputs"]["desired_beam_state_B"] = measured_b
    return shadow, {"measured_A": measured_a, "measured_B": measured_b,
                    "true_A": private["before_state"], "true_B": private["target_state"]}


def visual_grid_scores(record: Mapping[str, Any], bundle: Mapping[str, Any]) -> np.ndarray:
    legacy_states = predicted_grid_states(record, bundle)
    inputs = record["prompt_inputs"]
    sensor_states = [sensor_state(state, inputs["setup"], action)
                     for state, action in zip(legacy_states, inputs["action_grid"], strict=True)]
    return score_grid(record, sensor_states)


def visual_replay_map(data_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    return {row["pair_id"]: row for row in
            read_jsonl(data_dir / "inverse/private" / f"{split}_visual_replay.jsonl")}


def measurement_metrics(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"pair_count": len(items), "state_count": len(items) * 2}
    for key in STATE_KEYS:
        errors = [abs(float(item[f"measured_{side}"][key]) - float(item[f"true_{side}"][key]))
                  for item in items for side in ("A", "B")]
        result[f"{key}_mae"] = float(np.mean(errors))
        result[f"{key}_max_error"] = float(np.max(errors))
    all_pass = []
    for item in items:
        for side in ("A", "B"):
            measured, true = item[f"measured_{side}"], item[f"true_{side}"]
            all_pass.append(
                abs(measured["centroid_x_px"] - true["centroid_x_px"]) <= 1.0
                and abs(measured["centroid_y_px"] - true["centroid_y_px"]) <= 1.0
                and abs(measured["sigma_x_px"] - true["sigma_x_px"]) <= 2.0
                and abs(measured["sigma_y_px"] - true["sigma_y_px"]) <= 2.0
                and abs(measured["peak_intensity"] - true["peak_intensity"])
                <= 0.05 * max(abs(true["peak_intensity"]), 1e-12)
            )
    result["strict_all_five_state_success"] = float(np.mean(all_pass))
    return result


def report_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Visual measurement and inverse-control pipeline", "",
             "A deterministic moment-based image meter extracts calibrated beam states. The learned forward model then searches the 81-action grid; cached simulator states are used only for private scoring.", "",
             "| Split | Image state all-five | Centroid-x MAE | Width-x MAE | Peak MAE | Inverse target reached |",
             "|---|---:|---:|---:|---:|---:|"]
    for split in ("val", "eval_iid", "eval_ood"):
        m, c = summary[split]["measurement"], summary[split]["controller"]
        lines.append(f"| {split} | {m['strict_all_five_state_success']:.3f} | {m['centroid_x_px_mae']:.3f} | {m['sigma_x_px_mae']:.3f} | {m['peak_intensity_mae']:.3f} | {c['selected_action_target_success_feasible']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    with args.forward_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    records: dict[str, list[dict[str, Any]]] = {}; measurements = {}; grids = {}
    for split in ("val", "eval_iid", "eval_ood"):
        private = visual_replay_map(args.data_dir, split)
        built = [measured_numeric_record(args.data_dir, numeric, visual,
                                         private[visual["match_group_id"]])
                 for numeric, visual in paired_records(args.data_dir, split)]
        records[split] = [item[0] for item in built]
        measurements[split] = [item[1] for item in built]
        grids[split] = [visual_grid_scores(row, bundle) for row in records[split]]
    calibration = tune_calibration(records["val"], grids["val"])
    summary: dict[str, Any] = {"version": "direction_inverse_v1_visual_pipeline",
                               "image_model": "deterministic calibrated intensity moments",
                               "simulator_at_inference": False,
                               "calibration_selected_on": "val", "controller_calibration": calibration}
    details = []
    for split in ("val", "eval_iid", "eval_ood"):
        controller, controller_details = evaluate(records[split], grids[split],
                                                  visual_replay_map(args.data_dir, split), calibration, split)
        summary[split] = {"measurement": measurement_metrics(measurements[split]),
                          "controller": controller}
        for measured, controller_detail in zip(measurements[split], controller_details, strict=True):
            details.append({**controller_detail, **measured})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "details.jsonl", details)
    (args.output_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
