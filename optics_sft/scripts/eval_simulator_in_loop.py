#!/usr/bin/env python3
"""Simulator-in-the-loop evaluation for inverse_control physics predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optical_sim.src.optical_elements import OpticalSetup, setup_from_dict
from optics_sft.physics.sim_adapter import (
    apply_action_to_setup,
    residual_error_px,
    simulate_and_measure,
)


CONTROL_KEYS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
LENS_KEYS = ("lens_x_delta_mm", "lens_y_delta_mm")
SIGN_EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inverse_control predictions in the simulator loop.")
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def first_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(prediction_row: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(prediction_row, Mapping):
        return None, "missing_prediction"
    parsed_json = prediction_row.get("parsed_json")
    if isinstance(parsed_json, dict):
        return parsed_json, None
    if isinstance(prediction_row.get("prediction"), dict):
        return prediction_row["prediction"], None

    raw_text = prediction_row.get("raw_prediction_text")
    if not isinstance(raw_text, str):
        raw_text = prediction_row.get("raw_text")
    if not isinstance(raw_text, str):
        return None, "prediction_has_no_json_or_raw_text"

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        direct_error = str(exc)
    else:
        if isinstance(parsed, dict):
            return parsed, None
        return None, "raw prediction is valid JSON but not an object"

    block = first_json_object_text(raw_text)
    if block is None:
        return None, direct_error
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "extracted JSON is not an object"
    return parsed, None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def extract_control_plan(parsed_prediction: Mapping[str, Any] | None) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(parsed_prediction, Mapping):
        return None, "prediction_json_missing"
    plan = parsed_prediction.get("control_plan")
    if not isinstance(plan, Mapping):
        return None, "control_plan_missing"
    missing = [key for key in CONTROL_KEYS if key not in plan]
    if missing:
        return None, f"control_plan_missing_keys:{','.join(missing)}"
    non_numeric = [key for key in CONTROL_KEYS if not is_number(plan.get(key))]
    if non_numeric:
        return None, f"control_plan_non_numeric_keys:{','.join(non_numeric)}"
    return {key: float(plan[key]) for key in CONTROL_KEYS}, None


def setup_snapshot_to_config(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    cfg = {
        "source": dict(snapshot.get("source", {})),
        "lens": dict(snapshot.get("lens", {})),
        "sensor": dict(snapshot.get("sensor", {})),
        "geometry": dict(snapshot.get("geometry", {})),
        "camera": dict(snapshot.get("camera", {})),
        "alignment": dict(snapshot.get("alignment", {})),
        "simulation": dict(snapshot.get("simulation", {})),
    }
    cfg["alignment"].setdefault("x_offset", 0.0)
    cfg["alignment"].setdefault("y_offset", 0.0)
    cfg["alignment"].setdefault("tilt_x", 0.0)
    cfg["alignment"].setdefault("tilt_y", 0.0)
    cfg["alignment"].setdefault("defocus", 0.0)
    return cfg


def reconstruct_setup(row: Mapping[str, Any]) -> tuple[OpticalSetup | None, str | None]:
    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        return None, "private_eval_missing"

    simulator_config = private_eval.get("simulator_config")
    if isinstance(simulator_config, Mapping):
        for key in ("current_setup", "initial_setup", "setup"):
            setup_snapshot = simulator_config.get(key)
            if isinstance(setup_snapshot, Mapping):
                return setup_from_dict(setup_snapshot_to_config(setup_snapshot)), None
        sampled = simulator_config.get("sampled_setup_parameters")
        if isinstance(sampled, Mapping):
            return None, "simulator_config_has_sampled_parameters_but_no_reconstructable_setup"

    for key in ("current_setup", "initial_setup", "setup"):
        setup_snapshot = private_eval.get(key)
        if isinstance(setup_snapshot, Mapping):
            return setup_from_dict(setup_snapshot_to_config(setup_snapshot)), None

    return None, "reconstructable_setup_missing"


def vector_norm_px(value: Any) -> float | None:
    if is_number(value):
        return float(value)
    if isinstance(value, Mapping):
        if is_number(value.get("norm")):
            return float(value["norm"])
        if is_number(value.get("x")) and is_number(value.get("y")):
            return math.hypot(float(value["x"]), float(value["y"]))
    return None


def initial_error(row: Mapping[str, Any], target_state: Mapping[str, Any] | None) -> float | None:
    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        return None
    for key in ("initial_error_norm_px", "initial_error_px"):
        value = vector_norm_px(private_eval.get(key))
        if value is not None:
            return value
    current_state = private_eval.get("current_state")
    if isinstance(current_state, Mapping) and isinstance(target_state, Mapping):
        return residual_error_px(current_state, target_state)
    return None


def sign(value: float, epsilon: float = SIGN_EPSILON) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def prediction_index(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(predictions):
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            indexed[sample_id] = row
        elif "sample_index" in row:
            indexed[str(row["sample_index"])] = row
        else:
            indexed[str(index)] = row
    return indexed


def evaluate_row(row: dict[str, Any], prediction_row: dict[str, Any] | None) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    detail: dict[str, Any] = {
        "sample_id": sample_id,
        "sample_type": row.get("sample_type"),
        "json_valid": False,
        "control_plan_valid": False,
        "valid_for_simulation": False,
        "error": None,
    }

    parsed_prediction, parse_error = extract_json_object(prediction_row)
    if parsed_prediction is None:
        detail["error"] = parse_error
        return detail
    detail["json_valid"] = True

    predicted_plan, plan_error = extract_control_plan(parsed_prediction)
    if predicted_plan is None:
        detail["error"] = plan_error
        return detail
    detail["control_plan_valid"] = True
    detail["predicted_control_plan"] = predicted_plan

    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        detail["error"] = "private_eval_missing"
        return detail
    target_state = private_eval.get("target_state")
    if not isinstance(target_state, Mapping):
        detail["error"] = "target_state_missing"
        return detail

    setup, setup_error = reconstruct_setup(row)
    if setup is None:
        detail["error"] = setup_error
        return detail

    initial = initial_error(row, target_state)
    if initial is None:
        detail["error"] = "initial_error_missing"
        return detail

    after_setup = apply_action_to_setup(setup, predicted_plan)
    after = simulate_and_measure(after_setup)
    post = residual_error_px(after["state"], target_state)
    reduction_ratio = (initial - post) / initial if initial > 0.0 else None

    detail.update(
        {
            "valid_for_simulation": True,
            "initial_error_px": initial,
            "post_action_error_px": post,
            "error_reduction_ratio": reduction_ratio,
            "success_under_1px": post <= 1.0,
            "success_under_2px": post <= 2.0,
            "success_under_5px": post <= 5.0,
            "diverged": post > initial,
        }
    )

    true_plan = private_eval.get("true_control_plan")
    if isinstance(true_plan, Mapping):
        detail["true_control_plan"] = {key: float(true_plan[key]) for key in CONTROL_KEYS if is_number(true_plan.get(key))}
        for key in CONTROL_KEYS:
            if is_number(true_plan.get(key)):
                detail[f"{key}_abs_error"] = abs(predicted_plan[key] - float(true_plan[key]))
        for key in LENS_KEYS:
            if is_number(true_plan.get(key)):
                detail[f"{key}_sign_correct"] = sign(predicted_plan[key]) == sign(float(true_plan[key]))

    return detail


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(details)
    json_valid = sum(1 for row in details if row.get("json_valid"))
    control_plan_valid = sum(1 for row in details if row.get("control_plan_valid"))
    simulated = [row for row in details if row.get("valid_for_simulation")]
    initial_values = [float(row["initial_error_px"]) for row in simulated]
    post_values = [float(row["post_action_error_px"]) for row in simulated]
    ratios = [float(row["error_reduction_ratio"]) for row in simulated if row.get("error_reduction_ratio") is not None]

    action_mae: dict[str, float | None] = {}
    for key in CONTROL_KEYS:
        values = [float(row[f"{key}_abs_error"]) for row in simulated if f"{key}_abs_error" in row]
        action_mae[key] = safe_mean(values)

    sign_accuracy: dict[str, float | None] = {}
    for key in LENS_KEYS:
        values = [bool(row[f"{key}_sign_correct"]) for row in simulated if f"{key}_sign_correct" in row]
        sign_accuracy[key] = safe_mean([1.0 if value else 0.0 for value in values])

    error_counts: dict[str, int] = {}
    for row in details:
        error = row.get("error")
        if error:
            error_counts[str(error)] = error_counts.get(str(error), 0) + 1

    return {
        "count": count,
        "simulated_count": len(simulated),
        "json_valid_rate": rate(json_valid, count),
        "control_plan_valid_rate": rate(control_plan_valid, count),
        "mean_initial_error_px": safe_mean(initial_values),
        "mean_post_action_error_px": safe_mean(post_values),
        "mean_error_reduction_ratio": safe_mean(ratios),
        "median_error_reduction_ratio": safe_median(ratios),
        "success_rate_under_1px": rate(sum(1 for row in simulated if row["success_under_1px"]), len(simulated)),
        "success_rate_under_2px": rate(sum(1 for row in simulated if row["success_under_2px"]), len(simulated)),
        "success_rate_under_5px": rate(sum(1 for row in simulated if row["success_under_5px"]), len(simulated)),
        "divergence_rate": rate(sum(1 for row in simulated if row["diverged"]), len(simulated)),
        "missing_simulator_config_count": sum(
            1
            for row in details
            if row.get("error") in {"reconstructable_setup_missing", "private_eval_missing"}
        ),
        "target_state_missing_count": sum(1 for row in details if row.get("error") == "target_state_missing"),
        "action_mae": action_mae,
        "lens_sign_accuracy": sign_accuracy,
        "error_counts": error_counts,
    }


def write_csv(path: Path, details: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "sample_type",
        "json_valid",
        "control_plan_valid",
        "valid_for_simulation",
        "initial_error_px",
        "post_action_error_px",
        "error_reduction_ratio",
        "success_under_1px",
        "success_under_2px",
        "success_under_5px",
        "diverged",
        "error",
        *[f"pred_{key}" for key in CONTROL_KEYS],
        *[f"true_{key}" for key in CONTROL_KEYS],
        *[f"{key}_abs_error" for key in CONTROL_KEYS],
        *[f"{key}_sign_correct" for key in LENS_KEYS],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            row = {key: detail.get(key) for key in fieldnames}
            pred_plan = detail.get("predicted_control_plan")
            if isinstance(pred_plan, Mapping):
                for key in CONTROL_KEYS:
                    row[f"pred_{key}"] = pred_plan.get(key)
            true_plan = detail.get("true_control_plan")
            if isinstance(true_plan, Mapping):
                for key in CONTROL_KEYS:
                    row[f"true_{key}"] = true_plan.get(key)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    test_rows = [row for row in read_jsonl(args.test_jsonl) if row.get("sample_type") == "inverse_control"]
    predictions = read_jsonl(args.predictions_jsonl)
    predictions_by_id = prediction_index(predictions)

    details = [
        evaluate_row(row, predictions_by_id.get(str(row.get("sample_id"))))
        for row in test_rows
    ]
    summary = summarize(details)
    report = {
        "test_jsonl": str(args.test_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "summary": summary,
        "details": details,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        write_csv(args.output_csv, details)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
