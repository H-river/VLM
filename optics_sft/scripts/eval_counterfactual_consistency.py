#!/usr/bin/env python3
"""Evaluate counterfactual_pair physics predictions for action consistency."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ACTION_FIELDS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
IDENTICAL_ACTION_TOL_MM = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual_pair physics predictions.")
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
    prediction = prediction_row.get("prediction")
    if isinstance(prediction, dict):
        return prediction, None

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


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def extract_action(obj: Mapping[str, Any] | None, key: str) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(obj, Mapping):
        return None, "source_object_missing"
    action = obj.get(key)
    if not isinstance(action, Mapping):
        return None, f"{key}_missing"
    missing = [field for field in ACTION_FIELDS if field not in action]
    if missing:
        return None, f"{key}_missing_fields:{','.join(missing)}"
    non_numeric = [field for field in ACTION_FIELDS if not is_number(action.get(field))]
    if non_numeric:
        return None, f"{key}_non_numeric_fields:{','.join(non_numeric)}"
    return {field: float(action[field]) for field in ACTION_FIELDS}, None


def target_action(row: Mapping[str, Any], scenario: str) -> tuple[dict[str, float] | None, str | None]:
    key = f"scenario_{scenario}_control_plan"
    target = row.get("target")
    action, error = extract_action(target if isinstance(target, Mapping) else None, key)
    if action is not None:
        return action, None

    private_eval = row.get("private_eval")
    if isinstance(private_eval, Mapping):
        scenario_eval = private_eval.get(f"scenario_{scenario}")
        action, private_error = extract_action(
            scenario_eval if isinstance(scenario_eval, Mapping) else None,
            "searched_control_action",
        )
        if action is not None:
            return action, None
        return None, f"{error}; private_eval.scenario_{scenario}.{private_error}"
    return None, error


def action_abs_errors(predicted: Mapping[str, float], target: Mapping[str, float]) -> dict[str, float]:
    return {field: abs(float(predicted[field]) - float(target[field])) for field in ACTION_FIELDS}


def action_difference(action_b: Mapping[str, float], action_a: Mapping[str, float]) -> dict[str, float]:
    return {field: float(action_b[field]) - float(action_a[field]) for field in ACTION_FIELDS}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def mean_action_error(errors: Mapping[str, float] | None) -> float | None:
    if not isinstance(errors, Mapping):
        return None
    values = [float(errors[field]) for field in ACTION_FIELDS if is_number(errors.get(field))]
    return mean(values)


def actions_identical(action_a: Mapping[str, float], action_b: Mapping[str, float]) -> bool:
    return all(abs(float(action_a[field]) - float(action_b[field])) <= IDENTICAL_ACTION_TOL_MM for field in ACTION_FIELDS)


def changed_parameter_from_row(row: Mapping[str, Any]) -> str | None:
    target = row.get("target")
    if isinstance(target, Mapping) and isinstance(target.get("changed_parameter"), str):
        return target["changed_parameter"]
    private_eval = row.get("private_eval")
    if isinstance(private_eval, Mapping):
        simulator_configs = private_eval.get("simulator_configs")
        if isinstance(simulator_configs, Mapping) and isinstance(simulator_configs.get("changed_parameter"), str):
            return simulator_configs["changed_parameter"]
    return None


def evaluate_row(row: dict[str, Any], prediction_row: dict[str, Any] | None) -> dict[str, Any]:
    target = row.get("target")
    target_mapping = target if isinstance(target, Mapping) else {}
    true_changed_parameter = changed_parameter_from_row(row)
    true_should_change = parse_bool(target_mapping.get("should_action_change"))
    detail: dict[str, Any] = {
        "sample_id": row.get("sample_id"),
        "sample_type": row.get("sample_type"),
        "changed_parameter": true_changed_parameter,
        "json_valid": False,
        "required_fields_valid": False,
        "should_action_change_correct": None,
        "changed_parameter_correct": None,
        "predicted_identical_actions": None,
        "identical_action_when_should_change": None,
        "error": None,
    }

    parsed_prediction, parse_error = extract_json_object(prediction_row)
    if parsed_prediction is None:
        detail["error"] = parse_error
        return detail
    detail["json_valid"] = True

    predicted_changed_parameter = parsed_prediction.get("changed_parameter")
    predicted_should_change = parse_bool(parsed_prediction.get("should_action_change"))
    predicted_a, pred_a_error = extract_action(parsed_prediction, "scenario_a_control_plan")
    predicted_b, pred_b_error = extract_action(parsed_prediction, "scenario_b_control_plan")
    true_a, true_a_error = target_action(row, "a")
    true_b, true_b_error = target_action(row, "b")

    field_errors = []
    if not isinstance(predicted_changed_parameter, str):
        field_errors.append("changed_parameter_missing_or_non_string")
    if predicted_should_change is None:
        field_errors.append("should_action_change_missing_or_non_bool")
    if predicted_a is None:
        field_errors.append(str(pred_a_error))
    if predicted_b is None:
        field_errors.append(str(pred_b_error))
    if true_changed_parameter is None:
        field_errors.append("target_changed_parameter_missing")
    if true_should_change is None:
        field_errors.append("target_should_action_change_missing_or_non_bool")
    if true_a is None:
        field_errors.append(f"target_scenario_a_action:{true_a_error}")
    if true_b is None:
        field_errors.append(f"target_scenario_b_action:{true_b_error}")
    if field_errors:
        detail["error"] = "; ".join(field_errors)

    if isinstance(predicted_changed_parameter, str) and true_changed_parameter is not None:
        detail["predicted_changed_parameter"] = predicted_changed_parameter
        detail["changed_parameter_correct"] = predicted_changed_parameter == true_changed_parameter
    if predicted_should_change is not None and true_should_change is not None:
        detail["predicted_should_action_change"] = predicted_should_change
        detail["should_action_change_correct"] = predicted_should_change == true_should_change

    if predicted_a is None or predicted_b is None or true_a is None or true_b is None:
        return detail

    detail["required_fields_valid"] = (
        isinstance(predicted_changed_parameter, str)
        and predicted_should_change is not None
        and true_changed_parameter is not None
        and true_should_change is not None
    )
    scenario_a_errors = action_abs_errors(predicted_a, true_a)
    scenario_b_errors = action_abs_errors(predicted_b, true_b)
    predicted_difference = action_difference(predicted_b, predicted_a)
    true_difference = action_difference(true_b, true_a)
    difference_errors = action_abs_errors(predicted_difference, true_difference)

    detail["scenario_a_action_mae"] = mean_action_error(scenario_a_errors)
    detail["scenario_b_action_mae"] = mean_action_error(scenario_b_errors)
    detail["action_difference_mae"] = mean_action_error(difference_errors)
    for field, value in scenario_a_errors.items():
        detail[f"scenario_a_{field}_abs_error"] = value
    for field, value in scenario_b_errors.items():
        detail[f"scenario_b_{field}_abs_error"] = value
    for field, value in difference_errors.items():
        detail[f"action_difference_{field}_abs_error"] = value

    predicted_identical = actions_identical(predicted_a, predicted_b)
    detail["predicted_identical_actions"] = predicted_identical
    if true_should_change is True:
        detail["identical_action_when_should_change"] = predicted_identical
    return detail


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metric_mean(details: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in details if is_number(row.get(key))]
    return mean(values)


def boolean_accuracy(details: list[dict[str, Any]], key: str) -> float | None:
    usable = [row for row in details if isinstance(row.get(key), bool)]
    return rate(sum(1 for row in usable if row[key]), len(usable))


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(details)
    json_valid = sum(1 for row in details if row.get("json_valid"))
    required_valid = sum(1 for row in details if row.get("required_fields_valid"))
    should_change_true = [row for row in details if row.get("identical_action_when_should_change") is not None]
    identical_when_should_change = sum(1 for row in should_change_true if row.get("identical_action_when_should_change"))
    error_counts: dict[str, int] = {}
    for row in details:
        error = row.get("error")
        if error:
            error_counts[str(error)] = error_counts.get(str(error), 0) + 1

    return {
        "count": count,
        "json_valid_rate": rate(json_valid, count),
        "invalid_json_rate": rate(count - json_valid, count),
        "required_fields_valid_rate": rate(required_valid, count),
        "missing_or_invalid_required_fields_rate": rate(count - required_valid, count),
        "should_action_change_accuracy": boolean_accuracy(details, "should_action_change_correct"),
        "changed_parameter_accuracy": boolean_accuracy(details, "changed_parameter_correct"),
        "scenario_a_action_mae": metric_mean(details, "scenario_a_action_mae"),
        "scenario_b_action_mae": metric_mean(details, "scenario_b_action_mae"),
        "action_difference_mae": metric_mean(details, "action_difference_mae"),
        "identical_action_rate_when_should_change": rate(
            identical_when_should_change,
            len(should_change_true),
        ),
        "scenario_a_action_mae_by_field": {
            field: metric_mean(details, f"scenario_a_{field}_abs_error") for field in ACTION_FIELDS
        },
        "scenario_b_action_mae_by_field": {
            field: metric_mean(details, f"scenario_b_{field}_abs_error") for field in ACTION_FIELDS
        },
        "action_difference_mae_by_field": {
            field: metric_mean(details, f"action_difference_{field}_abs_error") for field in ACTION_FIELDS
        },
        "error_counts": error_counts,
    }


def summarize_by_changed_parameter(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        changed_parameter = row.get("changed_parameter")
        key = changed_parameter if isinstance(changed_parameter, str) else "unknown"
        grouped.setdefault(key, []).append(row)
    return {key: summarize(rows) for key, rows in sorted(grouped.items())}


def write_csv(path: Path, details: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "sample_type",
        "changed_parameter",
        "predicted_changed_parameter",
        "json_valid",
        "required_fields_valid",
        "should_action_change_correct",
        "changed_parameter_correct",
        "predicted_identical_actions",
        "identical_action_when_should_change",
        "scenario_a_action_mae",
        "scenario_b_action_mae",
        "action_difference_mae",
        *[f"scenario_a_{field}_abs_error" for field in ACTION_FIELDS],
        *[f"scenario_b_{field}_abs_error" for field in ACTION_FIELDS],
        *[f"action_difference_{field}_abs_error" for field in ACTION_FIELDS],
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            writer.writerow({key: detail.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    test_rows = [row for row in read_jsonl(args.test_jsonl) if row.get("sample_type") == "counterfactual_pair"]
    predictions = read_jsonl(args.predictions_jsonl)
    predictions_by_id = prediction_index(predictions)
    details = [
        evaluate_row(row, predictions_by_id.get(str(row.get("sample_id"))))
        for row in test_rows
    ]
    report = {
        "test_jsonl": str(args.test_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "summary": summarize(details),
        "by_changed_parameter": summarize_by_changed_parameter(details),
        "details": details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        write_csv(args.output_csv, details)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
