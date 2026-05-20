#!/usr/bin/env python3
"""Evaluate forward_transition physics predictions against private simulator states."""

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


STATE_FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
)
DELTA_FIELDS = ("x", "y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forward_transition physics predictions.")
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


def extract_predicted_after_state(prediction: Mapping[str, Any] | None) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(prediction, Mapping):
        return None, "prediction_json_missing"
    state = prediction.get("predicted_after_state")
    if not isinstance(state, Mapping):
        return None, "predicted_after_state_missing"
    missing = [field for field in STATE_FIELDS if field not in state]
    if missing:
        return None, f"predicted_after_state_missing_fields:{','.join(missing)}"
    non_numeric = [field for field in STATE_FIELDS if not is_number(state.get(field))]
    if non_numeric:
        return None, f"predicted_after_state_non_numeric_fields:{','.join(non_numeric)}"
    return {field: float(state[field]) for field in STATE_FIELDS}, None


def extract_predicted_delta(prediction: Mapping[str, Any] | None) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(prediction, Mapping):
        return None, "prediction_json_missing"
    predicted_change = prediction.get("predicted_change")
    if not isinstance(predicted_change, Mapping):
        return None, "predicted_change_missing"

    centroid_shift = predicted_change.get("centroid_shift_px")
    if isinstance(centroid_shift, Mapping):
        missing = [field for field in DELTA_FIELDS if field not in centroid_shift]
        if missing:
            return None, f"centroid_shift_px_missing_fields:{','.join(missing)}"
        non_numeric = [field for field in DELTA_FIELDS if not is_number(centroid_shift.get(field))]
        if non_numeric:
            return None, f"centroid_shift_px_non_numeric_fields:{','.join(non_numeric)}"
        return {field: float(centroid_shift[field]) for field in DELTA_FIELDS}, None

    alt_keys = {
        "x": ("delta_centroid_x_px", "centroid_x_delta_px"),
        "y": ("delta_centroid_y_px", "centroid_y_delta_px"),
    }
    values: dict[str, float] = {}
    for axis, keys in alt_keys.items():
        found = None
        for key in keys:
            if key in predicted_change:
                found = predicted_change[key]
                break
        if not is_number(found):
            return None, f"predicted_change_missing_numeric_{axis}"
        values[axis] = float(found)
    return values, None


def true_delta(before_state: Mapping[str, Any], after_state: Mapping[str, Any]) -> dict[str, float]:
    return {
        "x": float(after_state["centroid_x_px"]) - float(before_state["centroid_x_px"]),
        "y": float(after_state["centroid_y_px"]) - float(before_state["centroid_y_px"]),
    }


def evaluate_row(row: dict[str, Any], prediction_row: dict[str, Any] | None) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "sample_id": row.get("sample_id"),
        "sample_type": row.get("sample_type"),
        "json_valid": False,
        "predicted_after_state_valid": False,
        "predicted_change_valid": False,
        "error": None,
        "change_error": None,
    }
    parsed_prediction, parse_error = extract_json_object(prediction_row)
    if parsed_prediction is None:
        detail["error"] = parse_error
        detail["change_error"] = parse_error
        return detail
    detail["json_valid"] = True

    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        detail["error"] = "private_eval_missing"
        detail["change_error"] = "private_eval_missing"
        return detail
    after_state = private_eval.get("after_state")
    before_state = private_eval.get("before_state")
    if not isinstance(after_state, Mapping):
        detail["error"] = "private_eval.after_state_missing"
        return detail

    predicted_state, state_error = extract_predicted_after_state(parsed_prediction)
    if predicted_state is None:
        detail["error"] = state_error
    else:
        detail["predicted_after_state_valid"] = True
        for field in STATE_FIELDS:
            error = abs(predicted_state[field] - float(after_state[field]))
            detail[f"{field}_abs_error"] = error
        detail["centroid_euclidean_error_px"] = math.hypot(
            float(predicted_state["centroid_x_px"]) - float(after_state["centroid_x_px"]),
            float(predicted_state["centroid_y_px"]) - float(after_state["centroid_y_px"]),
        )

    if not isinstance(before_state, Mapping):
        detail["change_error"] = "private_eval.before_state_missing"
        return detail
    predicted_delta, delta_error = extract_predicted_delta(parsed_prediction)
    if predicted_delta is None:
        detail["change_error"] = delta_error
        return detail

    target_delta = true_delta(before_state, after_state)
    detail["predicted_change_valid"] = True
    detail["delta_centroid_x_abs_error_px"] = abs(predicted_delta["x"] - target_delta["x"])
    detail["delta_centroid_y_abs_error_px"] = abs(predicted_delta["y"] - target_delta["y"])
    detail["delta_centroid_euclidean_error_px"] = math.hypot(
        predicted_delta["x"] - target_delta["x"],
        predicted_delta["y"] - target_delta["y"],
    )
    return detail


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric_mean(details: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in details if is_number(row.get(key))]
    return safe_mean(values)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(details)
    json_valid = sum(1 for row in details if row.get("json_valid"))
    state_valid = sum(1 for row in details if row.get("predicted_after_state_valid"))
    change_valid = sum(1 for row in details if row.get("predicted_change_valid"))
    error_counts: dict[str, int] = {}
    for row in details:
        for key in ("error", "change_error"):
            error = row.get(key)
            if error:
                error_counts[str(error)] = error_counts.get(str(error), 0) + 1

    return {
        "count": count,
        "json_valid_rate": rate(json_valid, count),
        "invalid_json_rate": rate(count - json_valid, count),
        "predicted_after_state_valid_rate": rate(state_valid, count),
        "predicted_after_state_missing_or_invalid_rate": rate(count - state_valid, count),
        "predicted_change_valid_rate": rate(change_valid, count),
        "predicted_change_missing_or_invalid_rate": rate(count - change_valid, count),
        "centroid_x_mae_px": metric_mean(details, "centroid_x_px_abs_error"),
        "centroid_y_mae_px": metric_mean(details, "centroid_y_px_abs_error"),
        "centroid_euclidean_mae_px": metric_mean(details, "centroid_euclidean_error_px"),
        "sigma_x_mae_px": metric_mean(details, "sigma_x_px_abs_error"),
        "sigma_y_mae_px": metric_mean(details, "sigma_y_px_abs_error"),
        "peak_intensity_mae": metric_mean(details, "peak_intensity_abs_error"),
        "delta_centroid_x_mae_px": metric_mean(details, "delta_centroid_x_abs_error_px"),
        "delta_centroid_y_mae_px": metric_mean(details, "delta_centroid_y_abs_error_px"),
        "delta_centroid_euclidean_mae_px": metric_mean(details, "delta_centroid_euclidean_error_px"),
        "error_counts": error_counts,
    }


def write_csv(path: Path, details: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "sample_type",
        "json_valid",
        "predicted_after_state_valid",
        "predicted_change_valid",
        "centroid_x_px_abs_error",
        "centroid_y_px_abs_error",
        "centroid_euclidean_error_px",
        "sigma_x_px_abs_error",
        "sigma_y_px_abs_error",
        "peak_intensity_abs_error",
        "delta_centroid_x_abs_error_px",
        "delta_centroid_y_abs_error_px",
        "delta_centroid_euclidean_error_px",
        "error",
        "change_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            writer.writerow({key: detail.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    test_rows = [row for row in read_jsonl(args.test_jsonl) if row.get("sample_type") == "forward_transition"]
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
