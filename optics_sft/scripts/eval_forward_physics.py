#!/usr/bin/env python3
"""Evaluate forward_transition physics predictions against private simulator states."""

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


CENTROID_STATE_FIELDS = ("centroid_x_px", "centroid_y_px")
FULL_STATE_EXTRA_FIELDS = ("sigma_x_px", "sigma_y_px", "peak_intensity")
FULL_STATE_FIELDS = CENTROID_STATE_FIELDS + FULL_STATE_EXTRA_FIELDS
METRICS_MODES = ("auto", "centroid_only", "full_state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forward_transition physics predictions.")
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--metrics-mode",
        choices=METRICS_MODES,
        default="auto",
        help="Metric fields to require. auto infers from row target_mode or prediction task.",
    )
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


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return parsed if math.isfinite(parsed) else None


def is_number(value: Any) -> bool:
    return number(value) is not None


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


def target_mode(row: Mapping[str, Any]) -> str | None:
    target = row.get("target")
    if isinstance(target, Mapping):
        mode = target.get("target_mode")
        if mode in ("centroid_only", "full_state"):
            return str(mode)
    return None


def infer_metrics_mode(row: Mapping[str, Any], prediction: Mapping[str, Any] | None, requested: str) -> str:
    if requested != "auto":
        return requested
    mode = target_mode(row)
    if mode is not None:
        return mode
    if isinstance(prediction, Mapping):
        pred_mode = prediction.get("target_mode")
        if pred_mode in ("centroid_only", "full_state"):
            return str(pred_mode)
        task = prediction.get("task")
        if task == "forward_centroid_transition":
            return "centroid_only"
        if task in ("forward_optics_prediction", "physics_aware_forward_transition"):
            return "full_state"
    return "full_state"


def required_state_fields(metrics_mode: str) -> tuple[str, ...]:
    if metrics_mode == "centroid_only":
        return CENTROID_STATE_FIELDS
    if metrics_mode == "full_state":
        return FULL_STATE_FIELDS
    raise ValueError(f"Unsupported metrics_mode: {metrics_mode}")


def extract_predicted_after_state(
    prediction: Mapping[str, Any] | None,
    fields: tuple[str, ...],
) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(prediction, Mapping):
        return None, "prediction_json_missing"
    state = prediction.get("predicted_after_state")
    if not isinstance(state, Mapping):
        return None, "predicted_after_state_missing"
    missing = [field for field in fields if field not in state]
    if missing:
        return None, f"predicted_after_state_missing_fields:{','.join(missing)}"
    non_numeric = [field for field in fields if not is_number(state.get(field))]
    if non_numeric:
        return None, f"predicted_after_state_non_numeric_fields:{','.join(non_numeric)}"
    return {field: float(number(state[field])) for field in fields if number(state[field]) is not None}, None


def extract_predicted_delta(prediction: Mapping[str, Any] | None) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(prediction, Mapping):
        return None, "prediction_json_missing"
    predicted_change = prediction.get("predicted_change")
    if not isinstance(predicted_change, Mapping):
        return None, "predicted_change_missing"

    direct = {
        "x": predicted_change.get("delta_centroid_x_px"),
        "y": predicted_change.get("delta_centroid_y_px"),
    }
    if all(is_number(value) for value in direct.values()):
        return {axis: float(number(value)) for axis, value in direct.items() if number(value) is not None}, None

    centroid_shift = predicted_change.get("centroid_shift_px")
    if isinstance(centroid_shift, Mapping):
        missing = [field for field in ("x", "y") if field not in centroid_shift]
        if missing:
            return None, f"centroid_shift_px_missing_fields:{','.join(missing)}"
        non_numeric = [field for field in ("x", "y") if not is_number(centroid_shift.get(field))]
        if non_numeric:
            return None, f"centroid_shift_px_non_numeric_fields:{','.join(non_numeric)}"
        return {field: float(number(centroid_shift[field])) for field in ("x", "y")}, None

    alt_keys = {
        "x": ("centroid_x_delta_px",),
        "y": ("centroid_y_delta_px",),
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
        parsed = number(found)
        if parsed is not None:
            values[axis] = parsed
    return values, None


def true_delta(before_state: Mapping[str, Any], after_state: Mapping[str, Any]) -> dict[str, float]:
    return {
        "x": float(after_state["centroid_x_px"]) - float(before_state["centroid_x_px"]),
        "y": float(after_state["centroid_y_px"]) - float(before_state["centroid_y_px"]),
    }


def evaluate_row(
    row: dict[str, Any],
    prediction_row: dict[str, Any] | None,
    requested_metrics_mode: str,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "sample_id": row.get("sample_id"),
        "sample_type": row.get("sample_type"),
        "metrics_mode": None,
        "json_valid": False,
        "predicted_after_state_valid": False,
        "predicted_change_valid": False,
        "error": None,
        "change_error": None,
    }
    parsed_prediction, parse_error = extract_json_object(prediction_row)
    if parsed_prediction is None:
        detail["metrics_mode"] = target_mode(row) or ("centroid_only" if requested_metrics_mode == "centroid_only" else "full_state")
        detail["error"] = parse_error
        detail["change_error"] = parse_error
        return detail
    detail["json_valid"] = True

    metrics_mode = infer_metrics_mode(row, parsed_prediction, requested_metrics_mode)
    detail["metrics_mode"] = metrics_mode
    state_fields = required_state_fields(metrics_mode)

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

    predicted_state, state_error = extract_predicted_after_state(parsed_prediction, state_fields)
    if predicted_state is None:
        detail["error"] = state_error
    else:
        detail["predicted_after_state_valid"] = True
        for field in state_fields:
            parsed_true = number(after_state.get(field))
            if parsed_true is not None and field in predicted_state:
                detail[f"{field}_abs_error"] = abs(predicted_state[field] - parsed_true)
        detail["centroid_euclidean_error_px"] = math.hypot(
            float(predicted_state["centroid_x_px"]) - float(after_state["centroid_x_px"]),
            float(predicted_state["centroid_y_px"]) - float(after_state["centroid_y_px"]),
        )

        if metrics_mode == "centroid_only" and isinstance(parsed_prediction.get("predicted_after_state"), Mapping):
            optional_state = parsed_prediction["predicted_after_state"]
            for field in FULL_STATE_EXTRA_FIELDS:
                parsed_pred = number(optional_state.get(field))
                parsed_true = number(after_state.get(field))
                if parsed_pred is not None and parsed_true is not None:
                    detail[f"diagnostic_{field}_abs_error"] = abs(parsed_pred - parsed_true)

    if not isinstance(before_state, Mapping):
        detail["change_error"] = "private_eval.before_state_missing"
        return detail
    target_delta = true_delta(before_state, after_state)
    detail["true_delta_centroid_x_px"] = target_delta["x"]
    detail["true_delta_centroid_y_px"] = target_delta["y"]
    detail["no_action_delta_centroid_euclidean_error_px"] = math.hypot(target_delta["x"], target_delta["y"])

    predicted_delta, delta_error = extract_predicted_delta(parsed_prediction)
    if predicted_delta is None:
        detail["change_error"] = delta_error
        return detail

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


def metric_values(details: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in details if is_number(row.get(key))]


def metric_mean(details: list[dict[str, Any]], key: str) -> float | None:
    return safe_mean(metric_values(details, key))


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def dominant_metrics_mode(details: list[dict[str, Any]], requested: str) -> str:
    if requested != "auto":
        return requested
    counts: dict[str, int] = {}
    for row in details:
        mode = row.get("metrics_mode")
        if isinstance(mode, str):
            counts[mode] = counts.get(mode, 0) + 1
    if not counts:
        return "full_state"
    return max(counts.items(), key=lambda item: item[1])[0]


def mean_delta_baseline(details: list[dict[str, Any]]) -> float | None:
    xs = metric_values(details, "true_delta_centroid_x_px")
    ys = metric_values(details, "true_delta_centroid_y_px")
    if not xs or not ys or len(xs) != len(ys):
        return None
    mean_x = safe_mean(xs)
    mean_y = safe_mean(ys)
    if mean_x is None or mean_y is None:
        return None
    errors = [math.hypot(mean_x - x, mean_y - y) for x, y in zip(xs, ys)]
    return safe_mean(errors)


def summarize(details: list[dict[str, Any]], requested_metrics_mode: str) -> dict[str, Any]:
    count = len(details)
    metrics_mode = dominant_metrics_mode(details, requested_metrics_mode)
    json_valid = sum(1 for row in details if row.get("json_valid"))
    state_valid = sum(1 for row in details if row.get("predicted_after_state_valid"))
    change_valid = sum(1 for row in details if row.get("predicted_change_valid"))
    error_counts: dict[str, int] = {}
    for row in details:
        for key in ("error", "change_error"):
            error = row.get(key)
            if error:
                error_counts[str(error)] = error_counts.get(str(error), 0) + 1

    summary: dict[str, Any] = {
        "count": count,
        "metrics_mode": metrics_mode,
        "json_valid_rate": rate(json_valid, count),
        "invalid_json_rate": rate(count - json_valid, count),
        "predicted_after_state_valid_rate": rate(state_valid, count),
        "predicted_after_state_missing_or_invalid_rate": rate(count - state_valid, count),
        "predicted_change_valid_rate": rate(change_valid, count),
        "predicted_change_missing_or_invalid_rate": rate(count - change_valid, count),
        "centroid_x_mae_px": metric_mean(details, "centroid_x_px_abs_error"),
        "centroid_y_mae_px": metric_mean(details, "centroid_y_px_abs_error"),
        "centroid_euclidean_mae_px": metric_mean(details, "centroid_euclidean_error_px"),
        "delta_centroid_x_mae_px": metric_mean(details, "delta_centroid_x_abs_error_px"),
        "delta_centroid_y_mae_px": metric_mean(details, "delta_centroid_y_abs_error_px"),
        "delta_centroid_euclidean_mae_px": metric_mean(details, "delta_centroid_euclidean_error_px"),
        "model_delta_centroid_euclidean_mae_px": metric_mean(details, "delta_centroid_euclidean_error_px"),
        "no_action_delta_centroid_euclidean_mae_px": metric_mean(details, "no_action_delta_centroid_euclidean_error_px"),
        "mean_delta_centroid_euclidean_mae_px": mean_delta_baseline(details),
        "error_counts": error_counts,
    }
    if metrics_mode == "full_state":
        summary.update(
            {
                "sigma_x_mae_px": metric_mean(details, "sigma_x_px_abs_error"),
                "sigma_y_mae_px": metric_mean(details, "sigma_y_px_abs_error"),
                "peak_intensity_mae": metric_mean(details, "peak_intensity_abs_error"),
            }
        )
    else:
        diagnostics = {
            "diagnostic_sigma_x_mae_px_if_predicted": metric_mean(details, "diagnostic_sigma_x_px_abs_error"),
            "diagnostic_sigma_y_mae_px_if_predicted": metric_mean(details, "diagnostic_sigma_y_px_abs_error"),
            "diagnostic_peak_intensity_mae_if_predicted": metric_mean(details, "diagnostic_peak_intensity_abs_error"),
        }
        summary["optional_full_state_diagnostics"] = diagnostics
    return summary


def write_csv(path: Path, details: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "sample_type",
        "metrics_mode",
        "json_valid",
        "predicted_after_state_valid",
        "predicted_change_valid",
        "centroid_x_px_abs_error",
        "centroid_y_px_abs_error",
        "centroid_euclidean_error_px",
        "sigma_x_px_abs_error",
        "sigma_y_px_abs_error",
        "peak_intensity_abs_error",
        "diagnostic_sigma_x_px_abs_error",
        "diagnostic_sigma_y_px_abs_error",
        "diagnostic_peak_intensity_abs_error",
        "delta_centroid_x_abs_error_px",
        "delta_centroid_y_abs_error_px",
        "delta_centroid_euclidean_error_px",
        "no_action_delta_centroid_euclidean_error_px",
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
        evaluate_row(row, predictions_by_id.get(str(row.get("sample_id"))), args.metrics_mode)
        for row in test_rows
    ]
    summary = summarize(details, args.metrics_mode)
    report = {
        "test_jsonl": str(args.test_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "requested_metrics_mode": args.metrics_mode,
        "metrics_mode": summary.get("metrics_mode"),
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
