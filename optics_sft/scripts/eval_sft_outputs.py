#!/usr/bin/env python3
"""Evaluate generated optics SFT JSON control plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTROL_KEYS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate optics SFT JSON outputs.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def read_json_or_jsonl_lenient(path: Path) -> tuple[list[Any], int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], 0
    if path.suffix == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [None], 1
        return (parsed if isinstance(parsed, list) else [parsed]), 0

    rows: list[Any] = []
    invalid = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            invalid += 1
            rows.append(None)
    return rows, invalid


def extract_prediction(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    if isinstance(row.get("prediction"), dict):
        return row["prediction"]
    if all(key in row for key in ("task", "diagnosis", "control_plan", "confidence")):
        return row
    if isinstance(row.get("parsed_json"), dict):
        return row["parsed_json"]
    raw_text = row.get("raw_text")
    if isinstance(raw_text, str):
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def control_plan(row: Any) -> dict[str, Any] | None:
    prediction = extract_prediction(row)
    if prediction is not None and isinstance(prediction.get("control_plan"), dict):
        return prediction["control_plan"]
    if not isinstance(row, dict):
        return None
    if isinstance(row.get("control_plan"), dict):
        return row["control_plan"]
    label = row.get("label")
    if isinstance(label, dict) and isinstance(label.get("control_plan"), dict):
        return label["control_plan"]
    return None


def has_required_keys(row: Any) -> bool:
    prediction = extract_prediction(row)
    if not isinstance(prediction, dict):
        return False
    if not all(key in prediction for key in ("task", "diagnosis", "control_plan", "confidence")):
        return False
    plan = prediction.get("control_plan")
    return isinstance(plan, dict) and all(key in plan for key in CONTROL_KEYS)


def generated_json_is_valid(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if isinstance(row.get("json_valid"), bool):
        return row["json_valid"]
    return extract_prediction(row) is not None


def has_parse_error(row: Any) -> bool:
    return isinstance(row, dict) and isinstance(row.get("parse_error"), str) and bool(row["parse_error"])


def has_numeric_actuators(row: Any) -> bool:
    plan = control_plan(row)
    return isinstance(plan, dict) and all(is_number(plan.get(key)) for key in CONTROL_KEYS)


def action_mae(predictions: list[Any], labels: list[Any]) -> dict[str, float | None]:
    totals = {key: 0.0 for key in CONTROL_KEYS}
    counts = {key: 0 for key in CONTROL_KEYS}

    for pred_row, label_row in zip(predictions, labels):
        pred_plan = control_plan(pred_row)
        label_plan = control_plan(label_row)
        if pred_plan is None or label_plan is None:
            continue
        for key in CONTROL_KEYS:
            pred_value = pred_plan.get(key)
            label_value = label_plan.get(key)
            if is_number(pred_value) and is_number(label_value):
                totals[key] += abs(float(pred_value) - float(label_value))
                counts[key] += 1

    return {
        key: (totals[key] / counts[key] if counts[key] else None)
        for key in CONTROL_KEYS
    }


def actuator_range_violation_rate(
    predictions: list[Any], min_delta_mm: float = -5.0, max_delta_mm: float = 5.0
) -> float | None:
    checked = 0
    violations = 0
    for row in predictions:
        plan = control_plan(row)
        if plan is None:
            continue
        for key in CONTROL_KEYS:
            value = plan.get(key)
            if is_number(value):
                checked += 1
                if not min_delta_mm <= float(value) <= max_delta_mm:
                    violations += 1
    return violations / checked if checked else None


def main() -> None:
    args = parse_args()
    predictions, invalid_predictions = read_json_or_jsonl_lenient(args.predictions_jsonl)
    if args.labels_jsonl is not None:
        labels, invalid_labels = read_json_or_jsonl_lenient(args.labels_jsonl)
    else:
        labels, invalid_labels = [], 0

    prediction_count = len(predictions)
    valid_json_count = sum(1 for row in predictions if generated_json_is_valid(row))
    required_key_count = sum(1 for row in predictions if has_required_keys(row))
    numeric_actuator_count = sum(1 for row in predictions if has_numeric_actuators(row))
    parse_error_count = sum(1 for row in predictions if has_parse_error(row))

    report = {
        "sample_count": prediction_count,
        "prediction_count": prediction_count,
        "label_count": len(labels),
        "json_valid_rate": valid_json_count / prediction_count if prediction_count else None,
        "required_schema_key_validity_rate": required_key_count / prediction_count if prediction_count else None,
        "required_key_rate": required_key_count / prediction_count if prediction_count else None,
        "numeric_actuator_rate": numeric_actuator_count / prediction_count if prediction_count else None,
        "parse_error_count": parse_error_count,
        "invalid_prediction_json_lines": invalid_predictions,
        "invalid_label_json_lines": invalid_labels,
        "action_mae": action_mae(predictions, labels) if labels else None,
        "actuator_range_violation_rate": actuator_range_violation_rate(predictions),
        "simulator_post_action_error": None,
        "todo": "Connect optional simulator post-action evaluation when the control loop is ready.",
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote evaluation report to {args.output_report}")


if __name__ == "__main__":
    main()
