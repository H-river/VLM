#!/usr/bin/env python3
"""Evaluate generated optics SFT JSON control plans."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CONTROL_KEYS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
LENS_KEYS = ("lens_x_delta_mm", "lens_y_delta_mm")
SIGN_EPSILON = 1e-6


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sign(value: float, epsilon: float = SIGN_EPSILON) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


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


def paired_control_values(
    predictions: list[Any],
    labels: list[Any],
) -> dict[str, list[tuple[float, float]]]:
    pairs: dict[str, list[tuple[float, float]]] = {key: [] for key in CONTROL_KEYS}

    for pred_row, label_row in zip(predictions, labels):
        pred_plan = control_plan(pred_row)
        label_plan = control_plan(label_row)
        if pred_plan is None or label_plan is None:
            continue
        for key in CONTROL_KEYS:
            pred_value = pred_plan.get(key)
            label_value = label_plan.get(key)
            if is_number(pred_value) and is_number(label_value):
                pairs[key].append((float(pred_value), float(label_value)))

    return pairs


def label_control_values(labels: list[Any]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {key: [] for key in CONTROL_KEYS}
    for label_row in labels:
        label_plan = control_plan(label_row)
        if label_plan is None:
            continue
        for key in CONTROL_KEYS:
            label_value = label_plan.get(key)
            if is_number(label_value):
                values[key].append(float(label_value))
    return values


def action_mae(predictions: list[Any], labels: list[Any]) -> dict[str, float | None]:
    pairs = paired_control_values(predictions, labels)
    return {
        key: (
            sum(abs(pred_value - label_value) for pred_value, label_value in values) / len(values)
            if values
            else None
        )
        for key, values in pairs.items()
    }


def sign_accuracy(
    predictions: list[Any],
    labels: list[Any],
    epsilon: float = SIGN_EPSILON,
) -> dict[str, float | None]:
    pairs = paired_control_values(predictions, labels)
    accuracies: dict[str, float | None] = {}
    for key, values in pairs.items():
        total = 0
        correct = 0
        for pred_value, label_value in values:
            label_sign = sign(label_value, epsilon)
            if label_sign == 0:
                continue
            total += 1
            if sign(pred_value, epsilon) == label_sign:
                correct += 1
        accuracies[key] = correct / total if total else None
    return accuracies


def overall_lens_sign_accuracy(
    predictions: list[Any],
    labels: list[Any],
    epsilon: float = SIGN_EPSILON,
) -> float | None:
    pairs = paired_control_values(predictions, labels)
    total = 0
    correct = 0
    for key in LENS_KEYS:
        for pred_value, label_value in pairs[key]:
            label_sign = sign(label_value, epsilon)
            if label_sign == 0:
                continue
            total += 1
            if sign(pred_value, epsilon) == label_sign:
                correct += 1
    return correct / total if total else None


def zero_action_baseline_mae(labels: list[Any]) -> dict[str, float | None]:
    values = label_control_values(labels)
    return {
        key: (sum(abs(label_value) for label_value in key_values) / len(key_values) if key_values else None)
        for key, key_values in values.items()
    }


def zero_action_baseline_sign_accuracy(
    labels: list[Any],
    epsilon: float = SIGN_EPSILON,
) -> dict[str, float | None]:
    values = label_control_values(labels)
    accuracies: dict[str, float | None] = {}
    for key, key_values in values.items():
        total = 0
        correct = 0
        for label_value in key_values:
            label_sign = sign(label_value, epsilon)
            if label_sign == 0:
                continue
            total += 1
            if sign(0.0, epsilon) == label_sign:
                correct += 1
        accuracies[key] = correct / total if total else None
    return accuracies


def model_vs_zero_baseline(
    model_mae: dict[str, float | None],
    baseline_mae: dict[str, float | None],
) -> dict[str, dict[str, float | None]]:
    comparison: dict[str, dict[str, float | None]] = {}
    for key in CONTROL_KEYS:
        model_value = model_mae.get(key)
        baseline_value = baseline_mae.get(key)
        improvement = (
            baseline_value - model_value
            if model_value is not None and baseline_value is not None
            else None
        )
        relative_improvement = (
            improvement / baseline_value
            if improvement is not None and baseline_value is not None and baseline_value > 0
            else None
        )
        comparison[key] = {
            "model_mae": model_value,
            "zero_baseline_mae": baseline_value,
            "improvement": improvement,
            "relative_improvement": relative_improvement,
        }
    return comparison


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    if not values:
        return None
    value_mean = sum(values) / len(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / len(values))


def prediction_statistics(predictions: list[Any], labels: list[Any]) -> dict[str, dict[str, float | int | None]]:
    pairs = paired_control_values(predictions, labels)
    return {
        key: {
            "count": len(values),
            "pred_mean": mean([pred_value for pred_value, _ in values]),
            "label_mean": mean([label_value for _, label_value in values]),
            "pred_std": std([pred_value for pred_value, _ in values]),
            "label_std": std([label_value for _, label_value in values]),
            "bias": mean([pred_value - label_value for pred_value, label_value in values]),
        }
        for key, values in pairs.items()
    }


def sample_details(
    predictions: list[Any],
    labels: list[Any],
    epsilon: float = SIGN_EPSILON,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for index, (pred_row, label_row) in enumerate(zip(predictions, labels)):
        pred_plan = control_plan(pred_row)
        label_plan = control_plan(label_row)
        absolute_error: dict[str, float | None] = {}
        sign_correct: dict[str, bool | None] = {}

        for key in CONTROL_KEYS:
            pred_value = pred_plan.get(key) if isinstance(pred_plan, dict) else None
            label_value = label_plan.get(key) if isinstance(label_plan, dict) else None
            if is_number(pred_value) and is_number(label_value):
                pred_float = float(pred_value)
                label_float = float(label_value)
                absolute_error[key] = abs(pred_float - label_float)
                sign_correct[key] = (
                    None
                    if sign(label_float, epsilon) == 0
                    else sign(pred_float, epsilon) == sign(label_float, epsilon)
                )
            else:
                absolute_error[key] = None
                sign_correct[key] = None

        sample_index = pred_row.get("sample_index", index) if isinstance(pred_row, dict) else index
        details.append(
            {
                "sample_index": sample_index,
                "label_control_plan": label_plan,
                "predicted_control_plan": pred_plan,
                "absolute_error": absolute_error,
                "sign_correct": sign_correct,
            }
        )

    return details


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
    mae = action_mae(predictions, labels) if labels else None
    baseline_mae = zero_action_baseline_mae(labels) if labels else None

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
        "action_mae": mae,
        "zero_action_baseline_mae": baseline_mae,
        "zero_action_baseline_sign_accuracy": zero_action_baseline_sign_accuracy(labels) if labels else None,
        "model_vs_zero_baseline": model_vs_zero_baseline(mae, baseline_mae)
        if mae is not None and baseline_mae is not None
        else None,
        "sign_epsilon": SIGN_EPSILON,
        "per_actuator_sign_accuracy": sign_accuracy(predictions, labels) if labels else None,
        "overall_lens_sign_accuracy": overall_lens_sign_accuracy(predictions, labels) if labels else None,
        "per_actuator_prediction_statistics": prediction_statistics(predictions, labels) if labels else None,
        "sample_details": sample_details(predictions, labels) if labels else None,
        "actuator_range_violation_rate": actuator_range_violation_rate(predictions),
        "simulator_post_action_error": None,
        "todo": "Connect optional simulator post-action evaluation when the control loop is ready.",
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote evaluation report to {args.output_report}")


if __name__ == "__main__":
    main()
