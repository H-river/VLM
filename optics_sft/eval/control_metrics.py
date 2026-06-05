"""Shared control-plan evaluation metrics for SFT benchmarks."""

from __future__ import annotations

import json
import math
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


def parse_json_completion(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def extract_prediction(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    if isinstance(row.get("prediction"), dict):
        return row["prediction"]
    if isinstance(row.get("parsed_json"), dict):
        return row["parsed_json"]
    raw_text = row.get("raw_text")
    if isinstance(raw_text, str):
        return parse_json_completion(raw_text)
    generated = row.get("generated_text")
    if isinstance(generated, str):
        return parse_json_completion(generated)
    return None


def control_plan(row: Any) -> dict[str, Any] | None:
    prediction = extract_prediction(row)
    if isinstance(prediction, dict) and isinstance(prediction.get("control_plan"), dict):
        return prediction["control_plan"]
    if isinstance(row, dict) and isinstance(row.get("control_plan"), dict):
        return row["control_plan"]
    target = row.get("target") if isinstance(row, dict) else None
    if isinstance(target, dict) and isinstance(target.get("control_plan"), dict):
        return target["control_plan"]
    label = row.get("label") if isinstance(row, dict) else None
    if isinstance(label, dict) and isinstance(label.get("control_plan"), dict):
        return label["control_plan"]
    return None


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


def sign_accuracy(predictions: list[Any], labels: list[Any]) -> dict[str, float | None]:
    pairs = paired_control_values(predictions, labels)
    accuracies: dict[str, float | None] = {}
    for key, values in pairs.items():
        total = 0
        correct = 0
        for pred_value, label_value in values:
            label_sign = sign(label_value)
            if label_sign == 0:
                continue
            total += 1
            if sign(pred_value) == label_sign:
                correct += 1
        accuracies[key] = correct / total if total else None
    return accuracies


def overall_lens_sign_accuracy(predictions: list[Any], labels: list[Any]) -> float | None:
    pairs = paired_control_values(predictions, labels)
    total = 0
    correct = 0
    for key in LENS_KEYS:
        for pred_value, label_value in pairs[key]:
            label_sign = sign(label_value)
            if label_sign == 0:
                continue
            total += 1
            if sign(pred_value) == label_sign:
                correct += 1
    return correct / total if total else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_benchmark(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    model_name: str,
) -> dict[str, Any]:
    valid_json = 0
    parsed_predictions: list[dict[str, Any]] = []
    for row in predictions:
        parsed = extract_prediction(row)
        parsed_predictions.append({"parsed_json": parsed, "raw_text": row.get("generated_text")})
        if parsed is not None:
            valid_json += 1

    mae = action_mae(parsed_predictions, labels)
    lens_mae_values = [value for key, value in mae.items() if key in LENS_KEYS and value is not None]
    return {
        "model_name": model_name,
        "num_examples": len(predictions),
        "json_valid_rate": valid_json / len(predictions) if predictions else None,
        "control_plan_present_rate": sum(
            1 for row in parsed_predictions if isinstance(control_plan(row), dict)
        )
        / len(predictions)
        if predictions
        else None,
        "action_mae": mae,
        "mean_lens_action_mae": mean(lens_mae_values),
        "sign_accuracy": sign_accuracy(parsed_predictions, labels),
        "overall_lens_sign_accuracy": overall_lens_sign_accuracy(parsed_predictions, labels),
    }
