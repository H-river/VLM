#!/usr/bin/env python3
"""Evaluate image-only optics direction/magnitude classification outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_qwen25vl_qlora import target_for_label_mode


DIRECTION_KEYS = ("lens_x_direction", "lens_y_direction")
MAGNITUDE_KEYS = ("lens_x_magnitude_class", "lens_y_magnitude_class")
DIRECTION_CLASSES = ("negative", "zero", "positive")
MAGNITUDE_CLASSES = ("zero", "small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate optics direction classification outputs.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(None)
    return rows


def first_json_block(text: str) -> str | None:
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


def parse_raw_text(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        block = first_json_block(text)
        if block is None:
            return None
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def prediction_json(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    if isinstance(row.get("parsed_json"), dict):
        return row["parsed_json"]
    if isinstance(row.get("prediction"), dict):
        return row["prediction"]
    if isinstance(row.get("raw_text"), str):
        return parse_raw_text(row["raw_text"])
    if isinstance(row.get("control_intent"), dict):
        return row
    return None


def control_intent(row: Any) -> dict[str, Any] | None:
    prediction = prediction_json(row)
    if isinstance(prediction, dict) and isinstance(prediction.get("control_intent"), dict):
        return prediction["control_intent"]
    return None


def label_intent(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    try:
        target = target_for_label_mode(row, "direction_classification")
    except (KeyError, TypeError, ValueError):
        return None
    intent = target.get("control_intent")
    return intent if isinstance(intent, dict) else None


def has_schema(row: Any) -> bool:
    prediction = prediction_json(row)
    if not isinstance(prediction, dict):
        return False
    if prediction.get("task") != "beam_alignment_direction_classification":
        return False
    if not isinstance(prediction.get("visual_diagnosis"), dict):
        return False
    intent = prediction.get("control_intent")
    if not isinstance(intent, dict):
        return False
    return all(key in intent for key in DIRECTION_KEYS + MAGNITUDE_KEYS)


def empty_confusion(labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {label: {prediction: 0 for prediction in labels} for label in labels}


def update_confusion(
    matrix: dict[str, dict[str, int]],
    label_value: Any,
    pred_value: Any,
    classes: tuple[str, ...],
) -> None:
    label_key = label_value if label_value in classes else "invalid"
    pred_key = pred_value if pred_value in classes else "invalid"
    matrix.setdefault(label_key, {prediction: 0 for prediction in classes})
    matrix[label_key].setdefault("invalid", 0)
    if pred_key not in matrix[label_key]:
        matrix[label_key][pred_key] = 0
    matrix[label_key][pred_key] += 1


def accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def evaluate(predictions: list[Any], labels: list[Any]) -> dict[str, Any]:
    pair_count = min(len(predictions), len(labels))
    valid_json_count = 0
    schema_count = 0
    parse_error_count = 0

    direction_correct = {key: 0 for key in DIRECTION_KEYS}
    direction_total = {key: 0 for key in DIRECTION_KEYS}
    magnitude_correct = {key: 0 for key in MAGNITUDE_KEYS}
    magnitude_total = {key: 0 for key in MAGNITUDE_KEYS}
    direction_confusion = {key: empty_confusion(DIRECTION_CLASSES) for key in DIRECTION_KEYS}
    magnitude_confusion = {key: empty_confusion(MAGNITUDE_CLASSES) for key in MAGNITUDE_KEYS}
    sample_details: list[dict[str, Any]] = []

    for index, (prediction_row, label_row) in enumerate(zip(predictions, labels)):
        prediction = prediction_json(prediction_row)
        if prediction is not None:
            valid_json_count += 1
        elif isinstance(prediction_row, dict) and prediction_row.get("parse_error"):
            parse_error_count += 1

        if has_schema(prediction_row):
            schema_count += 1

        pred_intent = control_intent(prediction_row)
        expected_intent = label_intent(label_row)
        if pred_intent is None or expected_intent is None:
            continue

        detail = {
            "sample_index": prediction_row.get("sample_index", index)
            if isinstance(prediction_row, dict)
            else index,
            "label_control_intent": expected_intent,
            "predicted_control_intent": pred_intent,
            "correct": {},
        }

        for key in DIRECTION_KEYS:
            label_value = expected_intent.get(key)
            pred_value = pred_intent.get(key)
            update_confusion(direction_confusion[key], label_value, pred_value, DIRECTION_CLASSES)
            if label_value in DIRECTION_CLASSES:
                direction_total[key] += 1
                is_correct = pred_value == label_value
                if is_correct:
                    direction_correct[key] += 1
                detail["correct"][key] = is_correct

        for key in MAGNITUDE_KEYS:
            label_value = expected_intent.get(key)
            pred_value = pred_intent.get(key)
            update_confusion(magnitude_confusion[key], label_value, pred_value, MAGNITUDE_CLASSES)
            if label_value in MAGNITUDE_CLASSES:
                magnitude_total[key] += 1
                is_correct = pred_value == label_value
                if is_correct:
                    magnitude_correct[key] += 1
                detail["correct"][key] = is_correct

        sample_details.append(detail)

    overall_direction_correct = sum(direction_correct.values())
    overall_direction_total = sum(direction_total.values())

    return {
        "sample_count": pair_count,
        "prediction_count": len(predictions),
        "label_count": len(labels),
        "json_valid_rate": accuracy(valid_json_count, len(predictions)),
        "schema_validity_rate": accuracy(schema_count, len(predictions)),
        "parse_error_count": parse_error_count,
        "lens_x_direction_accuracy": accuracy(direction_correct["lens_x_direction"], direction_total["lens_x_direction"]),
        "lens_y_direction_accuracy": accuracy(direction_correct["lens_y_direction"], direction_total["lens_y_direction"]),
        "overall_direction_accuracy": accuracy(overall_direction_correct, overall_direction_total),
        "lens_x_magnitude_class_accuracy": accuracy(
            magnitude_correct["lens_x_magnitude_class"], magnitude_total["lens_x_magnitude_class"]
        ),
        "lens_y_magnitude_class_accuracy": accuracy(
            magnitude_correct["lens_y_magnitude_class"], magnitude_total["lens_y_magnitude_class"]
        ),
        "direction_confusion_matrices": direction_confusion,
        "magnitude_confusion_matrices": magnitude_confusion,
        "sample_details": sample_details,
    }


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions_jsonl)
    labels = read_jsonl(args.labels_jsonl)
    report = evaluate(predictions, labels)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote direction classification report to {args.output_report}")


if __name__ == "__main__":
    main()
