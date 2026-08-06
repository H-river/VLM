#!/usr/bin/env python3
"""Quantify default-value bias and baseline-relative quantitative skill."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import read_jsonl


FORWARD_FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "peak_intensity",
    "sigma_x_px",
    "sigma_y_px",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--details-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sign(value: float, deadband: float = 0.05) -> int:
    return 1 if value > deadband else -1 if value < -deadband else 0


def forward_metrics(
    records: list[dict[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    details: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [record for record in records if record["task_type"] == "forward_prediction"]
    parsed_rows: list[tuple[dict[str, Any], dict[str, float], dict[str, float]]] = []
    field_stats: dict[str, dict[str, Any]] = {}
    for record in rows:
        prediction = predictions.get(str(record["example_id"]), {}).get("parsed_json")
        predicted_change = prediction.get("answer", {}).get("change", {}) if isinstance(prediction, Mapping) else {}
        target_change = record["target"]["answer"]["change"]
        predicted = {field: finite_float(predicted_change.get(field)) for field in FORWARD_FIELDS}
        if all(value is not None for value in predicted.values()):
            parsed_rows.append((record, target_change, predicted))

    for field in FORWARD_FIELDS:
        targets = [float(target[field]) for _, target, _ in parsed_rows]
        predicted = [float(output[field]) for _, _, output in parsed_rows]
        model_abs_error = [abs(output - target) for output, target in zip(predicted, targets)]
        zero_abs_error = [abs(target) for target in targets]
        material = [index for index, target in enumerate(targets) if abs(target) > 0.05]
        denominator = sum(zero_abs_error)
        field_stats[field] = {
            "parseable_count": len(predicted),
            "exact_zero_count": sum(abs(value) <= 1e-12 for value in predicted),
            "near_zero_0_05_count": sum(abs(value) <= 0.05 for value in predicted),
            "median_absolute_prediction": statistics.median(abs(value) for value in predicted),
            "model_mae": sum(model_abs_error) / len(model_abs_error),
            "zero_baseline_mae": sum(zero_abs_error) / len(zero_abs_error),
            "baseline_relative_skill": 1.0 - sum(model_abs_error) / denominator if denominator else 0.0,
            "beats_zero_rate": sum(error < zero for error, zero in zip(model_abs_error, zero_abs_error)) / len(model_abs_error),
            "material_sign_accuracy": (
                sum(sign(predicted[index]) == sign(targets[index]) for index in material) / len(material)
                if material
                else 0.0
            ),
            "material_count": len(material),
        }

    all_near_zero = sum(
        all(abs(float(output[field])) <= 0.05 for field in FORWARD_FIELDS)
        for _, _, output in parsed_rows
    )
    strict_successes = []
    baseline_improvements = []
    for record, target, output in parsed_rows:
        group_ratios = []
        for fields in (
            ("centroid_x_px", "centroid_y_px"),
            ("sigma_x_px", "sigma_y_px"),
            ("peak_intensity",),
        ):
            zero_error = math.sqrt(sum(float(target[field]) ** 2 for field in fields))
            model_error = math.sqrt(
                sum((float(output[field]) - float(target[field])) ** 2 for field in fields)
            )
            if zero_error > 1e-9:
                group_ratios.append(model_error / zero_error)
        improvement = 1.0 - sum(group_ratios) / len(group_ratios) if group_ratios else 0.0
        baseline_improvements.append(improvement)
        material_fields = [field for field in FORWARD_FIELDS if abs(float(target[field])) > 0.05]
        material_signs_correct = bool(material_fields) and all(
            sign(float(output[field])) == sign(float(target[field])) for field in material_fields
        )
        sensor_pass = float(details[record["example_id"]]["task_score"]) == 1.0
        if sensor_pass and improvement >= 0.20 and material_signs_correct:
            strict_successes.append(record["example_id"])
    target_magnitude_score: list[tuple[float, float]] = []
    for record, target, _ in parsed_rows:
        magnitude = math.sqrt(
            float(target["centroid_x_px"]) ** 2
            + float(target["centroid_y_px"]) ** 2
            + float(target["sigma_x_px"]) ** 2
            + float(target["sigma_y_px"]) ** 2
        )
        target_magnitude_score.append((magnitude, float(details[record["example_id"]]["task_score"])))
    target_magnitude_score.sort()
    midpoint = len(target_magnitude_score) // 2
    low = target_magnitude_score[:midpoint]
    high = target_magnitude_score[midpoint:]
    return {
        "record_count": len(rows),
        "parseable_count": len(parsed_rows),
        "nominal_score_1_count": sum(float(details[row["example_id"]]["task_score"]) == 1.0 for row in rows),
        "all_fields_near_zero_0_05_count": all_near_zero,
        "all_fields_near_zero_0_05_rate": all_near_zero / len(parsed_rows),
        "mean_improvement_over_zero_baseline": sum(baseline_improvements) / len(baseline_improvements),
        "beats_zero_by_20pct_rate": sum(value >= 0.20 for value in baseline_improvements) / len(baseline_improvements),
        "strict_success_definition": (
            "rubric_v2_score_1 AND at_least_20pct_mean_error_reduction_vs_zero "
            "AND correct_sign_for_every_target_component_above_0.05"
        ),
        "strict_success_count": len(strict_successes),
        "strict_success_example_ids": strict_successes,
        "field_metrics": field_stats,
        "mean_task_score_by_target_magnitude_half": {
            "lower_half": sum(score for _, score in low) / len(low),
            "upper_half": sum(score for _, score in high) / len(high),
        },
    }


def status_bias(
    records: list[dict[str, Any]], predictions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    output = {}
    for task in ("information_sufficiency", "diagnosis", "constrained_intervention"):
        rows = [record for record in records if record["task_type"] == task]
        targets = Counter(str(record["target"]["status"]) for record in rows)
        predicted = Counter()
        correct = 0
        for record in rows:
            value = predictions.get(str(record["example_id"]), {}).get("parsed_json")
            status = str(value.get("status")) if isinstance(value, Mapping) else "unparsed"
            predicted[status] += 1
            correct += status == str(record["target"]["status"])
        majority = max(targets, key=targets.get)
        output[task] = {
            "target_counts": dict(targets),
            "prediction_counts": dict(predicted),
            "status_accuracy": correct / len(rows),
            "balanced_majority_baseline_accuracy": targets[majority] / len(rows),
        }
    return output


def modality_metrics(
    records: list[dict[str, Any]], details: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        groups[(str(record["task_type"]), str(record["modality"]))].append(
            float(details[record["example_id"]]["task_score"])
        )
    return {
        task: {
            modality: {"count": len(scores), "mean_task_score": sum(scores) / len(scores)}
            for modality in ("text", "visual")
            if (scores := groups.get((task, modality)))
        }
        for task in sorted({record["task_type"] for record in records})
    }


def analyze(
    records: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions = {str(row["example_id"]): row for row in prediction_rows}
    details = {str(row["example_id"]): row for row in detail_rows}
    if set(predictions) != {str(record["example_id"]) for record in records}:
        raise ValueError("prediction IDs do not exactly match records")
    if set(details) != set(predictions):
        raise ValueError("detail IDs do not exactly match predictions")
    return {
        "record_count": len(records),
        "forward_prediction": forward_metrics(records, predictions, details),
        "status_bias": status_bias(records, predictions),
        "task_modality": modality_metrics(records, details),
    }


def main() -> None:
    args = parse_args()
    result = analyze(
        read_jsonl(args.records_jsonl),
        read_jsonl(args.predictions_jsonl),
        read_jsonl(args.details_jsonl),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
