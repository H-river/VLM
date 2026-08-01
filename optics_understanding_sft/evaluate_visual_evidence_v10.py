#!/usr/bin/env python3
"""Evaluate calibrated visual state and paired-direction evidence records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .core import read_jsonl


STATE_FIELDS = (
    "centroid_horizontal_region",
    "centroid_vertical_region",
    "sigma_x_band",
    "sigma_y_band",
)
PAIR_FIELDS = ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row["example_id"] in result:
            raise ValueError(f"Duplicate prediction: {row['example_id']}")
        result[row["example_id"]] = row
    return result


def macro_f1(targets: list[str], predictions: list[str]) -> float:
    labels = sorted(set(targets) | set(predictions))
    scores: list[float] = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records_jsonl)
    predictions = prediction_map(args.predictions_jsonl)
    expected_ids = {row["example_id"] for row in records}
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        extra = sorted(set(predictions) - expected_ids)
        raise ValueError(f"Prediction IDs differ: missing={missing[:5]} extra={extra[:5]}")

    json_valid = 0
    schema_valid = 0
    exact = 0
    field_correct: Counter[str] = Counter()
    field_total: Counter[str] = Counter()
    field_targets: dict[str, list[str]] = defaultdict(list)
    field_predictions: dict[str, list[str]] = defaultdict(list)
    task_exact: Counter[str] = Counter()
    task_count: Counter[str] = Counter()
    source_task_exact: Counter[str] = Counter()
    source_task_count: Counter[str] = Counter()
    source_field_targets: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_field_predictions: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    bootstrap_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for record in records:
        prediction_row = predictions[record["example_id"]]
        parsed = prediction_row.get("parsed_json")
        valid_json = isinstance(parsed, dict)
        json_valid += int(valid_json)
        task = record["task_type"]
        source_task = str(record.get("source_task_type", "unknown"))
        task_count[task] += 1
        source_task_count[source_task] += 1
        target_answer = record["target"]["answer"]
        predicted_answer = parsed.get("answer") if valid_json else None
        if task == "visual_state_classification":
            target_values = target_answer
            predicted_values = predicted_answer
            fields = STATE_FIELDS
        else:
            target_values = target_answer["observed_direction_set"]
            predicted_values = (
                predicted_answer.get("observed_direction_set")
                if isinstance(predicted_answer, Mapping)
                else None
            )
            fields = PAIR_FIELDS
        valid_schema = (
            valid_json
            and parsed.get("status") == "answerable"
            and isinstance(predicted_values, Mapping)
            and all(field in predicted_values for field in fields)
        )
        schema_valid += int(valid_schema)
        comparisons: dict[str, bool] = {}
        for field in fields:
            target = str(target_values[field])
            predicted = str(predicted_values.get(field)) if isinstance(predicted_values, Mapping) else "<missing>"
            correct = predicted == target
            comparisons[field] = correct
            key = f"{task}.{field}"
            field_total[key] += 1
            field_correct[key] += int(correct)
            field_targets[key].append(target)
            field_predictions[key].append(predicted)
            source_field_targets[source_task][field].append(target)
            source_field_predictions[source_task][field].append(predicted)
        row_exact = valid_schema and all(comparisons.values())
        exact += int(row_exact)
        task_exact[task] += int(row_exact)
        source_task_exact[source_task] += int(row_exact)
        bootstrap_rows.append(
            {
                "group_id": str(record["group_id"]),
                "exact": bool(row_exact),
                "fields": {
                    field: (
                        str(target_values[field]),
                        str(predicted_values.get(field))
                        if isinstance(predicted_values, Mapping)
                        else "<missing>",
                    )
                    for field in fields
                },
            }
        )
        details.append(
            {
                "example_id": record["example_id"],
                "group_id": record["group_id"],
                "source_task_type": source_task,
                "task_type": task,
                "schema_valid": valid_schema,
                "exact": row_exact,
                "field_correct": comparisons,
                "target": record["target"],
                "prediction": parsed,
            }
        )

    field_accuracy = {
        key: field_correct[key] / field_total[key] for key in sorted(field_total)
    }
    field_macro_f1 = {
        key: macro_f1(field_targets[key], field_predictions[key]) for key in sorted(field_total)
    }
    confusion_matrices: dict[str, dict[str, dict[str, int]]] = {}
    for key in sorted(field_targets):
        matrix: dict[str, Counter[str]] = defaultdict(Counter)
        for target, prediction in zip(field_targets[key], field_predictions[key]):
            matrix[target][prediction] += 1
        confusion_matrices[key] = {
            target: dict(sorted(predictions.items()))
            for target, predictions in sorted(matrix.items())
        }
    by_source_task = {
        source_task: {
            "record_count": source_task_count[source_task],
            "joint_exact_match": source_task_exact[source_task]
            / source_task_count[source_task],
            "equal_field_macro_f1": float(
                np.mean(
                    [
                        macro_f1(
                            source_field_targets[source_task][field],
                            source_field_predictions[source_task][field],
                        )
                        for field in sorted(source_field_targets[source_task])
                    ]
                )
            ),
        }
        for source_task in sorted(source_task_count)
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bootstrap_rows:
        grouped[row["group_id"]].append(row)
    group_ids = sorted(grouped)
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_macro: list[float] = []
    bootstrap_joint: list[float] = []
    for _ in range(max(0, args.bootstrap_samples)):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        rows = [row for group_id in sampled for row in grouped[str(group_id)]]
        bootstrap_joint.append(float(np.mean([row["exact"] for row in rows])))
        per_field = []
        for field in sorted({field for row in rows for field in row["fields"]}):
            values = [row["fields"][field] for row in rows if field in row["fields"]]
            per_field.append(macro_f1([value[0] for value in values], [value[1] for value in values]))
        bootstrap_macro.append(float(np.mean(per_field)))

    def interval(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        low, high = np.percentile(np.asarray(values), [2.5, 97.5])
        return {"low": float(low), "high": float(high)}

    summary = {
        "record_count": len(records),
        "group_count": len(group_ids),
        "json_valid_rate": json_valid / len(records),
        "schema_valid_rate": schema_valid / len(records),
        "joint_exact_match": exact / len(records),
        "task_joint_exact_match": {
            task: task_exact[task] / count for task, count in sorted(task_count.items())
        },
        "field_accuracy": field_accuracy,
        "field_macro_f1": field_macro_f1,
        "equal_field_accuracy": sum(field_accuracy.values()) / len(field_accuracy),
        "equal_field_macro_f1": sum(field_macro_f1.values()) / len(field_macro_f1),
        "confusion_matrices": confusion_matrices,
        "by_source_task": by_source_task,
        "group_bootstrap": {
            "samples": max(0, args.bootstrap_samples),
            "seed": args.bootstrap_seed,
            "equal_field_macro_f1_95ci": interval(bootstrap_macro),
            "joint_exact_match_95ci": interval(bootstrap_joint),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_path = args.output_dir / "details.jsonl"
    with write_path.open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
