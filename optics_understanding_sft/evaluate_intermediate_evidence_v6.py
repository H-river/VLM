#!/usr/bin/env python3
"""Evaluate v6 tool choice, call construction, and result interpretation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import load_yaml, read_jsonl, write_jsonl
from .decision_tools import run_tool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", help="Filter to one derived supervision stage")
    parser.add_argument("--source-task-type", help="Filter to one original task family")
    parser.add_argument("--max-records", type=int, help="Evaluate a deterministic prefix for smoke runs")
    return parser.parse_args()


def value_equal(left: Any, right: Any, tolerance: float = 5e-5) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(value_equal(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(value_equal(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def parsed_prediction(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    parsed = row.get("parsed_json")
    if isinstance(parsed, Mapping):
        return parsed
    raw = row.get("raw_prediction_text")
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def schema_valid(stage: str, prediction: Mapping[str, Any] | None) -> bool:
    if prediction is None:
        return False
    if stage == "tool_choice":
        return set(prediction) == {"tool_name"} and isinstance(prediction["tool_name"], str)
    if stage == "tool_call_construction":
        return (
            set(prediction) == {"tool_name", "arguments"}
            and isinstance(prediction["tool_name"], str)
            and isinstance(prediction["arguments"], Mapping)
        )
    return (
        set(prediction) == {"intermediate_evidence", "status", "answer"}
        and isinstance(prediction["intermediate_evidence"], Mapping)
        and isinstance(prediction["status"], str)
        and isinstance(prediction["answer"], Mapping)
    )


def macro_f1(targets: list[str], predictions: list[str | None]) -> float:
    # Include predicted-only labels so their false positives cannot disappear
    # when a small evaluation panel happens to omit that target class.
    labels = sorted(set(targets) | {value for value in predictions if value is not None})
    scores = []
    for label in labels:
        tp = sum(target == label and prediction == label for target, prediction in zip(targets, predictions))
        fp = sum(target != label and prediction == label for target, prediction in zip(targets, predictions))
        fn = sum(target == label and prediction != label for target, prediction in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(
    config: Mapping[str, Any],
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prediction_by_id = {str(row["example_id"]): parsed_prediction(row) for row in predictions}
    details: list[dict[str, Any]] = []
    for record in records:
        example_id = str(record["example_id"])
        prediction = prediction_by_id.get(example_id)
        target = record["target"]
        stage = str(record["stage"])
        detail: dict[str, Any] = {
            "example_id": example_id,
            "source_example_id": record["source_example_id"],
            "source_task_type": record["source_task_type"],
            "stage": stage,
            "json_valid": prediction is not None,
            "schema_valid": schema_valid(stage, prediction),
            "target_exact": value_equal(prediction, target) if prediction is not None else False,
        }
        if stage == "tool_choice":
            detail["tool_name_correct"] = bool(prediction) and prediction.get("tool_name") == target["tool_name"]
        elif stage == "tool_call_construction":
            detail["tool_name_correct"] = bool(prediction) and prediction.get("tool_name") == target["tool_name"]
            detail["arguments_exact"] = bool(prediction) and value_equal(prediction.get("arguments"), target["arguments"])
            detail["tool_result_exact"] = False
            if prediction and isinstance(prediction.get("arguments"), Mapping):
                try:
                    observed_result = run_tool(str(prediction.get("tool_name")), prediction["arguments"])
                    expected_result = run_tool(target["tool_name"], target["arguments"])
                    detail["tool_result_exact"] = value_equal(observed_result, expected_result)
                except (TypeError, ValueError):
                    pass
        else:
            detail["intermediate_evidence_exact"] = bool(prediction) and value_equal(
                prediction.get("intermediate_evidence"), target["intermediate_evidence"]
            )
            detail["status_correct"] = bool(prediction) and prediction.get("status") == target["status"]
            detail["answer_exact"] = bool(prediction) and value_equal(prediction.get("answer"), target["answer"])
            detail["predicted_status"] = prediction.get("status") if prediction else None
            detail["target_status"] = target["status"]
        details.append(detail)

    def rate(stage: str, key: str) -> float:
        rows = [row for row in details if row["stage"] == stage]
        return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0

    interpretation = [row for row in details if row["stage"] == "tool_result_interpretation"]
    status_f1 = {}
    for task in sorted({row["source_task_type"] for row in interpretation}):
        rows = [row for row in interpretation if row["source_task_type"] == task]
        status_f1[task] = macro_f1(
            [str(row["target_status"]) for row in rows],
            [row["predicted_status"] for row in rows],
        )

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_source[str(row["source_example_id"])].append(row)
    end_to_end = sum(
        len(rows) == 3 and all(row["target_exact"] for row in rows)
        for rows in by_source.values()
    ) / len(by_source)

    source_match = {
        str(record["source_example_id"]): str(record["provenance"]["source_match_group_id"])
        for record in records
    }
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interpretation:
        pairs[source_match[str(row["source_example_id"])]].append(row)
    pair_joint = sum(
        len(rows) == 2 and all(row["status_correct"] for row in rows)
        for rows in pairs.values()
    ) / len(pairs) if pairs else 0.0

    summary = {
        "record_count": len(records),
        "prediction_count": len(predictions),
        "json_valid_rate": sum(row["json_valid"] for row in details) / len(details),
        "schema_valid_rate": sum(row["schema_valid"] for row in details) / len(details),
        "tool_choice_accuracy": rate("tool_choice", "tool_name_correct"),
        "tool_call_name_accuracy": rate("tool_call_construction", "tool_name_correct"),
        "tool_arguments_exact_match": rate("tool_call_construction", "arguments_exact"),
        "tool_execution_result_match": rate("tool_call_construction", "tool_result_exact"),
        "intermediate_evidence_exact_match": rate("tool_result_interpretation", "intermediate_evidence_exact"),
        "interpretation_status_macro_f1": status_f1,
        "interpretation_answer_exact_match": rate("tool_result_interpretation", "answer_exact"),
        "pair_joint_status_accuracy": pair_joint,
        "end_to_end_exact_match": end_to_end,
    }
    gates = config["promotion_gates"]
    checks = {
        "schema_valid_rate": summary["schema_valid_rate"] >= float(gates["schema_valid_rate"]),
        "tool_choice_accuracy": summary["tool_choice_accuracy"] >= float(gates["tool_choice_accuracy"]),
        "tool_arguments_exact_match": summary["tool_arguments_exact_match"] >= float(gates["tool_arguments_exact_match"]),
        "tool_execution_result_match": summary["tool_execution_result_match"] >= float(gates["tool_execution_result_match"]),
        "intermediate_evidence_exact_match": summary["intermediate_evidence_exact_match"] >= float(gates["intermediate_evidence_exact_match"]),
        "interpretation_status_macro_f1": all(value >= float(gates["interpretation_status_macro_f1"]) for value in status_f1.values()),
        "interpretation_answer_exact_match": summary["interpretation_answer_exact_match"] >= float(gates["interpretation_answer_exact_match"]),
        "pair_joint_status_accuracy": summary["pair_joint_status_accuracy"] >= float(gates["pair_joint_status_accuracy"]),
        "end_to_end_exact_match": summary["end_to_end_exact_match"] >= float(gates["end_to_end_exact_match"]),
    }
    gate_report = {"passed": all(checks.values()), "checks": checks, "thresholds": dict(gates), "metrics": summary}
    return details, summary, gate_report


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records_jsonl)
    if args.stage:
        records = [record for record in records if record.get("stage") == args.stage]
    if args.source_task_type:
        records = [
            record
            for record in records
            if record.get("source_task_type") == args.source_task_type
        ]
    if args.max_records is not None:
        records = records[: max(0, args.max_records)]
    details, summary, gates = evaluate(
        load_yaml(args.config), records, read_jsonl(args.predictions_jsonl)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "details.jsonl", details)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
