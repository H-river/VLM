#!/usr/bin/env python3
"""Evaluate v7.1 routing, mapped execution, and compact decisions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import load_yaml, read_jsonl, write_jsonl
from .evaluate_intermediate_evidence_v6 import macro_f1, parsed_prediction, value_equal
from .tool_path_adapter import run_mapped_tool


INTERPRETATION_STAGE = "compact_tool_result_interpretation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--source-task-type")
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


def schema_valid(record: Mapping[str, Any], prediction: Mapping[str, Any] | None) -> bool:
    if prediction is None:
        return False
    stage = str(record["stage"])
    if stage == "tool_choice":
        return set(prediction) == {"tool_name"} and isinstance(prediction["tool_name"], str)
    if stage == "tool_source_mapping":
        return (
            set(prediction) == {"tool_name", "source_map"}
            and isinstance(prediction["tool_name"], str)
            and isinstance(prediction["source_map"], Mapping)
        )
    expected = (
        {"status", "successful_action_indices", "selected_index", "best_residual_index"}
        if record["source_task_type"] == "constrained_intervention"
        else {"status", "observed_direction_set", "conflicting_pair_indices"}
    )
    return set(prediction) == expected and isinstance(prediction.get("status"), str)


def evaluate(
    config: Mapping[str, Any],
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    parsed = {str(row["example_id"]): parsed_prediction(row) for row in predictions}
    details: list[dict[str, Any]] = []
    for record in records:
        prediction = parsed.get(str(record["example_id"]))
        target = record["target"]
        stage = str(record["stage"])
        detail: dict[str, Any] = {
            "example_id": record["example_id"],
            "source_example_id": record["source_example_id"],
            "source_task_type": record["source_task_type"],
            "stage": stage,
            "json_valid": prediction is not None,
            "schema_valid": schema_valid(record, prediction),
            "target_exact": value_equal(prediction, target) if prediction is not None else False,
        }
        if stage == "tool_choice":
            detail["tool_name_correct"] = bool(prediction) and prediction.get("tool_name") == target["tool_name"]
        elif stage == "tool_source_mapping":
            detail["source_mapping_exact"] = bool(prediction) and value_equal(prediction.get("source_map"), target["source_map"])
            detail["mapped_tool_execution_match"] = False
            if prediction and isinstance(prediction.get("source_map"), Mapping):
                try:
                    observed = run_mapped_tool(
                        str(prediction.get("tool_name")),
                        record["prompt_inputs"]["visible_evidence"],
                        prediction["source_map"],
                    )
                    expected = run_mapped_tool(
                        target["tool_name"],
                        record["prompt_inputs"]["visible_evidence"],
                        target["source_map"],
                    )
                    detail["mapped_tool_execution_match"] = value_equal(observed, expected)
                except (KeyError, TypeError, ValueError):
                    pass
        else:
            detail["status_correct"] = bool(prediction) and prediction.get("status") == target["status"]
            detail["compact_evidence_exact"] = bool(prediction) and value_equal(
                {key: value for key, value in prediction.items() if key != "status"},
                {key: value for key, value in target.items() if key != "status"},
            )
            detail["predicted_status"] = prediction.get("status") if prediction else None
            detail["target_status"] = target["status"]
        details.append(detail)

    def stage_rate(stage: str, key: str) -> float:
        rows = [row for row in details if row["stage"] == stage]
        return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0

    interpretations = [row for row in details if row["stage"] == INTERPRETATION_STAGE]
    status_f1 = {
        task: macro_f1(
            [str(row["target_status"]) for row in interpretations if row["source_task_type"] == task],
            [row["predicted_status"] for row in interpretations if row["source_task_type"] == task],
        )
        for task in sorted({row["source_task_type"] for row in interpretations})
    }
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_source[str(row["source_example_id"])].append(row)
    end_to_end = (
        sum(len(rows) == 3 and all(row["target_exact"] for row in rows) for rows in by_source.values())
        / len(by_source)
        if by_source
        else 0.0
    )
    source_match = {
        str(record["source_example_id"]): str(record["provenance"]["source_match_group_id"])
        for record in records
    }
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interpretations:
        pairs[source_match[str(row["source_example_id"])]].append(row)
    pair_joint = (
        sum(len(rows) == 2 and all(row["status_correct"] for row in rows) for rows in pairs.values())
        / len(pairs)
        if pairs
        else 0.0
    )
    summary = {
        "record_count": len(records),
        "prediction_count": len(predictions),
        "json_valid_rate": sum(row["json_valid"] for row in details) / len(details),
        "schema_valid_rate": sum(row["schema_valid"] for row in details) / len(details),
        "tool_choice_accuracy": stage_rate("tool_choice", "tool_name_correct"),
        "source_mapping_exact_match": stage_rate("tool_source_mapping", "source_mapping_exact"),
        "mapped_tool_execution_match": stage_rate("tool_source_mapping", "mapped_tool_execution_match"),
        "compact_decision_exact_match": stage_rate(INTERPRETATION_STAGE, "target_exact"),
        "compact_evidence_exact_match": stage_rate(INTERPRETATION_STAGE, "compact_evidence_exact"),
        "interpretation_status_macro_f1": status_f1,
        "pair_joint_status_accuracy": pair_joint,
        "end_to_end_exact_match": end_to_end,
    }
    gates = config["promotion_gates"]
    checks = {
        "schema_valid_rate": summary["schema_valid_rate"] >= float(gates["schema_valid_rate"]),
        "tool_choice_accuracy": summary["tool_choice_accuracy"] >= float(gates["tool_choice_accuracy"]),
        "source_mapping_exact_match": summary["source_mapping_exact_match"] >= float(gates["source_mapping_exact_match"]),
        "mapped_tool_execution_match": summary["mapped_tool_execution_match"] >= float(gates["mapped_tool_execution_match"]),
        "compact_decision_exact_match": summary["compact_decision_exact_match"] >= float(gates["compact_decision_exact_match"]),
        "interpretation_status_macro_f1": bool(status_f1)
        and all(value >= float(gates["interpretation_status_macro_f1"]) for value in status_f1.values()),
        "pair_joint_status_accuracy": pair_joint >= float(gates["pair_joint_status_accuracy"]),
        "end_to_end_exact_match": end_to_end >= float(gates["end_to_end_exact_match"]),
    }
    return details, summary, {"passed": all(checks.values()), "checks": checks, "thresholds": dict(gates), "metrics": summary}


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records_jsonl)
    if args.stage:
        records = [row for row in records if row.get("stage") == args.stage]
    if args.source_task_type:
        records = [row for row in records if row.get("source_task_type") == args.source_task_type]
    if args.max_records is not None:
        records = records[: max(0, args.max_records)]
    details, summary, gates = evaluate(load_yaml(args.config), records, read_jsonl(args.predictions_jsonl))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "details.jsonl", details)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
