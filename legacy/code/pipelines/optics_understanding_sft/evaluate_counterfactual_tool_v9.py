#!/usr/bin/env python3
"""Evaluate paired counterfactual routing, mapping, replay, and interpretation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import read_jsonl, write_jsonl
from .counterfactual_tool import materialize_counterfactual_answer, run_mapped_counterfactual_tool
from .evaluate_intermediate_evidence_v6 import parsed_prediction, value_equal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--state-registry-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


def schema_valid(stage: str, prediction: Mapping[str, Any] | None) -> bool:
    if prediction is None:
        return False
    if stage == "tool_choice":
        return set(prediction) == {"tool_name"} and isinstance(prediction["tool_name"], str)
    if stage == "tool_source_mapping":
        return set(prediction) == {"tool_name", "source_map"} and isinstance(prediction["source_map"], Mapping)
    return (
        set(prediction) == {"status", "result_ready", "result_fields"}
        and isinstance(prediction["status"], str)
        and isinstance(prediction["result_ready"], bool)
        and isinstance(prediction["result_fields"], list)
    )


def evaluate(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = {str(row["example_id"]): parsed_prediction(row) for row in predictions}
    registry = {str(row["state_handle"]): row["setup_config"] for row in registry_rows}
    cached_results = {
        str(row["source_example_id"]): row["prompt_inputs"]["tool_result"]
        for row in records
        if row["stage"] == "compact_tool_result_interpretation"
    }
    details: list[dict[str, Any]] = []
    for record in records:
        prediction = parsed.get(str(record["example_id"]))
        stage = str(record["stage"])
        target = record["target"]
        detail: dict[str, Any] = {
            "example_id": record["example_id"],
            "source_example_id": record["source_example_id"],
            "source_modality": record["source_modality"],
            "stage": stage,
            "json_valid": prediction is not None,
            "schema_valid": schema_valid(stage, prediction),
            "target_exact": value_equal(prediction, target) if prediction is not None else False,
        }
        if stage == "tool_choice":
            detail["tool_choice_correct"] = bool(prediction) and prediction.get("tool_name") == target["tool_name"]
        elif stage == "tool_source_mapping":
            detail["source_mapping_exact"] = bool(prediction) and value_equal(prediction.get("source_map"), target["source_map"])
            detail["tool_result_exact"] = False
            detail["materialized_answer_exact"] = False
            if prediction and isinstance(prediction.get("source_map"), Mapping):
                try:
                    result = run_mapped_counterfactual_tool(
                        record["prompt_inputs"]["visible_evidence"], prediction["source_map"], registry
                    )
                    expected = cached_results[str(record["source_example_id"])]
                    detail["tool_result_exact"] = value_equal(result, expected)
                    detail["materialized_answer_exact"] = value_equal(
                        materialize_counterfactual_answer(result),
                        {"status": "answerable", "answer": expected},
                    )
                except (KeyError, TypeError, ValueError):
                    pass
        else:
            detail["result_interpretation_exact"] = detail["target_exact"]
        details.append(detail)

    def rate(stage: str, key: str, modality: str | None = None) -> float:
        rows = [
            row for row in details
            if row["stage"] == stage and (modality is None or row["source_modality"] == modality)
        ]
        return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_source[str(row["source_example_id"])].append(row)
    end_to_end_by_modality: dict[str, float] = {}
    for modality in ("text", "visual"):
        groups = [rows for rows in by_source.values() if rows[0]["source_modality"] == modality]
        end_to_end_by_modality[modality] = (
            sum(
                len(rows) == 3
                and all(row["target_exact"] for row in rows)
                and next(row for row in rows if row["stage"] == "tool_source_mapping")["tool_result_exact"]
                for rows in groups
            ) / len(groups)
            if groups else 0.0
        )
    end_to_end = sum(
        len(rows) == 3
        and all(row["target_exact"] for row in rows)
        and next(row for row in rows if row["stage"] == "tool_source_mapping")["tool_result_exact"]
        for rows in by_source.values()
    ) / len(by_source) if by_source else 0.0
    summary = {
        "record_count": len(records),
        "prediction_count": len(predictions),
        "json_valid_rate": sum(row["json_valid"] for row in details) / len(details),
        "schema_valid_rate": sum(row["schema_valid"] for row in details) / len(details),
        "tool_choice_accuracy": rate("tool_choice", "tool_choice_correct"),
        "source_mapping_exact_match": rate("tool_source_mapping", "source_mapping_exact"),
        "simulator_tool_result_exact_match": rate("tool_source_mapping", "tool_result_exact"),
        "materialized_quantitative_answer_exact_match": rate("tool_source_mapping", "materialized_answer_exact"),
        "result_interpretation_exact_match": rate("compact_tool_result_interpretation", "result_interpretation_exact"),
        "end_to_end_exact_match": end_to_end,
        "end_to_end_by_source_modality": end_to_end_by_modality,
    }
    thresholds = {
        "json_valid_rate": 0.98,
        "schema_valid_rate": 0.98,
        "tool_choice_accuracy": 0.95,
        "source_mapping_exact_match": 0.95,
        "simulator_tool_result_exact_match": 0.98,
        "materialized_quantitative_answer_exact_match": 0.98,
        "result_interpretation_exact_match": 0.95,
        "end_to_end_exact_match": 0.90,
    }
    checks = {key: summary[key] >= threshold for key, threshold in thresholds.items()}
    return details, {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds, "metrics": summary}


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records_jsonl)
    if args.max_records is not None:
        records = records[: max(0, args.max_records)]
    details, gates = evaluate(records, read_jsonl(args.predictions_jsonl), read_jsonl(args.state_registry_jsonl))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "details.jsonl", details)
    (args.output_dir / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(gates["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
