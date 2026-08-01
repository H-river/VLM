#!/usr/bin/env python3
"""Evaluate visual tool selection, ordered source mapping, and interpretation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import read_jsonl, write_jsonl
from .evaluate_intermediate_evidence_v6 import parsed_prediction, value_equal
from .visual_state_tool_v10_1 import load_calibration
from .visual_tool_adapter_v10_2 import run_mapped_visual_tool


def schema_valid(stage: str, prediction: Mapping[str, Any] | None) -> bool:
    if prediction is None:
        return False
    if stage == "tool_choice":
        return set(prediction) == {"tool_name"} and isinstance(prediction["tool_name"], str)
    if stage == "tool_source_mapping":
        return (
            set(prediction) == {"tool_name", "source_map"}
            and isinstance(prediction["tool_name"], str)
            and isinstance(prediction["source_map"], Mapping)
        )
    return (
        set(prediction) == {"status", "answer"}
        and prediction.get("status") == "answerable"
        and isinstance(prediction.get("answer"), Mapping)
    )


def evaluate(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    image_root: Path,
    state_calibration: Mapping[str, Any],
    pair_calibration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = {str(row["example_id"]): parsed_prediction(row) for row in predictions}
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
            "source_task_type": record["source_task_type"],
            "stage": stage,
            "json_valid": prediction is not None,
            "schema_valid": schema_valid(stage, prediction),
            "target_exact": value_equal(prediction, target) if prediction is not None else False,
        }
        if stage == "tool_choice":
            detail["tool_choice_correct"] = bool(prediction) and prediction.get("tool_name") == target["tool_name"]
        elif stage == "tool_source_mapping":
            detail["source_mapping_exact"] = bool(prediction) and value_equal(prediction.get("source_map"), target["source_map"])
            detail["tool_execution_exact"] = False
            if prediction and isinstance(prediction.get("source_map"), Mapping):
                try:
                    result = run_mapped_visual_tool(
                        str(prediction.get("tool_name")),
                        record["prompt_inputs"]["visible_evidence"],
                        prediction["source_map"],
                        image_root=image_root,
                        state_calibration=state_calibration,
                        pair_calibration=pair_calibration,
                    )
                    detail["tool_execution_exact"] = value_equal(
                        result, cached_results[str(record["source_example_id"])]
                    )
                except (KeyError, TypeError, ValueError, OSError):
                    pass
        else:
            detail["result_interpretation_exact"] = detail["target_exact"]
        details.append(detail)

    def rate(stage: str, key: str, source_task: str | None = None) -> float:
        rows = [
            row
            for row in details
            if row["stage"] == stage
            and (source_task is None or row["source_task_type"] == source_task)
        ]
        return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_source[str(row["source_example_id"])].append(row)
    end_to_end = sum(
        len(rows) == 3
        and all(row["target_exact"] for row in rows)
        and next(row for row in rows if row["stage"] == "tool_source_mapping")["tool_execution_exact"]
        for rows in by_source.values()
    ) / len(by_source)
    production_end_to_end = sum(
        len(rows) == 3
        and next(row for row in rows if row["stage"] == "tool_choice")[
            "tool_choice_correct"
        ]
        and next(row for row in rows if row["stage"] == "tool_source_mapping")[
            "tool_execution_exact"
        ]
        and next(
            row
            for row in rows
            if row["stage"] == "compact_tool_result_interpretation"
        )["result_interpretation_exact"]
        for rows in by_source.values()
    ) / len(by_source)
    summary = {
        "record_count": len(records),
        "source_count": len(by_source),
        "json_valid_rate": sum(row["json_valid"] for row in details) / len(details),
        "schema_valid_rate": sum(row["schema_valid"] for row in details) / len(details),
        "tool_choice_accuracy": rate("tool_choice", "tool_choice_correct"),
        "source_mapping_exact_match": rate("tool_source_mapping", "source_mapping_exact"),
        "validated_source_mapping_exact_match": rate(
            "tool_source_mapping", "tool_execution_exact"
        ),
        "tool_execution_exact_match": rate("tool_source_mapping", "tool_execution_exact"),
        "result_interpretation_exact_match": rate(
            "compact_tool_result_interpretation", "result_interpretation_exact"
        ),
        "end_to_end_exact_match": end_to_end,
        "validated_production_end_to_end_exact_match": production_end_to_end,
        "by_source_task": {
            source_task: {
                "tool_choice_accuracy": rate("tool_choice", "tool_choice_correct", source_task),
                "source_mapping_exact_match": rate("tool_source_mapping", "source_mapping_exact", source_task),
                "tool_execution_exact_match": rate("tool_source_mapping", "tool_execution_exact", source_task),
                "result_interpretation_exact_match": rate(
                    "compact_tool_result_interpretation", "result_interpretation_exact", source_task
                ),
            }
            for source_task in ("visual_state_classification", "visual_pair_direction_extraction")
        },
    }
    thresholds = {
        "json_valid_rate": 0.98,
        "schema_valid_rate": 0.98,
        "tool_choice_accuracy": 0.95,
        "source_mapping_exact_match": 0.95,
        "tool_execution_exact_match": 0.95,
        "result_interpretation_exact_match": 0.95,
        "end_to_end_exact_match": 0.90,
    }
    checks = {key: summary[key] >= value for key, value in thresholds.items()}
    return details, {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds, "metrics": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--state-calibration", type=Path, required=True)
    parser.add_argument("--pair-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    records = read_jsonl(args.records_jsonl)
    if args.max_samples is not None:
        records = records[: max(0, args.max_samples)]
    details, gates = evaluate(
        records,
        read_jsonl(args.predictions_jsonl),
        image_root=args.image_root,
        state_calibration=load_calibration(args.state_calibration),
        pair_calibration=load_calibration(args.pair_calibration),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "details.jsonl", details)
    (args.output_dir / "summary.json").write_text(
        json.dumps(gates["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gates, indent=2, sort_keys=True))
    if not gates["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
