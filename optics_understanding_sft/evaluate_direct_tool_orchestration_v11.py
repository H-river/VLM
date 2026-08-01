#!/usr/bin/env python3
"""Evaluate setup, causal, and diagnosis registered-tool orchestration."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import read_jsonl, write_jsonl
from .direct_reasoning_tools_v11 import run_mapped_direct_tool
from .evaluate_intermediate_evidence_v6 import parsed_prediction, value_equal


def load_setup_registry(path: Path) -> dict[str, Mapping[str, Any]]:
    return {str(row["state_handle"]): row["setup_config"] for row in read_jsonl(path)}


def load_observation_registry(path: Path) -> dict[str, Mapping[str, Any]]:
    return {str(row["observation_handle"]): row["state"] for row in read_jsonl(path)}


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
        and isinstance(prediction.get("status"), str)
        and isinstance(prediction.get("answer"), Mapping)
    )


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, float]:
    """Return a two-sided Wilson score interval for a Bernoulli success rate."""
    if total <= 0:
        return {"low": 0.0, "high": 1.0}
    rate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return {"low": max(0.0, center - radius), "high": min(1.0, center + radius)}


def evaluate(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    setup_registry: Mapping[str, Mapping[str, Any]],
    observation_registry: Mapping[str, Mapping[str, Any]],
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
            detail["tool_choice_correct"] = bool(prediction) and prediction.get(
                "tool_name"
            ) == target["tool_name"]
        elif stage == "tool_source_mapping":
            detail["source_mapping_exact"] = bool(prediction) and value_equal(
                prediction.get("source_map"), target["source_map"]
            )
            detail["tool_execution_exact"] = False
            if prediction and isinstance(prediction.get("source_map"), Mapping):
                try:
                    result = run_mapped_direct_tool(
                        str(prediction.get("tool_name")),
                        record["prompt_inputs"]["visible_evidence"],
                        prediction["source_map"],
                        setup_registry=setup_registry,
                        observation_registry=observation_registry,
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
    end_to_end_by_source = {
        source_id: bool(
            len(rows) == 3
            and all(row["target_exact"] for row in rows)
            and next(row for row in rows if row["stage"] == "tool_source_mapping")[
                "tool_execution_exact"
            ]
        )
        for source_id, rows in by_source.items()
    }
    end_to_end_successes = sum(end_to_end_by_source.values())
    end_to_end = end_to_end_successes / len(by_source)
    group_by_source = {
        str(row["source_example_id"]): str(row["group_id"]) for row in records
    }
    sources_by_group: dict[str, list[str]] = defaultdict(list)
    for source_id, group_id in group_by_source.items():
        sources_by_group[group_id].append(source_id)
    end_to_end_by_group = {
        group_id: all(end_to_end_by_source[source_id] for source_id in source_ids)
        for group_id, source_ids in sources_by_group.items()
    }
    group_successes = sum(end_to_end_by_group.values())
    summary = {
        "record_count": len(records),
        "source_count": len(by_source),
        "json_valid_rate": sum(row["json_valid"] for row in details) / len(details),
        "schema_valid_rate": sum(row["schema_valid"] for row in details) / len(details),
        "tool_choice_accuracy": rate("tool_choice", "tool_choice_correct"),
        "source_mapping_exact_match": rate("tool_source_mapping", "source_mapping_exact"),
        "tool_execution_exact_match": rate("tool_source_mapping", "tool_execution_exact"),
        "result_interpretation_exact_match": rate(
            "compact_tool_result_interpretation", "result_interpretation_exact"
        ),
        "end_to_end_exact_match": end_to_end,
        "end_to_end_successes": end_to_end_successes,
        "end_to_end_wilson_95ci": wilson_interval(end_to_end_successes, len(by_source)),
        "physical_group_count": len(sources_by_group),
        "physical_group_successes": group_successes,
        "physical_group_end_to_end_exact_match": group_successes / len(sources_by_group),
        "physical_group_end_to_end_wilson_95ci": wilson_interval(
            group_successes, len(sources_by_group)
        ),
        "by_source_task": {
            task: {
                "tool_choice_accuracy": rate("tool_choice", "tool_choice_correct", task),
                "source_mapping_exact_match": rate(
                    "tool_source_mapping", "source_mapping_exact", task
                ),
                "tool_execution_exact_match": rate(
                    "tool_source_mapping", "tool_execution_exact", task
                ),
                "result_interpretation_exact_match": rate(
                    "compact_tool_result_interpretation",
                    "result_interpretation_exact",
                    task,
                ),
                "source_count": sum(
                    rows[0]["source_task_type"] == task for rows in by_source.values()
                ),
                "end_to_end_successes": sum(
                    end_to_end_by_source[source_id]
                    for source_id, rows in by_source.items()
                    if rows[0]["source_task_type"] == task
                ),
                "end_to_end_exact_match": sum(
                    end_to_end_by_source[source_id]
                    for source_id, rows in by_source.items()
                    if rows[0]["source_task_type"] == task
                )
                / sum(rows[0]["source_task_type"] == task for rows in by_source.values()),
                "end_to_end_wilson_95ci": wilson_interval(
                    sum(
                        end_to_end_by_source[source_id]
                        for source_id, rows in by_source.items()
                        if rows[0]["source_task_type"] == task
                    ),
                    sum(
                        rows[0]["source_task_type"] == task
                        for rows in by_source.values()
                    ),
                ),
            }
            for task in ("setup_interpretation", "causal_effects", "diagnosis")
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
    return details, {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "metrics": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--setup-registry", type=Path, required=True)
    parser.add_argument("--observation-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    details, gates = evaluate(
        read_jsonl(args.records_jsonl),
        read_jsonl(args.predictions_jsonl),
        setup_registry=load_setup_registry(args.setup_registry),
        observation_registry=load_observation_registry(args.observation_registry),
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
