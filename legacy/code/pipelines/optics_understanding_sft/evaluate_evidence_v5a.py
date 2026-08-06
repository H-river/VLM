#!/usr/bin/env python3
"""Evaluate v5A evidence aggregation and apply its frozen promotion gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import load_yaml, read_jsonl, write_jsonl
from .evaluate import evaluate_rows, markdown_report, prediction_json


TASKS = ("constrained_intervention", "information_sufficiency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--master-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "confirmation"), required=True)
    return parser.parse_args()


def value_equal(left: Any, right: Any, tolerance: float = 5e-5) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(value_equal(left[key], right[key], tolerance) for key in left)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def witness_valid(record: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    answer = prediction.get("answer")
    if not isinstance(answer, Mapping):
        return False
    witness = answer.get("visible_conflicting_witness")
    if not isinstance(witness, list) or len(witness) != 2:
        return False
    trials = record["prompt_inputs"]["compatible_completion_trials"]
    by_value = {
        round(float(trial["hidden_value_mm"]), 9): str(trial["measured_direction"])
        for trial in trials
    }
    try:
        left, right = (round(float(value), 9) for value in witness)
    except (TypeError, ValueError):
        return False
    return left in by_value and right in by_value and by_value[left] != by_value[right]


def status_recalls(records: list[dict[str, Any]], parsed: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    result = {}
    for status in sorted({str(record["target"]["status"]) for record in records}):
        targets = [record for record in records if record["target"]["status"] == status]
        result[status] = sum(parsed.get(record["example_id"], {}).get("status") == status for record in targets) / len(targets)
    return result


def pair_metrics(records: list[dict[str, Any]], parsed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["provenance"]["match_group_id"])].append(record)
    by_task = {}
    for task in TASKS:
        pairs = [pair for pair in groups.values() if len(pair) == 2 and pair[0]["task_type"] == task]
        both = sum(
            all(parsed.get(record["example_id"], {}).get("status") == record["target"]["status"] for record in pair)
            for pair in pairs
        )
        changed = sum(
            len({parsed.get(record["example_id"], {}).get("status") for record in pair}) == 2
            for pair in pairs
        )
        by_task[task] = {
            "pair_count": len(pairs),
            "both_statuses_correct_rate": both / len(pairs) if pairs else 0.0,
            "prediction_changes_with_pair_rate": changed / len(pairs) if pairs else 0.0,
        }
    return by_task


def evaluate_v5a(
    config: Mapping[str, Any],
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    masters: list[dict[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    details, summary = evaluate_rows(records, predictions, masters, rubric_version="v2")
    prediction_rows = {str(row["example_id"]): row for row in predictions}
    parsed = {}
    for example_id, row in prediction_rows.items():
        value, _ = prediction_json(row)
        parsed[example_id] = value or {}
    recalls = status_recalls(records, parsed)
    pairs = pair_metrics(records, parsed)
    feasible = [record for record in records if record["target"]["status"] == "feasible"]
    answerable = [record for record in records if record["target"]["status"] == "answerable"]
    insufficient = [record for record in records if record["target"]["status"] == "insufficient_information"]
    detail_by_id = {row["example_id"]: row for row in details}
    action_exact = sum(
        value_equal(
            parsed.get(record["example_id"], {}).get("answer", {}).get("control_plan"),
            record["target"]["answer"]["control_plan"],
        )
        for record in feasible
    ) / len(feasible)
    simulator_success = sum(float(detail_by_id[record["example_id"]].get("simulator_outcome_success", 0.0)) for record in feasible) / len(feasible)
    minimum_motion = sum(float(detail_by_id[record["example_id"]].get("minimum_motion_optimal", 0.0)) for record in feasible) / len(feasible)
    direction_accuracy = sum(
        parsed.get(record["example_id"], {}).get("answer", {}).get("centroid_x_direction")
        == record["target"]["answer"]["centroid_x_direction"]
        for record in answerable
    ) / len(answerable)
    witness_accuracy = sum(witness_valid(record, parsed.get(record["example_id"], {})) for record in insufficient) / len(insufficient)
    metrics = {
        "split": split,
        "record_count": len(records),
        "schema_valid_rate": summary["schema_valid_rate"],
        "per_task_status_macro_f1": {
            task: summary["per_task"][task]["status_macro_f1"] for task in TASKS
        },
        "per_status_recall": recalls,
        "pair_metrics": pairs,
        "control_action_exact_match": action_exact,
        "control_simulator_success": simulator_success,
        "control_minimum_motion_correct": minimum_motion,
        "sufficiency_answerable_direction_accuracy": direction_accuracy,
        "sufficiency_witness_validity": witness_accuracy,
    }
    gates = config["promotion_gates"]
    checks = {
        "schema_valid_rate": metrics["schema_valid_rate"] >= float(gates["schema_valid_rate"]),
        "per_task_status_macro_f1": all(
            value >= float(gates["per_task_status_macro_f1"])
            for value in metrics["per_task_status_macro_f1"].values()
        ),
        "per_status_recall": all(
            value >= float(gates["per_status_recall"]) for value in recalls.values()
        ),
        "pair_joint_status_accuracy": all(
            item["both_statuses_correct_rate"] >= float(gates["pair_joint_status_accuracy"])
            for item in pairs.values()
        ),
        "control_action_exact_match": action_exact >= float(gates["control_action_exact_match"]),
        "control_simulator_success": simulator_success >= float(gates["control_simulator_success"]),
        "control_minimum_motion_correct": minimum_motion >= float(gates["control_minimum_motion_correct"]),
        "sufficiency_answerable_direction_accuracy": direction_accuracy >= float(gates["sufficiency_answerable_direction_accuracy"]),
        "sufficiency_witness_validity": witness_accuracy >= float(gates["sufficiency_witness_validity"]),
    }
    gate_report = {
        "split": split,
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": dict(gates),
        "metrics": metrics,
        "confirmation_may_open": split == "dev" and all(checks.values()),
    }
    return details, summary, gate_report


def main() -> None:
    args = parse_args()
    details, summary, gates = evaluate_v5a(
        load_yaml(args.config),
        read_jsonl(args.records_jsonl),
        read_jsonl(args.predictions_jsonl),
        read_jsonl(args.master_jsonl),
        args.split,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "details.jsonl", details)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({"summary": summary["macro_task_score"], "gates": gates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
