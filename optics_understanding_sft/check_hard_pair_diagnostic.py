#!/usr/bin/env python3
"""Apply the frozen hard-pair holdout gates before unchanged-dev inference."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def recall(details: list[Mapping[str, Any]], label: str) -> float:
    rows = [
        row
        for row in details
        if row["task_type"] == "constrained_intervention" and row["target_status"] == label
    ]
    if not rows:
        raise ValueError(f"missing control label in diagnostic: {label}")
    return sum(row.get("predicted_status") == label for row in rows) / len(rows)


def check(result_dir: Path, records_path: Path) -> dict[str, Any]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    details = read_jsonl(result_dir / "details.jsonl")
    records = read_jsonl(records_path)
    if len(details) != len(records):
        raise ValueError(f"diagnostic must be complete: {len(details)} of {len(records)}")
    record_index = {row["example_id"]: row for row in records}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        record = record_index[row["example_id"]]
        groups[str(record["provenance"]["match_group_id"])].append(row)
    invalid_groups = [group_id for group_id, rows in groups.items() if len(rows) != 2]
    if invalid_groups:
        raise ValueError(f"invalid minimal-pair groups: {invalid_groups[:3]}")
    joint = statistics.fmean(
        all(row.get("predicted_status") == row["target_status"] for row in rows)
        for rows in groups.values()
    )
    feasible_rows = [
        row
        for row in details
        if row["task_type"] == "constrained_intervention" and row["target_status"] == "feasible"
    ]
    metrics = {
        "schema_valid_rate": float(summary["schema_valid_rate"]),
        "sufficiency_status_macro_f1": float(
            summary["per_task"]["information_sufficiency"]["status_macro_f1"]
        ),
        "control_status_macro_f1": float(
            summary["per_task"]["constrained_intervention"]["status_macro_f1"]
        ),
        "control_feasible_recall": recall(details, "feasible"),
        "control_infeasible_recall": recall(details, "infeasible_within_limits"),
        "control_feasible_simulator_success": statistics.fmean(
            float(row.get("simulator_outcome_success", 0.0)) for row in feasible_rows
        ),
        "minimal_pair_joint_status_accuracy": joint,
    }
    gates = {
        "schema_valid_rate_at_least_0_93": metrics["schema_valid_rate"] >= 0.93,
        "sufficiency_status_macro_f1_at_least_0_60": metrics[
            "sufficiency_status_macro_f1"
        ]
        >= 0.60,
        "control_status_macro_f1_at_least_0_60": metrics["control_status_macro_f1"]
        >= 0.60,
        "control_feasible_recall_at_least_0_60": metrics["control_feasible_recall"] >= 0.60,
        "control_infeasible_recall_at_least_0_60": metrics["control_infeasible_recall"]
        >= 0.60,
        "control_simulator_success_at_least_0_40": metrics[
            "control_feasible_simulator_success"
        ]
        >= 0.40,
        "minimal_pair_joint_accuracy_at_least_0_40": metrics[
            "minimal_pair_joint_status_accuracy"
        ]
        >= 0.40,
    }
    return {
        "protocol": "hard_pairs_v4",
        "evaluated_records": len(details),
        "minimal_pair_count": len(groups),
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "sealed_test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    result = check(args.result_dir, args.records_jsonl)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
