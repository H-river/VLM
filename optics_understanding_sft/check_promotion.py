#!/usr/bin/env python3
"""Apply the frozen corrective-v2 promotion gates to evaluator outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from .core import read_jsonl


ANCHOR_TASKS = (
    "setup_interpretation",
    "causal_effects",
    "forward_prediction",
    "counterfactual_reasoning",
)
STATUS_TASKS = (
    "information_sufficiency",
    "diagnosis",
    "constrained_intervention",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=RESULT_DIR")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    details = read_jsonl(path / "details.jsonl")
    if summary.get("rubric_version") != "v2":
        raise ValueError(f"promotion requires rubric v2: {path}")
    return summary, details


def recall(details: list[Mapping[str, Any]], task: str, label: str) -> float:
    rows = [row for row in details if row["task_type"] == task and row["target_status"] == label]
    if not rows:
        raise ValueError(f"no target rows for {task}/{label}")
    return sum(row.get("predicted_status") == label for row in rows) / len(rows)


def candidate_gates(
    baseline: Mapping[str, Any], summary: Mapping[str, Any], details: list[dict[str, Any]]
) -> dict[str, Any]:
    feasible_rows = [
        row
        for row in details
        if row["task_type"] == "constrained_intervention" and row["target_status"] == "feasible"
    ]
    feasible_success = statistics.fmean(float(row.get("simulator_outcome_success", 0.0)) for row in feasible_rows)
    metrics = {
        "control_feasible_recall": recall(details, "constrained_intervention", "feasible"),
        "control_infeasible_recall": recall(
            details, "constrained_intervention", "infeasible_within_limits"
        ),
        "control_feasible_simulator_success": feasible_success,
        "schema_valid_rate": float(summary["schema_valid_rate"]),
        "status_macro_f1": {
            task: float(summary["per_task"][task]["status_macro_f1"]) for task in STATUS_TASKS
        },
        "anchor_regressions": {
            task: float(summary["per_task"][task]["task_score"])
            - float(baseline["per_task"][task]["task_score"])
            for task in ANCHOR_TASKS
        },
    }
    gates = {
        "control_feasible_recall_at_least_0_60": metrics["control_feasible_recall"] >= 0.60,
        "control_infeasible_recall_at_least_0_60": metrics["control_infeasible_recall"] >= 0.60,
        "control_simulator_success_at_least_0_40": feasible_success >= 0.40,
        "schema_valid_rate_at_least_0_93": metrics["schema_valid_rate"] >= 0.93,
        "all_status_macro_f1_at_least_0_60": all(value >= 0.60 for value in metrics["status_macro_f1"].values()),
        "no_anchor_regression_below_minus_0_05": all(
            value >= -0.05 for value in metrics["anchor_regressions"].values()
        ),
    }
    return {"macro_task_score": summary["macro_task_score"], "metrics": metrics, "gates": gates, "passed": all(gates.values())}


def main() -> None:
    args = parse_args()
    baseline_summary, _ = load_result(args.baseline_dir)
    candidates: dict[str, Any] = {}
    for raw in args.candidate:
        if "=" not in raw:
            raise ValueError(f"invalid candidate: {raw}")
        name, path = raw.split("=", 1)
        summary, details = load_result(Path(path))
        candidates[name] = candidate_gates(baseline_summary, summary, details)
    passing = sorted(name for name, result in candidates.items() if result["passed"])
    result = {
        "protocol": "corrective_v2",
        "candidate_count": len(candidates),
        "required_passing_seeds": 2,
        "passing_seeds": passing,
        "promotion_status": "promotable" if len(passing) >= 2 else "no_promotion",
        "sealed_test_evaluated": False,
        "candidates": candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
