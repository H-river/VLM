#!/usr/bin/env python3
"""Apply frozen promotion gates and select one orchestration checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STAGE1_GATES = {
    "schema_valid_rate": 0.99,
    "status_exact_accuracy": 0.97,
    "ready_route_exact_accuracy": 0.95,
    "clarification_recall": 0.95,
    "unsupported_recall": 0.95,
}

STAGE2_GATES = {
    "schema_valid_rate": 0.99,
    "registry_valid_ready_call_rate": 0.99,
    "ready_route_exact_accuracy": 0.95,
    "required_argument_group_exact_accuracy": 0.97,
    "numeric_value_unit_exact_accuracy": 0.95,
    "image_role_exact_accuracy": 0.98,
    "clarification_recall": 0.95,
    "unsupported_recall": 0.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def checkpoint_step(report: dict[str, Any]) -> int:
    match = re.search(r"checkpoint-(\d+)", report["adapter_path"])
    if not match:
        raise ValueError(
            f"adapter path has no checkpoint step: {report['adapter_path']}"
        )
    return int(match.group(1))


def assess(report: dict[str, Any], stage: str) -> dict[str, Any]:
    metrics = report["metrics"]
    gates = STAGE1_GATES if stage == "stage1" else STAGE2_GATES
    checks = {
        name: {
            "value": float(metrics.get(name, 0.0)),
            "minimum": minimum,
            "passed": float(metrics.get(name, 0.0)) >= minimum,
        }
        for name, minimum in gates.items()
    }
    checks["ready_prediction_on_unsupported"] = {
        "value": float(metrics.get("ready_prediction_on_unsupported", 1.0)),
        "maximum": 0.01,
        "passed": float(metrics.get("ready_prediction_on_unsupported", 1.0))
        <= 0.01,
    }
    return {
        "adapter_path": report["adapter_path"],
        "checkpoint_step": checkpoint_step(report),
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
        "metrics": metrics,
    }


def selection_key(candidate: dict[str, Any], stage: str) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    if stage == "stage1":
        return (
            float(metrics["ready_route_exact_accuracy"]),
            float(metrics["status_exact_accuracy"]),
            -float(candidate["checkpoint_step"]),
        )
    return (
        float(metrics["registry_valid_ready_call_rate"]),
        float(metrics["ready_route_exact_accuracy"]),
        float(metrics["ready_arguments_exact_accuracy"]),
        -float(candidate["checkpoint_step"]),
    )


def select(
    reports: list[dict[str, Any]], stage: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    assessments = [assess(report, stage) for report in reports]
    passing = [candidate for candidate in assessments if candidate["passed"]]
    selected = max(passing, key=lambda item: selection_key(item, stage), default=None)
    return assessments, selected


def main() -> None:
    args = parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.summary
    ]
    for report in reports:
        if report["stage"] != args.stage:
            raise ValueError(
                f"summary stage {report['stage']} does not match {args.stage}"
            )
    assessments, selected = select(reports, args.stage)
    output = {
        "stage": args.stage,
        "selection_policy": (
            "all gates, then lexicographic metrics, then earliest checkpoint"
        ),
        "candidates": assessments,
        "selected_checkpoint": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
