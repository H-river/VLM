#!/usr/bin/env python3
"""Evaluate development-only confidence fallback from gain belief to nominal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_confidence_gate_synthetic_v1"
THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    gain_key = (
        "true_gain_evaluator_only"
        if "true_gain_evaluator_only" in row
        else "evaluator_only_true_gain"
    )
    return str(row["case_id"]), float(row[gain_key])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    direct = {_key(row): row for row in _jsonl(root / "control" / "direct.jsonl")}
    probe = {_key(row): row for row in _jsonl(root / "control" / "probe_replan.jsonl")}
    predictions = {_key(row): row for row in _jsonl(args.oof_predictions.resolve())}
    if set(direct) != set(probe) or set(direct) != set(predictions):
        raise ValueError("confidence gate inputs are not matched over all 150 episodes")
    rows = []
    for threshold in THRESHOLDS:
        decisions = []
        for key in sorted(direct):
            use_gain = float(predictions[key]["confidence"]) >= threshold
            chosen = probe[key] if use_gain else direct[key]
            decisions.append(
                {
                    "key": key,
                    "use_estimated_gain": use_gain,
                    "success": bool(chosen["strict_success"]),
                    "direct_success": bool(direct[key]["strict_success"]),
                    "total_steps": (
                        int(probe[key]["total_additional_steps"])
                        if use_gain
                        else 2 + int(direct[key]["total_additional_steps"])
                    ),
                }
            )
        fault = [row for row in decisions if row["key"][1] != 1.0]
        rate = lambda values: float(np.mean([row["success"] for row in values]))
        direct_fault = float(np.mean([row["direct_success"] for row in fault]))
        rows.append(
            {
                "confidence_threshold": threshold,
                "episodes": len(decisions),
                "estimated_gain_usage_rate": float(
                    np.mean([row["use_estimated_gain"] for row in decisions])
                ),
                "overall_success": rate(decisions),
                "fault_success": rate(fault),
                "fault_success_gain_over_direct": rate(fault) - direct_fault,
                "matched_direct_failure_recoveries": int(
                    sum(not row["direct_success"] and row["success"] for row in fault)
                ),
                "matched_direct_success_regressions": int(
                    sum(row["direct_success"] and not row["success"] for row in fault)
                ),
                "mean_total_additional_steps": float(
                    np.mean([row["total_steps"] for row in decisions])
                ),
            }
        )
    selected = sorted(
        rows,
        key=lambda row: (
            -float(row["fault_success"]),
            int(row["matched_direct_success_regressions"]),
            -float(row["confidence_threshold"]),
        ),
    )[0]
    report: dict[str, Any] = {
        "version": VERSION,
        "split": "development_only_group_out_of_fold",
        "protected_set_used": False,
        "method": (
            "exact outcome substitution: symmetric probe returns to identical state; "
            "below threshold uses the matched nominal/direct controller outcome"
        ),
        "thresholds": rows,
        "selected_development_threshold": selected,
        "selection_rule": "fault_success_then_fewer_regressions_then_higher_threshold",
        "requires_closed_loop_replay_confirmation": True,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
