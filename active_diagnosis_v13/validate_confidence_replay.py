#!/usr/bin/env python3
"""Validate a confidence-gated controller against its exact synthetic projection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_confidence_replay_validation_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    gain_key = (
        "true_gain_evaluator_only"
        if "true_gain_evaluator_only" in row
        else "evaluator_only_true_gain"
    )
    return str(row["case_id"]), float(row[gain_key])


def _trace_projection(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "command_mm": step["command_mm"],
            "predicted_next_metrics": step["predicted_next_metrics"],
            "observed_next_metrics": step["observed_next_metrics"],
            "actual_target_cost": step["actual_target_cost"],
        }
        for step in row["trace"]
    ]


def _values_close(left: Any, right: Any, atol: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _values_close(left[key], right[key], atol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_close(a, b, atol) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--actual-policy", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    sources = {
        "direct": {_key(row): row for row in _jsonl(root / "control/direct.jsonl")},
        "probe": {
            _key(row): row for row in _jsonl(root / "control/probe_replan.jsonl")
        },
        "actual": {
            _key(row): row
            for row in _jsonl(root / "control" / f"{args.actual_policy}.jsonl")
        },
        "prediction": {
            _key(row): row for row in _jsonl(args.oof_predictions.resolve())
        },
    }
    expected_keys = set(sources["direct"])
    if len(expected_keys) != 150 or any(
        set(rows) != expected_keys for rows in sources.values()
    ):
        raise ValueError("confidence replay inputs must match over 150 episodes")
    if any(int(case.rsplit("_", 1)[1]) >= 10 for case, _ in expected_keys):
        raise ValueError("protected case present in confidence replay")

    mismatches: list[dict[str, Any]] = []
    max_distance_difference = 0.0
    estimated_gain_uses = 0
    for key in sorted(expected_keys):
        prediction = sources["prediction"][key]
        actual = sources["actual"][key]
        use_estimate = float(prediction["confidence"]) >= args.threshold
        estimated_gain_uses += int(use_estimate)
        expected = sources["probe"][key] if use_estimate else sources["direct"][key]
        distance_difference = abs(
            float(actual["final_normalized_distance"])
            - float(expected["final_normalized_distance"])
        )
        max_distance_difference = max(max_distance_difference, distance_difference)
        reasons = []
        if bool(actual["confidence_fallback_to_nominal"]) == use_estimate:
            reasons.append("fallback flag")
        if not np.isclose(
            float(actual["gain_estimate_confidence"]),
            float(prediction["confidence"]),
            rtol=0.0,
            atol=1e-12,
        ):
            reasons.append("confidence")
        if not np.isclose(
            float(actual["raw_gain_estimate"]),
            float(prediction["estimated_gain"]),
            rtol=0.0,
            atol=1e-12,
        ):
            reasons.append("raw gain estimate")
        if bool(actual["strict_success"]) != bool(expected["strict_success"]):
            reasons.append("strict outcome")
        if distance_difference > 1e-12:
            reasons.append("final distance")
        if not _values_close(actual["final_metrics"], expected["final_metrics"]):
            reasons.append("final metrics")
        if not _values_close(_trace_projection(actual), _trace_projection(expected)):
            reasons.append("control trace")
        if reasons:
            mismatches.append(
                {
                    "case_id": key[0],
                    "true_gain": key[1],
                    "use_estimated_gain": use_estimate,
                    "reasons": reasons,
                    "final_distance_difference": distance_difference,
                }
            )

    actual_rows = list(sources["actual"].values())
    fault = [row for row in actual_rows if float(row["evaluator_only_true_gain"]) != 1.0]
    direct_fault = [
        row
        for row in sources["direct"].values()
        if float(row["evaluator_only_true_gain"]) != 1.0
    ]
    rate = lambda rows: float(np.mean([bool(row["strict_success"]) for row in rows]))
    report: dict[str, Any] = {
        "version": VERSION,
        "split": "development_only_group_out_of_fold",
        "protected_set_used": False,
        "actual_policy": args.actual_policy,
        "confidence_threshold": args.threshold,
        "episodes": len(actual_rows),
        "estimated_gain_usage_rate": estimated_gain_uses / len(actual_rows),
        "fault_success": rate(fault),
        "fault_success_gain_over_direct": rate(fault) - rate(direct_fault),
        "exact_projection_matches": len(actual_rows) - len(mismatches),
        "maximum_final_distance_difference": max_distance_difference,
        "mismatches": mismatches,
        "passes": not mismatches,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
