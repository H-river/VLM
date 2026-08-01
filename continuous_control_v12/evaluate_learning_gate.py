#!/usr/bin/env python3
"""Evaluate the pre-registered v12 forward-model gate for learned MPC."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def check(
    name: str,
    value: Any,
    threshold: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overfit-report", type=Path, required=True)
    parser.add_argument("--learning-curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite gate result: {output}")
    overfit = json.loads(
        args.overfit_report.resolve().read_text(encoding="utf-8")
    )
    curve = json.loads(
        args.learning_curve.resolve().read_text(encoding="utf-8")
    )
    by_groups = {int(row["train_groups"]): row for row in curve}
    missing = sorted({16, 32, 64, 128} - set(by_groups))
    if missing:
        raise ValueError(f"learning curve is missing nested groups: {missing}")
    small = by_groups[16]
    largest = by_groups[128]
    overfit_accuracy = float(overfit["fit"]["strict_all_five_accuracy"])
    dev_16 = float(small["development_normalized_mae"])
    dev_128 = float(largest["development_normalized_mae"])
    direction = float(largest["paired_direction_sign_accuracy"])
    jacobian = float(
        largest["median_relative_directional_jacobian_error"]
    )
    rollout_h1 = float(largest["rollout_h1_normalized_mae"])
    rollout_h3 = float(largest["rollout_h3_normalized_mae"])
    finite = bool(largest["finite_predictions"]) and all(
        math.isfinite(value)
        for value in (
            overfit_accuracy,
            dev_16,
            dev_128,
            direction,
            jacobian,
            rollout_h1,
            rollout_h3,
        )
    )
    checks = [
        check(
            "production_overfit_strict_all_five_accuracy",
            overfit_accuracy,
            ">= 0.95",
            overfit_accuracy >= 0.95,
        ),
        check(
            "finite_predictions_and_gate_metrics",
            finite,
            "true",
            finite,
        ),
        check(
            "development_mae_improves_16_to_128_groups",
            {"groups_16": dev_16, "groups_128": dev_128},
            "groups_128 < groups_16",
            dev_128 < dev_16,
        ),
        check(
            "test_paired_direction_sign_accuracy",
            direction,
            ">= 0.65",
            direction >= 0.65,
        ),
        check(
            "test_median_relative_directional_jacobian_error",
            jacobian,
            "< 1.0",
            jacobian < 1.0,
        ),
        check(
            "test_horizon_3_rollout_mae",
            {"horizon_1": rollout_h1, "horizon_3": rollout_h3},
            "finite and horizon_3 <= 2 * horizon_1",
            math.isfinite(rollout_h3) and rollout_h3 <= 2.0 * rollout_h1,
        ),
    ]
    result = {
        "version": "continuous_forward_learning_gate_v12_v1",
        "preregistered": True,
        "passed": all(item["passed"] for item in checks),
        "learned_mpc_authorized": all(item["passed"] for item in checks),
        "checks": checks,
        "overfit_report": str(args.overfit_report.resolve()),
        "learning_curve": str(args.learning_curve.resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
