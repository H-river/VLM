#!/usr/bin/env python3
"""Compare exact control-horizon curves across planner seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "active_diagnosis_v13_control_horizon_seed_robustness_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_curve(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("curve must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("curve must be LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", action="append", type=_parse_curve, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.curve) < 2:
        raise ValueError("at least two planner-seed curves are required")
    curves = {label: json.loads(path.resolve().read_text()) for label, path in args.curve}
    if len(curves) != len(args.curve):
        raise ValueError("curve labels must be unique")
    for label, curve in curves.items():
        if curve.get("protected_set_used") is not False:
            raise ValueError(f"{label} is not development-only")
    output_arms = {}
    for arm in ("direct", "probe"):
        indexed = {
            label: {
                int(row["horizon"]): row for row in curve["arms"][arm]["curve"]
            }
            for label, curve in curves.items()
        }
        horizons = sorted(set.intersection(*(set(rows) for rows in indexed.values())))
        if horizons != list(range(4, 9)):
            raise ValueError(f"{arm} curves do not contain exact horizons 4 through 8")
        success = {
            label: {
                horizon: float(rows[horizon]["summary"]["fault"]["strict_success"])
                for horizon in horizons
            }
            for label, rows in indexed.items()
        }
        per_horizon = []
        for horizon in horizons:
            values = {label: success[label][horizon] for label in sorted(success)}
            per_horizon.append(
                {
                    "horizon": horizon,
                    "fault_success_by_seed": values,
                    "mean_fault_success": float(np.mean(list(values.values()))),
                    "minimum_fault_success": min(values.values()),
                    "maximum_fault_success": max(values.values()),
                }
            )
        gains6 = {label: rows[6] - rows[4] for label, rows in success.items()}
        gains8 = {label: rows[8] - rows[4] for label, rows in success.items()}
        marginal68 = {label: rows[8] - rows[6] for label, rows in success.items()}
        earliest_max = {
            label: min(
                horizon
                for horizon in horizons
                if rows[horizon] == max(rows.values())
            )
            for label, rows in success.items()
        }
        earliest_all_final = next(
            (
                horizon
                for horizon in horizons
                if all(
                    success[label][horizon] >= success[label][8]
                    for label in success
                )
            ),
            None,
        )
        output_arms[arm] = {
            "curve": per_horizon,
            "fault_gain_h6_vs_h4_by_seed": gains6,
            "fault_gain_h8_vs_h4_by_seed": gains8,
            "fault_marginal_h8_vs_h6_by_seed": marginal68,
            "all_seeds_positive_h6_vs_h4": all(value > 0.0 for value in gains6.values()),
            "mean_gain_h6_vs_h4": float(np.mean(list(gains6.values()))),
            "minimum_gain_h6_vs_h4": min(gains6.values()),
            "mean_marginal_h8_vs_h6": float(np.mean(list(marginal68.values()))),
            "earliest_maximum_horizon_by_seed": earliest_max,
            "earliest_horizon_matching_every_seed_h8_success": earliest_all_final,
        }
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "planner_seeds": sorted(curves),
        "arms": output_arms,
        "interpretation_guard": (
            "Every curve is reconstructed from exact matched prefixes. Planner seeds vary "
            "CEM sampling while setup groups and the frozen estimator remain unchanged."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
