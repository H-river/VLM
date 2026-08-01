#!/usr/bin/env python3
"""Trace the success/saturation frontier for a visible posterior-mean margin guard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.compare_control_policies import _bootstrap


VERSION = "active_diagnosis_v13_guarded_margin_frontier_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--discrete-policy", required=True)
    parser.add_argument("--continuous-policy", required=True)
    parser.add_argument("--discrete-threshold", type=float, default=1.0)
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=(0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve() / "control"
    discrete = {_key(row): row for row in _jsonl(root / f"{args.discrete_policy}.jsonl")}
    continuous = {
        _key(row): row for row in _jsonl(root / f"{args.continuous_policy}.jsonl")
    }
    if len(discrete) != 150 or len(continuous) != 150 or set(discrete) != set(continuous):
        raise ValueError("discrete and continuous policies must contain 150 matched rows")
    margins = sorted(set(float(value) for value in args.margins))
    if not margins or margins[0] < 0:
        raise ValueError("margins must be nonnegative")
    discrete_fault = [
        row for row in discrete.values() if float(row["evaluator_only_true_gain"]) != 1.0
    ]
    cells: list[dict[str, Any]] = []
    for index, margin in enumerate(margins):
        selected: list[dict[str, Any]] = []
        continuous_keys: set[tuple[str, float]] = set()
        for key in sorted(discrete):
            discrete_row = discrete[key]
            continuous_row = continuous[key]
            use_continuous = (
                float(discrete_row["gain_belief"]) >= args.discrete_threshold
                and float(continuous_row["gain_belief"])
                >= float(discrete_row["gain_belief"]) - margin
            )
            if use_continuous:
                continuous_keys.add(key)
                selected.append(continuous_row)
            else:
                selected.append(discrete_row)
        selected_fault = [
            row for row in selected if float(row["evaluator_only_true_gain"]) != 1.0
        ]
        discrete_fault_by_key = {_key(row): row for row in discrete_fault}
        selected_fault_by_key = {_key(row): row for row in selected_fault}
        cells.append(
            {
                "margin": margin,
                "continuous_episode_count": len(continuous_keys),
                "continuous_fault_episode_count": int(
                    sum(key[1] != 1.0 for key in continuous_keys)
                ),
                "overall_success": _rate(selected),
                "fault_success": _rate(selected_fault),
                "fault_success_difference_over_discrete": _rate(selected_fault)
                - _rate(discrete_fault),
                "matched_group_bootstrap_95_over_discrete": _bootstrap(
                    discrete_fault,
                    selected_fault,
                    2026080140 + index,
                ),
                "fault_saturation_episode_rate": float(
                    np.mean(
                        [int(row["saturation_count"]) > 0 for row in selected_fault]
                    )
                ),
                "fault_recoveries_over_discrete": int(
                    sum(
                        not bool(discrete_fault_by_key[key]["strict_success"])
                        and bool(selected_fault_by_key[key]["strict_success"])
                        for key in discrete_fault_by_key
                    )
                ),
                "fault_regressions_over_discrete": int(
                    sum(
                        bool(discrete_fault_by_key[key]["strict_success"])
                        and not bool(selected_fault_by_key[key]["strict_success"])
                        for key in discrete_fault_by_key
                    )
                ),
            }
        )
    pareto_margins = []
    for cell in cells:
        dominated = any(
            other["fault_success"] >= cell["fault_success"]
            and other["fault_saturation_episode_rate"]
            <= cell["fault_saturation_episode_rate"]
            and (
                other["fault_success"] > cell["fault_success"]
                or other["fault_saturation_episode_rate"]
                < cell["fault_saturation_episode_rate"]
            )
            for other in cells
        )
        if not dominated:
            pareto_margins.append(cell["margin"])
    report = {
        "version": VERSION,
        "role": "development_only_cached_policy_switch_frontier_no_protected_reselection",
        "split": "development_only",
        "protected_set_used": False,
        "discrete_policy": args.discrete_policy,
        "continuous_policy": args.continuous_policy,
        "discrete_threshold": args.discrete_threshold,
        "selection_inputs_are_visible_predictions_only": True,
        "cells": cells,
        "pareto_margins": pareto_margins,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
