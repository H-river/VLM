#!/usr/bin/env python3
"""Separate a one-sided probe's physical treatment effect from diagnosis value."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.compare_control_policies import _bootstrap


VERSION = "active_diagnosis_v13_probe_treatment_confound_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _pair(
    left: list[dict[str, Any]], right: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    left_map, right_map = ({_key(row): row for row in rows} for rows in (left, right))
    if set(left_map) != set(right_map):
        raise ValueError("probe-treatment arms are not episode matched")
    rate = lambda rows: float(np.mean([bool(row["strict_success"]) for row in rows]))
    return {
        "success_gain": rate(right) - rate(left),
        "matched_recoveries": int(
            sum(
                not bool(left_map[key]["strict_success"])
                and bool(right_map[key]["strict_success"])
                for key in left_map
            )
        ),
        "matched_regressions": int(
            sum(
                bool(left_map[key]["strict_success"])
                and not bool(right_map[key]["strict_success"])
                for key in left_map
            )
        ),
        "matched_group_bootstrap_95": _bootstrap(left, right, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--estimated-policy", required=True)
    parser.add_argument("--constant-policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    modes = {
        "direct": _jsonl(root / "control/direct.jsonl"),
        "constant_nominal": _jsonl(
            root / "control" / f"{args.constant_policy}.jsonl"
        ),
        "estimated_gain": _jsonl(
            root / "control" / f"{args.estimated_policy}.jsonl"
        ),
    }
    if any(len(rows) != 150 for rows in modes.values()):
        raise ValueError("probe-treatment arms must each contain 150 episodes")
    if len({_key(row) for row in modes["direct"]}) != 150:
        raise ValueError("duplicate direct episodes")
    if any(
        int(str(row["case_id"]).rsplit("_", 1)[1]) >= 10
        for rows in modes.values()
        for row in rows
    ):
        raise ValueError("protected case present in probe-treatment analysis")
    fault = {
        name: [row for row in rows if float(row["evaluator_only_true_gain"]) != 1.0]
        for name, rows in modes.items()
    }
    rate = lambda rows: float(np.mean([bool(row["strict_success"]) for row in rows]))
    constant_rows = modes["constant_nominal"]
    estimated_rows = modes["estimated_gain"]
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "estimated_policy": args.estimated_policy,
        "constant_policy": args.constant_policy,
        "probe_design": estimated_rows[0]["probe_design"],
        "probe_fraction": estimated_rows[0]["probe_fraction"],
        "fault_success": {name: rate(rows) for name, rows in fault.items()},
        "physical_probe_treatment_effect_constant_nominal_vs_direct": _pair(
            fault["direct"], fault["constant_nominal"], 2026080118
        ),
        "incremental_diagnosis_value_estimated_vs_constant_nominal": _pair(
            fault["constant_nominal"], fault["estimated_gain"], 2026080119
        ),
        "total_estimated_policy_value_vs_direct": _pair(
            fault["direct"], fault["estimated_gain"], 2026080120
        ),
        "estimated_policy_gain_accuracy": float(
            np.mean([bool(row["gain_classification_correct"]) for row in estimated_rows])
        ),
        "constant_policy_gain_accuracy": float(
            np.mean([bool(row["gain_classification_correct"]) for row in constant_rows])
        ),
        "mean_final_beam_state_disturbance": float(
            np.mean(
                [
                    float(row["probe_record"]["absolute_beam_state_disturbance"])
                    for row in estimated_rows
                ]
            )
        ),
        "mean_signed_target_cost_disturbance": float(
            np.mean(
                [
                    float(row["probe_record"]["final_target_cost_after_probe"])
                    - float(row["probe_record"]["initial_target_cost"])
                    for row in estimated_rows
                ]
            )
        ),
        "interpretation_constraint": (
            "one-sided probe treatment and diagnosis are development-only ablations; "
            "they cannot replace the frozen safe symmetric design"
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
