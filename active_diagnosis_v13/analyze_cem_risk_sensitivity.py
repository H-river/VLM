#!/usr/bin/env python3
"""Audit whether CEM uncertainty penalties change plans or development outcomes."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import ACTION_FIELDS


VERSION = "active_diagnosis_v13_cem_risk_sensitivity_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _command_vector(step: dict[str, Any]) -> np.ndarray:
    command = step["command_mm"]
    return np.asarray(
        [command[name] for name in ACTION_FIELDS],
        dtype=np.float64,
    )


def _plan_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    step_deltas = [
        float(np.max(np.abs(_command_vector(a) - _command_vector(b))))
        for a, b in zip(left["trace"], right["trace"], strict=False)
    ]
    length_changed = len(left["trace"]) != len(right["trace"])
    maximum = max(step_deltas, default=0.0)
    return {
        "changed": bool(length_changed or maximum > 1e-12),
        "trace_length_changed": length_changed,
        "maximum_absolute_command_delta_mm": maximum,
    }


def _slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "strict_success": _rate(rows),
        "mean_final_normalized_distance": float(
            np.mean([float(row["final_normalized_distance"]) for row in rows])
        ),
        "mean_total_additional_steps": float(
            np.mean([int(row["total_additional_steps"]) for row in rows])
        ),
        "saturation_episode_rate": float(
            np.mean([int(row["saturation_count"]) > 0 for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Uncertainty weight and control filename stem as WEIGHT:NAME",
    )
    parser.add_argument("--baseline-weight", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve() / "control"
    parsed = []
    for value in args.arm:
        weight_text, name = value.split(":", 1)
        parsed.append((float(weight_text), name))
    if len({weight for weight, _ in parsed}) != len(parsed):
        raise ValueError("uncertainty weights must be unique")
    rows_by_weight = {
        weight: _jsonl(root / f"{name}.jsonl") for weight, name in parsed
    }
    names = dict(parsed)
    maps = {
        weight: {_key(row): row for row in rows}
        for weight, rows in rows_by_weight.items()
    }
    if args.baseline_weight not in maps:
        raise ValueError("baseline weight is absent")
    reference_keys = set(maps[args.baseline_weight])
    for weight, rows in rows_by_weight.items():
        if len(rows) != 150 or len(maps[weight]) != 150 or set(maps[weight]) != reference_keys:
            raise ValueError(f"weight {weight} is incomplete, duplicated, or unmatched")
    if any(int(case_id.rsplit("_", 1)[1]) >= 10 for case_id, _ in reference_keys):
        raise ValueError("protected case present in risk sensitivity")

    baseline = maps[args.baseline_weight]
    arms: list[dict[str, Any]] = []
    fault_outcome_vectors: list[tuple[bool, ...]] = []
    for weight, _ in sorted(parsed):
        current = maps[weight]
        rows = list(current.values())
        fault_keys = sorted(key for key in reference_keys if key[1] != 1.0)
        nominal_keys = sorted(key for key in reference_keys if key[1] == 1.0)
        fault = [current[key] for key in fault_keys]
        nominal = [current[key] for key in nominal_keys]
        deltas = {key: _plan_delta(baseline[key], current[key]) for key in reference_keys}
        changed = sorted(key for key, value in deltas.items() if value["changed"])
        recoveries = sorted(
            key
            for key in reference_keys
            if not bool(baseline[key]["strict_success"])
            and bool(current[key]["strict_success"])
        )
        regressions = sorted(
            key
            for key in reference_keys
            if bool(baseline[key]["strict_success"])
            and not bool(current[key]["strict_success"])
        )
        by_gain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_gain[f"{float(row['evaluator_only_true_gain']):g}"].append(row)
            by_stratum[str(row["stratum"])].append(row)
        arms.append(
            {
                "uncertainty_weight": weight,
                "policy": names[weight],
                "overall": _slice(rows),
                "fault": _slice(fault),
                "nominal": _slice(nominal),
                "by_gain": {
                    key: _slice(value)
                    for key, value in sorted(by_gain.items(), key=lambda item: float(item[0]))
                },
                "by_stratum": {
                    key: _slice(value) for key, value in sorted(by_stratum.items())
                },
                "episodes_with_changed_plan": len(changed),
                "fault_episodes_with_changed_plan": int(
                    sum(key[1] != 1.0 for key in changed)
                ),
                "trace_length_changed_episodes": int(
                    sum(value["trace_length_changed"] for value in deltas.values())
                ),
                "maximum_absolute_command_delta_mm": max(
                    (value["maximum_absolute_command_delta_mm"] for value in deltas.values()),
                    default=0.0,
                ),
                "matched_recoveries_over_baseline": len(recoveries),
                "matched_regressions_over_baseline": len(regressions),
                "fault_recoveries_over_baseline": int(
                    sum(key[1] != 1.0 for key in recoveries)
                ),
                "fault_regressions_over_baseline": int(
                    sum(key[1] != 1.0 for key in regressions)
                ),
                "exact_changed_plan_episode_ids": [
                    f"{case_id}__g{gain:g}" for case_id, gain in changed
                ],
                "exact_recovery_episode_ids": [
                    f"{case_id}__g{gain:g}" for case_id, gain in recoveries
                ],
                "exact_regression_episode_ids": [
                    f"{case_id}__g{gain:g}" for case_id, gain in regressions
                ],
            }
        )
        fault_outcome_vectors.append(
            tuple(bool(current[key]["strict_success"]) for key in fault_keys)
        )
    report = {
        "version": VERSION,
        "role": "supporting_development_risk_ablation_primary_branch_not_reselected",
        "split": "development_only",
        "protected_set_used": False,
        "baseline_uncertainty_weight": args.baseline_weight,
        "arms": arms,
        "fault_outcomes_identical_across_tested_weights": bool(
            len(set(fault_outcome_vectors)) == 1
        ),
        "interpretation_constraint": (
            "uncertainty-score sensitivity only; no H1 retraining and no protected selection"
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
