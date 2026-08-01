#!/usr/bin/env python3
"""Summarize a matched development-only CEM population-budget curve."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_cem_population_curve_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "strict_success": _rate(rows),
        "mean_final_distance": float(
            np.mean([float(row["final_normalized_distance"]) for row in rows])
        ),
        "mean_control_steps": float(
            np.mean([int(row["control_steps"]) for row in rows])
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
        help="Population and control filename stem as POPULATION:NAME",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    parsed_arms = []
    for value in args.arm:
        population_text, name = value.split(":", 1)
        parsed_arms.append((int(population_text), name))
    if len({population for population, _ in parsed_arms}) != len(parsed_arms):
        raise ValueError("population arms must be unique")

    rows_by_population = {
        population: _jsonl(root / "control" / f"{name}.jsonl")
        for population, name in parsed_arms
    }
    if any(len(rows) != 150 for rows in rows_by_population.values()):
        raise ValueError("all CEM population arms must contain 150 episodes")
    key_sets = {population: {_key(row) for row in rows} for population, rows in rows_by_population.items()}
    reference_keys = next(iter(key_sets.values()))
    if any(keys != reference_keys for keys in key_sets.values()):
        raise ValueError("CEM population arms are not episode-matched")
    if any(int(case.rsplit("_", 1)[1]) >= 10 for case, _ in reference_keys):
        raise ValueError("protected case present in CEM population curve")

    baseline_population = 24
    if baseline_population not in rows_by_population:
        raise ValueError("population 24 preregistered baseline is required")
    baseline = {_key(row): row for row in rows_by_population[baseline_population]}
    arms = []
    for population, name in sorted(parsed_arms):
        rows = rows_by_population[population]
        fault = [row for row in rows if float(row["evaluator_only_true_gain"]) != 1.0]
        nominal = [row for row in rows if float(row["evaluator_only_true_gain"]) == 1.0]
        by_gain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_gain[f"{float(row['evaluator_only_true_gain']):g}"].append(row)
            by_stratum[str(row["stratum"])].append(row)
        current = {_key(row): row for row in rows}
        fault_keys = [key for key in reference_keys if key[1] != 1.0]
        recoveries = sum(
            not bool(baseline[key]["strict_success"])
            and bool(current[key]["strict_success"])
            for key in fault_keys
        )
        regressions = sum(
            bool(baseline[key]["strict_success"])
            and not bool(current[key]["strict_success"])
            for key in fault_keys
        )
        arms.append(
            {
                "population": population,
                "policy": name,
                "overall": _slice(rows),
                "nominal": _slice(nominal),
                "fault": _slice(fault),
                "fault_success_gain_over_population_24": _rate(fault)
                - _rate(
                    [
                        row
                        for row in rows_by_population[24]
                        if float(row["evaluator_only_true_gain"]) != 1.0
                    ]
                ),
                "matched_fault_recoveries_over_population_24": int(recoveries),
                "matched_fault_regressions_over_population_24": int(regressions),
                "by_gain": {
                    key: _slice(value)
                    for key, value in sorted(by_gain.items(), key=lambda item: float(item[0]))
                },
                "by_stratum": {
                    key: _slice(value) for key, value in sorted(by_stratum.items())
                },
            }
        )
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "matched_episodes_per_arm": 150,
        "baseline_population": 24,
        "arms": arms,
        "best_development_fault_arm": max(
            arms, key=lambda row: float(row["fault"]["strict_success"])
        )["population"],
        "interpretation_constraint": (
            "supporting CEM sensitivity only; it cannot alter the frozen Gate-A policy"
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
