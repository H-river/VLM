#!/usr/bin/env python3
"""Summarize the frozen Branch-A refinement over matched planner seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_branch_a_refinement_seeds_v2"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(2026080101, 2026080102, 2026080103),
    )
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="Require matched oracle-known controls and summarize impact/recovery too.",
    )
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    seeds = tuple(args.seeds)
    if len(seeds) != len(set(seeds)) or 2026080101 not in seeds:
        raise ValueError("seeds must be unique and include preregistered seed 2026080101")
    reports = []
    for seed in seeds:
        direct_name = "direct" if seed == seeds[0] else f"direct_seed_{seed}"
        policy_name = (
            "probe_symmetric_pair_f0p1_no_residual"
            if seed == seeds[0]
            else f"probe_nores_seed_{seed}"
        )
        direct = _jsonl(root / "control" / f"{direct_name}.jsonl")
        policy = _jsonl(root / "control" / f"{policy_name}.jsonl")
        if len(direct) != 150 or len(policy) != 150:
            raise ValueError(f"seed {seed} is incomplete")
        oracle = None
        if args.include_oracle:
            oracle_name = "oracle_known" if seed == seeds[0] else f"oracle_known_seed_{seed}"
            oracle = _jsonl(root / "control" / f"{oracle_name}.jsonl")
            if len(oracle) != 150:
                raise ValueError(f"oracle seed {seed} is incomplete")
        nominal_direct = [
            row for row in direct if float(row["evaluator_only_true_gain"]) == 1.0
        ]
        fault_direct = [row for row in direct if float(row["evaluator_only_true_gain"]) != 1.0]
        fault_policy = [row for row in policy if float(row["evaluator_only_true_gain"]) != 1.0]
        rate = lambda rows: float(np.mean([bool(row["strict_success"]) for row in rows]))
        seed_report = {
            "seed": seed,
            "direct_fault_success": rate(fault_direct),
            "refinement_fault_success": rate(fault_policy),
            "refinement_control_value": rate(fault_policy) - rate(fault_direct),
            "gain_classification_accuracy": float(
                np.mean([bool(row["gain_classification_correct"]) for row in policy])
            ),
            "mean_total_additional_steps": float(
                np.mean([int(row["total_additional_steps"]) for row in policy])
            ),
        }
        if oracle is not None:
            fault_oracle = [
                row for row in oracle if float(row["evaluator_only_true_gain"]) != 1.0
            ]
            seed_report.update(
                {
                    "direct_nominal_success": rate(nominal_direct),
                    "fault_impact": rate(nominal_direct) - rate(fault_direct),
                    "oracle_fault_success": rate(fault_oracle),
                    "oracle_recovery": rate(fault_oracle) - rate(fault_direct),
                }
            )
        reports.append(seed_report)
    values = np.asarray([row["refinement_control_value"] for row in reports])
    rng = np.random.default_rng(2026080124)
    bootstrapped_means = np.asarray(
        [
            np.mean(values[rng.integers(0, len(values), len(values))])
            for _ in range(10000)
        ]
    )
    bootstrap_low, bootstrap_high = np.quantile(
        bootstrapped_means, (0.025, 0.975)
    )
    report: dict[str, Any] = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "seeds": reports,
        "control_value": {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "standard_deviation": float(values.std(ddof=1)),
            "planner_seed_bootstrap_mean_95": {
                "estimate": float(values.mean()),
                "low": float(bootstrap_low),
                "high": float(bootstrap_high),
                "resampling_unit": "planner_seed",
                "samples": 10000,
            },
            "passes_five_points_every_seed": bool(np.all(values >= 0.05)),
        },
    }
    if args.include_oracle:
        gate_components = {}
        for offset, metric in enumerate(("fault_impact", "oracle_recovery"), start=1):
            metric_values = np.asarray([row[metric] for row in reports])
            metric_rng = np.random.default_rng(2026080124 + offset)
            metric_bootstrap = np.asarray(
                [
                    np.mean(
                        metric_values[
                            metric_rng.integers(0, len(metric_values), len(metric_values))
                        ]
                    )
                    for _ in range(10000)
                ]
            )
            low, high = np.quantile(metric_bootstrap, (0.025, 0.975))
            gate_components[metric] = {
                "mean": float(metric_values.mean()),
                "minimum": float(metric_values.min()),
                "maximum": float(metric_values.max()),
                "standard_deviation": float(metric_values.std(ddof=1)),
                "planner_seed_bootstrap_mean_95": {
                    "estimate": float(metric_values.mean()),
                    "low": float(low),
                    "high": float(high),
                    "resampling_unit": "planner_seed",
                    "samples": 10000,
                },
                "passes_five_points_every_seed": bool(
                    np.all(metric_values >= 0.05)
                ),
            }
        report["gate_components"] = gate_components
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
