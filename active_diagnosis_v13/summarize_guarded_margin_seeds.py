#!/usr/bin/env python3
"""Summarize a margin-guarded source switch over matched planner seeds."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_guarded_margin_seed_summary_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _episode_id(key: tuple[str, float]) -> str:
    return f"{key[0]}__g{key[1]:g}"


def _use_continuous(
    discrete_row: dict[str, Any],
    continuous_row: dict[str, Any],
    *,
    discrete_threshold: float,
    downward_margin: float,
) -> bool:
    """Apply the predeclared guard using estimator-visible values only."""
    return bool(
        float(discrete_row["gain_belief"]) >= discrete_threshold
        and float(continuous_row["gain_belief"])
        >= float(discrete_row["gain_belief"]) - downward_margin
    )


def _seed_summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            np.mean(values[rng.integers(0, len(values), len(values))])
            for _ in range(10000)
        ]
    )
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std(ddof=1)),
        "planner_seed_bootstrap_mean_95": {
            "estimate": float(values.mean()),
            "low": float(low),
            "high": float(high),
            "resampling_unit": "planner_seed",
            "samples": 10000,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--discrete-base-policy", required=True)
    parser.add_argument("--discrete-seed-prefix", required=True)
    parser.add_argument("--continuous-base-policy", required=True)
    parser.add_argument("--continuous-seed-prefix", required=True)
    parser.add_argument("--discrete-threshold", type=float, default=1.0)
    parser.add_argument("--downward-margin", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve() / "control"
    base_seed = int(json.loads(args.config.resolve().read_text())["root_seed"])
    seeds = list(dict.fromkeys(args.seeds))
    if base_seed not in seeds:
        raise ValueError("seeds must include the configured base seed")
    if len(seeds) < 2:
        raise ValueError("at least two planner seeds are required")
    if args.downward_margin < 0:
        raise ValueError("downward margin must be nonnegative")
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        direct_name = "direct" if seed == base_seed else f"direct_seed_{seed}"
        discrete_name = (
            args.discrete_base_policy
            if seed == base_seed
            else f"{args.discrete_seed_prefix}_{seed}"
        )
        continuous_name = (
            args.continuous_base_policy
            if seed == base_seed
            else f"{args.continuous_seed_prefix}_{seed}"
        )
        direct = {_key(row): row for row in _jsonl(root / f"{direct_name}.jsonl")}
        discrete = {
            _key(row): row for row in _jsonl(root / f"{discrete_name}.jsonl")
        }
        continuous = {
            _key(row): row for row in _jsonl(root / f"{continuous_name}.jsonl")
        }
        if (
            len(direct) != 150
            or len(discrete) != 150
            or len(continuous) != 150
            or set(direct) != set(discrete)
            or set(direct) != set(continuous)
        ):
            raise ValueError(f"seed {seed} source policies are incomplete or unmatched")
        selected: dict[tuple[str, float], dict[str, Any]] = {}
        continuous_count = 0
        for key in sorted(discrete):
            discrete_row = discrete[key]
            continuous_row = continuous[key]
            use_continuous = _use_continuous(
                discrete_row,
                continuous_row,
                discrete_threshold=args.discrete_threshold,
                downward_margin=args.downward_margin,
            )
            selected[key] = continuous_row if use_continuous else discrete_row
            continuous_count += int(use_continuous)
        fault_keys = [key for key in direct if key[1] != 1.0]
        direct_fault = [direct[key] for key in fault_keys]
        discrete_fault = [discrete[key] for key in fault_keys]
        continuous_fault = [continuous[key] for key in fault_keys]
        selected_fault = [selected[key] for key in fault_keys]
        direct_success = _rate(direct_fault)
        discrete_success = _rate(discrete_fault)
        selected_success = _rate(selected_fault)
        recovery_keys = sorted(
            key
            for key in fault_keys
            if not bool(discrete[key]["strict_success"])
            and bool(selected[key]["strict_success"])
        )
        regression_keys = sorted(
            key
            for key in fault_keys
            if bool(discrete[key]["strict_success"])
            and not bool(selected[key]["strict_success"])
        )
        added_saturation_keys = sorted(
            key
            for key in fault_keys
            if int(discrete[key]["saturation_count"]) == 0
            and int(selected[key]["saturation_count"]) > 0
        )
        removed_saturation_keys = sorted(
            key
            for key in fault_keys
            if int(discrete[key]["saturation_count"]) > 0
            and int(selected[key]["saturation_count"]) == 0
        )
        reports.append(
            {
                "seed": seed,
                "direct_policy": direct_name,
                "discrete_policy": discrete_name,
                "continuous_source_policy": continuous_name,
                "continuous_episode_count": continuous_count,
                "direct_fault_success": direct_success,
                "discrete_fault_success": discrete_success,
                "selected_fault_success": selected_success,
                "control_value_over_direct": selected_success - direct_success,
                "gain_over_discrete": selected_success - discrete_success,
                "discrete_fault_saturation_episode_rate": float(
                    np.mean(
                        [int(row["saturation_count"]) > 0 for row in discrete_fault]
                    )
                ),
                "continuous_source_fault_saturation_episode_rate": float(
                    np.mean(
                        [int(row["saturation_count"]) > 0 for row in continuous_fault]
                    )
                ),
                "fault_saturation_episode_rate": float(
                    np.mean(
                        [int(row["saturation_count"]) > 0 for row in selected_fault]
                    )
                ),
                "recoveries_over_discrete": len(recovery_keys),
                "regressions_over_discrete": len(regression_keys),
                "net_recoveries_over_discrete": len(recovery_keys)
                - len(regression_keys),
                "exact_recovery_episode_ids": [
                    _episode_id(key) for key in recovery_keys
                ],
                "exact_regression_episode_ids": [
                    _episode_id(key) for key in regression_keys
                ],
                "added_saturation_over_discrete": len(added_saturation_keys),
                "removed_saturation_over_discrete": len(removed_saturation_keys),
                "net_added_saturation_over_discrete": len(added_saturation_keys)
                - len(removed_saturation_keys),
                "exact_added_saturation_episode_ids": [
                    _episode_id(key) for key in added_saturation_keys
                ],
                "exact_removed_saturation_episode_ids": [
                    _episode_id(key) for key in removed_saturation_keys
                ],
            }
        )
    control_values = np.asarray(
        [row["control_value_over_direct"] for row in reports], dtype=np.float64
    )
    gains = np.asarray([row["gain_over_discrete"] for row in reports], dtype=np.float64)
    saturations = np.asarray(
        [row["fault_saturation_episode_rate"] for row in reports], dtype=np.float64
    )
    recovery_seeds: defaultdict[str, list[int]] = defaultdict(list)
    regression_seeds: defaultdict[str, list[int]] = defaultdict(list)
    added_saturation_seeds: defaultdict[str, list[int]] = defaultdict(list)
    removed_saturation_seeds: defaultdict[str, list[int]] = defaultdict(list)
    for row in reports:
        for episode_id in row["exact_recovery_episode_ids"]:
            recovery_seeds[episode_id].append(int(row["seed"]))
        for episode_id in row["exact_regression_episode_ids"]:
            regression_seeds[episode_id].append(int(row["seed"]))
        for episode_id in row["exact_added_saturation_episode_ids"]:
            added_saturation_seeds[episode_id].append(int(row["seed"]))
        for episode_id in row["exact_removed_saturation_episode_ids"]:
            removed_saturation_seeds[episode_id].append(int(row["seed"]))
    report = {
        "version": VERSION,
        "role": "development_only_exact_source_switch_seed_summary_no_protected_reselection",
        "split": "development_only",
        "protected_set_used": False,
        "discrete_threshold": args.discrete_threshold,
        "downward_margin": args.downward_margin,
        "selection_inputs_are_visible_predictions_only": True,
        "seeds": reports,
        "control_value_over_direct": {
            **_seed_summary(control_values, 2026080145),
            "passes_five_points_every_seed": bool(np.all(control_values >= 0.05)),
        },
        "gain_over_discrete": _seed_summary(gains, 2026080146),
        "fault_saturation_episode_rate": {
            "mean": float(saturations.mean()),
            "minimum": float(saturations.min()),
            "maximum": float(saturations.max()),
        },
        "aggregate_seed_episode_tradeoff": {
            "net_recoveries_over_discrete": int(
                sum(row["net_recoveries_over_discrete"] for row in reports)
            ),
            "net_added_saturation_over_discrete": int(
                sum(
                    row["net_added_saturation_over_discrete"] for row in reports
                )
            ),
            "net_added_saturation_per_net_recovery": (
                float(
                    sum(
                        row["net_added_saturation_over_discrete"]
                        for row in reports
                    )
                )
                / sum(row["net_recoveries_over_discrete"] for row in reports)
                if sum(row["net_recoveries_over_discrete"] for row in reports) > 0
                else None
            ),
            "counting_unit": "planner_seed_by_fault_episode",
        },
        "cross_seed_episode_stability": {
            "recoveries": [
                {
                    "episode_id": episode_id,
                    "seed_count": len(seed_values),
                    "seeds": seed_values,
                }
                for episode_id, seed_values in sorted(
                    recovery_seeds.items(), key=lambda item: (-len(item[1]), item[0])
                )
            ],
            "regressions": [
                {
                    "episode_id": episode_id,
                    "seed_count": len(seed_values),
                    "seeds": seed_values,
                }
                for episode_id, seed_values in sorted(
                    regression_seeds.items(), key=lambda item: (-len(item[1]), item[0])
                )
            ],
            "added_saturation_over_discrete": [
                {
                    "episode_id": episode_id,
                    "seed_count": len(seed_values),
                    "seeds": seed_values,
                }
                for episode_id, seed_values in sorted(
                    added_saturation_seeds.items(),
                    key=lambda item: (-len(item[1]), item[0]),
                )
            ],
            "removed_saturation_over_discrete": [
                {
                    "episode_id": episode_id,
                    "seed_count": len(seed_values),
                    "seeds": seed_values,
                }
                for episode_id, seed_values in sorted(
                    removed_saturation_seeds.items(),
                    key=lambda item: (-len(item[1]), item[0]),
                )
            ],
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
