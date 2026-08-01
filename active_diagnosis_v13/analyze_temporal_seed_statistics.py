#!/usr/bin/env python3
"""Audit a frozen temporal continuation rule across planner seeds."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.analyze_control_step_budget import (
    _key,
    _read_jsonl,
    _switched_row,
)

VERSION = "active_diagnosis_v13_temporal_seed_statistics_v2"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _two_way_bootstrap(
    values: np.ndarray, *, seed: int, samples: int = 10000
) -> dict[str, float | int]:
    """Resample planner seeds and setup groups independently."""
    if values.ndim != 2 or not values.size:
        raise ValueError("values must be a nonempty seed-by-group matrix")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        seed_indices = rng.integers(0, values.shape[0], values.shape[0])
        group_indices = rng.integers(0, values.shape[1], values.shape[1])
        draws[index] = float(np.mean(values[np.ix_(seed_indices, group_indices)]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(np.mean(values)),
        "low": float(low),
        "high": float(high),
        "planner_seeds": int(values.shape[0]),
        "setup_groups": int(values.shape[1]),
        "samples": int(samples),
    }


def _paths(control: Path, seed: int) -> tuple[Path, Path]:
    if seed == 2026080101:
        return (
            control / "probe_symmetric_pair_f0p1_no_residual.jsonl",
            control / "probe_nores_budget6.jsonl",
        )
    return (
        control / f"probe_nores_seed_{seed}.jsonl",
        control / f"probe_nores_budget6_seed_{seed}.jsonl",
    )


def _seed_rows(
    *, control: Path, seed: int, rule: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    baseline_path, treatment_path = _paths(control, seed)
    baseline = {_key(row): row for row in _read_jsonl(baseline_path)}
    treatment = {_key(row): row for row in _read_jsonl(treatment_path)}
    if len(baseline) != 150 or set(baseline) != set(treatment):
        raise ValueError(f"seed {seed} does not contain 150 matched probe rows")
    maximum_raw = rule["maximum_final_distance"]
    maximum = float("inf") if maximum_raw == "infinity" else float(maximum_raw)
    adaptive = {
        key: _switched_row(
            baseline[key],
            treatment[key],
            minimum_last_improvement=float(rule["minimum_last_step_improvement"]),
            maximum_final_distance=maximum,
        )
        for key in baseline
    }
    fault_keys = [key for key in baseline if key[1] != 1.0]
    fixed_recovery_keys = {
        key
        for key in fault_keys
        if not bool(baseline[key]["strict_success"])
        and bool(treatment[key]["strict_success"])
    }
    adaptive_recovery_keys = {
        key
        for key in fault_keys
        if not bool(baseline[key]["strict_success"])
        and bool(adaptive[key]["strict_success"])
    }
    selected_keys = {
        key for key in fault_keys if bool(adaptive[key]["continuation_selected"])
    }
    grouped: defaultdict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"fixed": [], "adaptive": []}
    )
    for key in fault_keys:
        group = str(baseline[key]["group_id"])
        grouped[group]["fixed"].append(
            float(treatment[key]["strict_success"])
            - float(baseline[key]["strict_success"])
        )
        grouped[group]["adaptive"].append(
            float(adaptive[key]["strict_success"])
            - float(baseline[key]["strict_success"])
        )
    group_values = {
        group: {
            name: float(np.mean(values)) for name, values in blocks.items()
        }
        for group, blocks in grouped.items()
    }
    summary = {
        "seed": seed,
        "fault_episodes": len(fault_keys),
        "fixed6_fault_difference": float(
            np.mean(
                [
                    float(treatment[key]["strict_success"])
                    - float(baseline[key]["strict_success"])
                    for key in fault_keys
                ]
            )
        ),
        "adaptive_fault_difference": float(
            np.mean(
                [
                    float(adaptive[key]["strict_success"])
                    - float(baseline[key]["strict_success"])
                    for key in fault_keys
                ]
            )
        ),
        "fixed6_recoveries": len(fixed_recovery_keys),
        "adaptive_recoveries": len(adaptive_recovery_keys),
        "fixed6_regressions": sum(
            bool(baseline[key]["strict_success"])
            and not bool(treatment[key]["strict_success"])
            for key in fault_keys
        ),
        "adaptive_regressions": sum(
            bool(baseline[key]["strict_success"])
            and not bool(adaptive[key]["strict_success"])
            for key in fault_keys
        ),
        "selected_fault_continuations": len(selected_keys),
        "recoveries_retained_by_rule": len(fixed_recovery_keys & adaptive_recovery_keys),
        "recoveries_missed_by_rule": len(fixed_recovery_keys - adaptive_recovery_keys),
        "selected_without_recovery": len(selected_keys - fixed_recovery_keys),
        "mean_steps_saved_vs_fixed6": float(
            np.mean(
                [
                    int(treatment[key]["control_steps"])
                    - int(adaptive[key]["control_steps"])
                    for key in baseline
                ]
            )
        ),
    }
    return summary, group_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_report = json.loads(args.seed_report.resolve().read_text())
    if seed_report.get("protected_set_used") is not False:
        raise ValueError("seed report must be development-only")
    rule = seed_report["frozen_seed1_visible_rule"]
    seeds = [int(row["seed"]) for row in seed_report["seeds"]]
    summaries = []
    group_blocks = {}
    control = args.output_dir.resolve() / "control"
    for seed in seeds:
        summary, groups = _seed_rows(control=control, seed=seed, rule=rule)
        summaries.append(summary)
        group_blocks[seed] = groups
    groups = sorted(set.intersection(*(set(group_blocks[seed]) for seed in seeds)))
    if any(set(group_blocks[seed]) != set(groups) for seed in seeds):
        raise ValueError("planner seeds do not share identical setup groups")
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "frozen_seed1_visible_rule": rule,
        "per_seed": summaries,
        "all_observed_seeds_positive": {
            "fixed6": all(row["fixed6_fault_difference"] > 0.0 for row in summaries),
            "adaptive": all(row["adaptive_fault_difference"] > 0.0 for row in summaries),
        },
        "two_way_seed_group_bootstrap": {
            name: _two_way_bootstrap(
                np.asarray(
                    [
                        [group_blocks[seed][group][name] for group in groups]
                        for seed in seeds
                    ],
                    dtype=np.float64,
                ),
                seed=2026080171 + offset,
            )
            for offset, name in enumerate(("fixed", "adaptive"))
        },
        "interpretation_guard": (
            "The visible rule was selected on seed 1 and applied unchanged to every "
            f"additional planner seed. The two-way interval resamples all {len(seeds)} "
            f"planner seeds and {len(groups)} matched setup groups independently; no "
            "protected data are used."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
