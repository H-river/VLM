#!/usr/bin/env python3
"""Trace a visible guarded-mean success/saturation frontier across planner seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.summarize_guarded_margin_seeds import (
    _jsonl,
    _key,
    _rate,
    _seed_summary,
    _use_continuous,
)


VERSION = "active_diagnosis_v13_guarded_margin_seed_frontier_v1"


def _policy_name(seed: int, base_seed: int, base: str, prefix: str) -> str:
    return base if seed == base_seed else f"{prefix}_{seed}"


def _is_dominated(cell: dict[str, Any], cells: list[dict[str, Any]]) -> bool:
    return any(
        float(other["control_value_over_direct"]["mean"])
        >= float(cell["control_value_over_direct"]["mean"])
        and float(other["fault_saturation_episode_rate"]["mean"])
        <= float(cell["fault_saturation_episode_rate"]["mean"])
        and (
            float(other["control_value_over_direct"]["mean"])
            > float(cell["control_value_over_direct"]["mean"])
            or float(other["fault_saturation_episode_rate"]["mean"])
            < float(cell["fault_saturation_episode_rate"]["mean"])
        )
        for other in cells
    )


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
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=(0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve() / "control"
    base_seed = int(json.loads(args.config.resolve().read_text())["root_seed"])
    seeds = list(dict.fromkeys(args.seeds))
    margins = sorted(set(float(value) for value in args.margins))
    if base_seed not in seeds or len(seeds) < 2:
        raise ValueError("at least two seeds including the configured base are required")
    if not margins or margins[0] < 0:
        raise ValueError("margins must be nonnegative")

    sources: dict[
        int,
        tuple[
            dict[tuple[str, float], dict[str, Any]],
            dict[tuple[str, float], dict[str, Any]],
            dict[tuple[str, float], dict[str, Any]],
        ],
    ] = {}
    for seed in seeds:
        direct_name = "direct" if seed == base_seed else f"direct_seed_{seed}"
        discrete_name = _policy_name(
            seed,
            base_seed,
            args.discrete_base_policy,
            args.discrete_seed_prefix,
        )
        continuous_name = _policy_name(
            seed,
            base_seed,
            args.continuous_base_policy,
            args.continuous_seed_prefix,
        )
        direct_rows = _jsonl(root / f"{direct_name}.jsonl")
        discrete_rows = _jsonl(root / f"{discrete_name}.jsonl")
        continuous_rows = _jsonl(root / f"{continuous_name}.jsonl")
        direct = {_key(row): row for row in direct_rows}
        discrete = {_key(row): row for row in discrete_rows}
        continuous = {_key(row): row for row in continuous_rows}
        if (
            len(direct_rows) != 150
            or len(discrete_rows) != 150
            or len(continuous_rows) != 150
            or len(direct) != 150
            or len(discrete) != 150
            or len(continuous) != 150
            or set(direct) != set(discrete)
            or set(direct) != set(continuous)
        ):
            raise ValueError(f"seed {seed} sources are incomplete, duplicated, or unmatched")
        if any(int(case_id.rsplit("_", 1)[1]) >= 10 for case_id, _ in direct):
            raise ValueError(f"protected case present for seed {seed}")
        sources[seed] = direct, discrete, continuous

    cells: list[dict[str, Any]] = []
    for margin_index, margin in enumerate(margins):
        seed_rows: list[dict[str, Any]] = []
        for seed in seeds:
            direct, discrete, continuous = sources[seed]
            fault_keys = [key for key in direct if key[1] != 1.0]
            selected = {
                key: (
                    continuous[key]
                    if _use_continuous(
                        discrete[key],
                        continuous[key],
                        discrete_threshold=args.discrete_threshold,
                        downward_margin=margin,
                    )
                    else discrete[key]
                )
                for key in direct
            }
            direct_fault = [direct[key] for key in fault_keys]
            discrete_fault = [discrete[key] for key in fault_keys]
            selected_fault = [selected[key] for key in fault_keys]
            seed_rows.append(
                {
                    "seed": seed,
                    "continuous_episode_count": int(
                        sum(selected[key] is continuous[key] for key in selected)
                    ),
                    "control_value_over_direct": _rate(selected_fault)
                    - _rate(direct_fault),
                    "gain_over_discrete": _rate(selected_fault)
                    - _rate(discrete_fault),
                    "fault_saturation_episode_rate": float(
                        np.mean(
                            [
                                int(row["saturation_count"]) > 0
                                for row in selected_fault
                            ]
                        )
                    ),
                }
            )
        control_values = np.asarray(
            [row["control_value_over_direct"] for row in seed_rows],
            dtype=np.float64,
        )
        gains = np.asarray(
            [row["gain_over_discrete"] for row in seed_rows], dtype=np.float64
        )
        saturations = np.asarray(
            [row["fault_saturation_episode_rate"] for row in seed_rows],
            dtype=np.float64,
        )
        cells.append(
            {
                "margin": margin,
                "seeds": seed_rows,
                "control_value_over_direct": {
                    **_seed_summary(control_values, 2026080160 + margin_index),
                    "passes_five_points_every_seed": bool(
                        np.all(control_values >= 0.05)
                    ),
                    "seed_count_at_least_five_points": int(
                        np.sum(control_values >= 0.05)
                    ),
                },
                "gain_over_discrete": _seed_summary(
                    gains, 2026080180 + margin_index
                ),
                "fault_saturation_episode_rate": {
                    "mean": float(saturations.mean()),
                    "minimum": float(saturations.min()),
                    "maximum": float(saturations.max()),
                },
            }
        )
    pareto_margins = [
        float(cell["margin"]) for cell in cells if not _is_dominated(cell, cells)
    ]
    report = {
        "version": VERSION,
        "role": "development_only_cross_seed_cached_policy_frontier_no_protected_reselection",
        "split": "development_only",
        "protected_set_used": False,
        "selection_inputs_are_visible_predictions_only": True,
        "discrete_threshold": args.discrete_threshold,
        "seeds": seeds,
        "cells": cells,
        "mean_success_saturation_pareto_margins": pareto_margins,
        "margins_passing_five_points_every_seed": [
            float(cell["margin"])
            for cell in cells
            if cell["control_value_over_direct"]["passes_five_points_every_seed"]
        ],
        "interpretation_constraint": (
            "cross-seed exploratory frontier only; it cannot alter the frozen or protected policy"
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
