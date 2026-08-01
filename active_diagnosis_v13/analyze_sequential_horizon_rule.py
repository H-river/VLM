#!/usr/bin/env python3
"""Select one visible sequential stopping rule on seed 1 and hold it across seeds."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from active_diagnosis_v13.analyze_control_horizon_curve import _cap, _summary
from active_diagnosis_v13.analyze_control_step_budget import (
    _candidate_rules,
    _key,
    _read_jsonl,
    _rule_dict,
)
from active_diagnosis_v13.analyze_temporal_seed_statistics import _two_way_bootstrap

VERSION = "active_diagnosis_v13_sequential_horizon_rule_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_seed(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("seed arm must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("seed arm must be LABEL=PATH")
    return label, Path(path)


def _sequential_cap(
    row: Mapping[str, Any], *, minimum_improvement: float, maximum_distance: float
) -> dict[str, Any]:
    full_trace = row["trace"]
    horizon = min(4, len(full_trace))
    decisions = []
    while horizon < min(8, len(full_trace)):
        step = full_trace[horizon - 1]
        distance = float(step["actual_target_cost"])
        improvement = float(step["before_target_cost"]) - distance
        continue_control = bool(
            distance > 1.0
            and improvement >= minimum_improvement
            and distance <= maximum_distance
        )
        decisions.append(
            {
                "after_control_step": horizon,
                "observed_distance": distance,
                "observed_last_step_improvement": improvement,
                "continued": continue_control,
            }
        )
        if not continue_control:
            break
        horizon += 1
    capped = _cap(dict(row), horizon)
    capped["sequential_rule_decisions"] = decisions
    capped["selected_additional_controls_after_four"] = max(0, horizon - 4)
    capped["stopped_by_visible_rule"] = bool(decisions and not decisions[-1]["continued"])
    return capped


def _rule_rows(
    rows: Mapping[tuple[str, float], Mapping[str, Any]],
    rule: tuple[float, float],
    keys: Sequence[tuple[str, float]],
) -> dict[tuple[str, float], dict[str, Any]]:
    return {
        key: _sequential_cap(
            rows[key], minimum_improvement=rule[0], maximum_distance=rule[1]
        )
        for key in keys
    }


def _score(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    return (
        float(np.mean([bool(row["strict_success"]) for row in rows])),
        -float(np.mean([int(row["control_steps"]) for row in rows])),
        -float(np.mean([int(row["saturation_count"]) > 0 for row in rows])),
    )


def _select_group_oof(
    rows: Mapping[tuple[str, float], Mapping[str, Any]]
) -> tuple[dict[tuple[str, float], dict[str, Any]], dict[str, Any]]:
    groups = sorted({str(row["group_id"]) for row in rows.values()})
    selected = {}
    folds = []
    for group in groups:
        training = [key for key, row in rows.items() if str(row["group_id"]) != group]
        held = [key for key, row in rows.items() if str(row["group_id"]) == group]
        scored = [
            (_score(list(_rule_rows(rows, rule, training).values())), rule)
            for rule in _candidate_rules()
        ]
        score, rule = max(
            scored,
            key=lambda item: (item[0], -item[1][0], -item[1][1]),
        )
        selected.update(_rule_rows(rows, rule, held))
        folds.append(
            {
                "held_out_group": group,
                "selected_rule": _rule_dict(rule),
                "training_score": {
                    "strict_success": score[0],
                    "negative_mean_control_steps": score[1],
                    "negative_saturation_rate": score[2],
                },
            }
        )
    frequencies = Counter(
        (
            fold["selected_rule"]["minimum_last_step_improvement"],
            fold["selected_rule"]["maximum_final_distance"],
        )
        for fold in folds
    )
    return selected, {
        "folds": folds,
        "rule_frequencies": [
            {
                "rule": {
                    "minimum_last_step_improvement": minimum,
                    "maximum_final_distance": maximum,
                },
                "fold_count": count,
            }
            for (minimum, maximum), count in sorted(
                frequencies.items(), key=lambda item: (-item[1], str(item[0]))
            )
        ],
    }


def _select_full_rule(
    rows: Mapping[tuple[str, float], Mapping[str, Any]]
) -> tuple[tuple[float, float], list[dict[str, Any]]]:
    frontier = []
    for rule in _candidate_rules():
        values = list(_rule_rows(rows, rule, sorted(rows)).values())
        score = _score(values)
        frontier.append(
            {
                "rule": _rule_dict(rule),
                "strict_success": score[0],
                "mean_control_steps": -score[1],
                "saturation_episode_rate": -score[2],
            }
        )
    selected = max(
        frontier,
        key=lambda row: (
            float(row["strict_success"]),
            -float(row["mean_control_steps"]),
            -float(row["saturation_episode_rate"]),
            -float(row["rule"]["minimum_last_step_improvement"]),
            -(
                float("inf")
                if row["rule"]["maximum_final_distance"] == "infinity"
                else float(row["rule"]["maximum_final_distance"])
            ),
        ),
    )
    maximum_raw = selected["rule"]["maximum_final_distance"]
    return (
        (
            float(selected["rule"]["minimum_last_step_improvement"]),
            float("inf") if maximum_raw == "infinity" else float(maximum_raw),
        ),
        frontier,
    )


def _seed_summary(
    label: str,
    source: Mapping[tuple[str, float], Mapping[str, Any]],
    adaptive: Mapping[tuple[str, float], Mapping[str, Any]],
) -> dict[str, Any]:
    fixed = {
        horizon: {key: _cap(dict(row), horizon) for key, row in source.items()}
        for horizon in (4, 6, 8)
    }
    fault_keys = [key for key in source if key[1] != 1.0]
    boundary_keys = [
        key
        for key in fault_keys
        if str(source[key]["stratum"]) == "reachable_boundary_or_clipping"
    ]
    rate = lambda values, keys: float(
        np.mean([bool(values[key]["strict_success"]) for key in keys])
    )
    fixed8_recoveries = [
        f"{key[0]}__g{key[1]:g}"
        for key in sorted(fault_keys)
        if not bool(fixed[4][key]["strict_success"])
        and bool(fixed[8][key]["strict_success"])
    ]
    sequential_recoveries = [
        f"{key[0]}__g{key[1]:g}"
        for key in sorted(fault_keys)
        if not bool(fixed[4][key]["strict_success"])
        and bool(adaptive[key]["strict_success"])
    ]
    return {
        "seed": label,
        "fault_success": {
            "fixed4": rate(fixed[4], fault_keys),
            "fixed6": rate(fixed[6], fault_keys),
            "fixed8": rate(fixed[8], fault_keys),
            "sequential_rule": rate(adaptive, fault_keys),
        },
        "boundary_fault_success": {
            "fixed4": rate(fixed[4], boundary_keys),
            "fixed6": rate(fixed[6], boundary_keys),
            "fixed8": rate(fixed[8], boundary_keys),
            "sequential_rule": rate(adaptive, boundary_keys),
        },
        "sequential_rule_gain_over_fixed4": rate(adaptive, fault_keys)
        - rate(fixed[4], fault_keys),
        "mean_control_steps": float(
            np.mean([int(row["control_steps"]) for row in adaptive.values()])
        ),
        "fault_saturation_episode_rate": float(
            np.mean([int(adaptive[key]["saturation_count"]) > 0 for key in fault_keys])
        ),
        "additional_controls_after_four": int(
            sum(
                int(row["selected_additional_controls_after_four"])
                for row in adaptive.values()
            )
        ),
        "recoveries": sum(
            not bool(fixed[4][key]["strict_success"])
            and bool(adaptive[key]["strict_success"])
            for key in fault_keys
        ),
        "regressions": sum(
            bool(fixed[4][key]["strict_success"])
            and not bool(adaptive[key]["strict_success"])
            for key in fault_keys
        ),
        "fixed8_recovery_episode_ids": fixed8_recoveries,
        "sequential_recovery_episode_ids": sequential_recoveries,
        "fixed8_recoveries_missed_by_sequential_rule": sorted(
            set(fixed8_recoveries) - set(sequential_recoveries)
        ),
        "summary": _summary(list(adaptive.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=_parse_seed, required=True)
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        label: {_key(row): row for row in _read_jsonl(path)}
        for label, path in args.seed
    }
    if len(sources) != len(args.seed) or args.selection_seed not in sources:
        raise ValueError("seed labels must be unique and include the selection seed")
    for label, rows in sources.items():
        if len(rows) != 150:
            raise ValueError(f"{label} does not contain 150 budget-eight rows")
    selection_rows = sources[args.selection_seed]
    oof_rows, oof_selection = _select_group_oof(selection_rows)
    frozen_rule, frontier = _select_full_rule(selection_rows)
    adaptive = {
        label: _rule_rows(rows, frozen_rule, sorted(rows))
        for label, rows in sources.items()
    }
    seed_summaries = [
        _seed_summary(label, sources[label], adaptive[label]) for label in sorted(sources)
    ]
    groups = sorted({str(row["group_id"]) for row in selection_rows.values()})
    matrices = []
    for label in sorted(sources):
        fixed4 = {key: _cap(dict(row), 4) for key, row in sources[label].items()}
        grouped: defaultdict[str, list[float]] = defaultdict(list)
        for key in sources[label]:
            if key[1] != 1.0:
                grouped[str(sources[label][key]["group_id"])].append(
                    float(adaptive[label][key]["strict_success"])
                    - float(fixed4[key]["strict_success"])
                )
        if set(grouped) != set(groups):
            raise ValueError(f"{label} setup groups differ from the selection seed")
        matrices.append([float(np.mean(grouped[group])) for group in groups])
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "selection_seed": args.selection_seed,
        "visible_features": [
            "observed strict-success state after each control",
            "observed normalized target distance after each control",
            "observed last-step normalized-distance improvement",
        ],
        "rule_application": "apply the same visible rule after steps four through seven",
        "seed1_group_oof": {
            "summary": _summary(list(oof_rows.values())),
            "selection": oof_selection,
        },
        "frozen_full_selection_seed_rule": _rule_dict(frozen_rule),
        "selection_seed_frontier": frontier,
        "seeds": seed_summaries,
        "all_observed_seeds_positive_over_fixed4": all(
            row["sequential_rule_gain_over_fixed4"] > 0.0 for row in seed_summaries
        ),
        "two_way_seed_group_bootstrap_gain_over_fixed4": _two_way_bootstrap(
            np.asarray(matrices, dtype=np.float64), seed=2026080121
        ),
        "interpretation_guard": (
            "The rule is selected only on seed-1 development groups and applied unchanged "
            "to other planner seeds. Setup groups are shared, so independent new-setup "
            "validation remains required; no protected data are used."
        ),
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
