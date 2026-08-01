#!/usr/bin/env python3
"""Compare setup difficulty while applying one frozen sequential rule."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.analyze_control_horizon_curve import _cap
from active_diagnosis_v13.analyze_control_step_budget import _read_jsonl
from active_diagnosis_v13.analyze_sequential_horizon_rule import _sequential_cap

VERSION = "active_diagnosis_v13_setup_cohort_shift_v1"


def _parse_cohort(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("cohort must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("cohort must be LABEL=PATH")
    return label, Path(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _summary(path: Path, *, minimum: float, maximum: float) -> tuple[dict[str, Any], set[str]]:
    source = _read_jsonl(path.resolve())
    if len(source) != 150:
        raise ValueError(f"{path} does not contain 150 episodes")
    seeds = {int(row["planner_root_seed"]) for row in source}
    if len(seeds) != 1:
        raise ValueError(f"{path} mixes planner root seeds")
    groups = {str(row["group_id"]) for row in source}
    if len(groups) != 30:
        raise ValueError(f"{path} does not contain 30 setup groups")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        by_group[str(row["group_id"])].append(row)
    group_rows = []
    for group, rows in sorted(by_group.items()):
        distances = {float(row["initial_normalized_distance"]) for row in rows}
        strata = {str(row["stratum"]) for row in rows}
        if len(rows) != 5 or len(distances) != 1 or len(strata) != 1:
            raise ValueError(f"group {group} is not a consistent five-gain setup")
        group_rows.append(
            {
                "group_id": group,
                "stratum": next(iter(strata)),
                "initial_normalized_distance": next(iter(distances)),
            }
        )
    fixed4 = [_cap(dict(row), 4) for row in source]
    sequential = [
        _sequential_cap(
            row, minimum_improvement=minimum, maximum_distance=maximum
        )
        for row in source
    ]
    strata = sorted({str(row["stratum"]) for row in source})
    fault_fixed = [
        row for row in fixed4 if float(row["evaluator_only_true_gain"]) != 1.0
    ]
    fault_sequential = [
        row for row in sequential if float(row["evaluator_only_true_gain"]) != 1.0
    ]
    distance_by_stratum = {}
    policy_by_stratum = {}
    for stratum in strata:
        distances = np.asarray(
            [
                row["initial_normalized_distance"]
                for row in group_rows
                if row["stratum"] == stratum
            ],
            dtype=np.float64,
        )
        distance_by_stratum[stratum] = {
            "groups": int(len(distances)),
            "minimum": float(np.min(distances)),
            "median": float(np.median(distances)),
            "maximum": float(np.max(distances)),
        }
        fixed_rows = [row for row in fault_fixed if str(row["stratum"]) == stratum]
        sequential_rows = [
            row for row in fault_sequential if str(row["stratum"]) == stratum
        ]
        policy_by_stratum[stratum] = {
            "fault_episodes": len(fixed_rows),
            "fixed4_success": _rate(fixed_rows),
            "frozen_sequential_success": _rate(sequential_rows),
            "frozen_sequential_gain_over_fixed4": _rate(sequential_rows)
            - _rate(fixed_rows),
        }
    distances = np.asarray(
        [row["initial_normalized_distance"] for row in group_rows], dtype=np.float64
    )
    gain_accuracy = float(
        np.mean([bool(row["gain_classification_correct"]) for row in source])
    )
    return (
        {
            "source": str(path.resolve()),
            "planner_root_seed": next(iter(seeds)),
            "setup_groups": len(groups),
            "stratum_group_counts": dict(Counter(row["stratum"] for row in group_rows)),
            "initial_distance": {
                "minimum": float(np.min(distances)),
                "median": float(np.median(distances)),
                "maximum": float(np.max(distances)),
            },
            "initial_distance_by_stratum": distance_by_stratum,
            "fault": {
                "episodes": len(fault_fixed),
                "fixed4_success": _rate(fault_fixed),
                "frozen_sequential_success": _rate(fault_sequential),
                "frozen_sequential_gain_over_fixed4": _rate(fault_sequential)
                - _rate(fault_fixed),
            },
            "fault_by_stratum": policy_by_stratum,
            "probe_gain_accuracy": gain_accuracy,
        },
        groups,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", type=_parse_cohort, required=True)
    parser.add_argument("--rule-report", type=Path, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.cohort) < 2:
        raise ValueError("at least two setup cohorts are required")
    rule_report = json.loads(args.rule_report.resolve().read_text())
    if rule_report.get("protected_set_used") is not False:
        raise ValueError("rule report is not development-only")
    rule = rule_report["frozen_full_selection_seed_rule"]
    minimum = float(rule["minimum_last_step_improvement"])
    maximum_raw = rule["maximum_final_distance"]
    maximum = float("inf") if maximum_raw == "infinity" else float(maximum_raw)
    cohorts = {}
    group_sets = {}
    for label, path in args.cohort:
        if label in cohorts:
            raise ValueError(f"duplicate cohort label {label}")
        cohorts[label], group_sets[label] = _summary(
            path, minimum=minimum, maximum=maximum
        )
    if args.reference_label not in cohorts:
        raise ValueError("reference label is not one of the supplied cohorts")
    labels = sorted(cohorts)
    overlaps = {}
    for index, label in enumerate(labels):
        for other in labels[index + 1 :]:
            overlaps[f"{label}__{other}"] = len(group_sets[label] & group_sets[other])
    reference = cohorts[args.reference_label]
    comparisons = []
    for label in labels:
        if label == args.reference_label:
            continue
        cohort = cohorts[label]
        comparisons.append(
            {
                "cohort": label,
                "reference": args.reference_label,
                "fixed4_fault_success_difference": float(
                    cohort["fault"]["fixed4_success"]
                    - reference["fault"]["fixed4_success"]
                ),
                "frozen_sequential_gain_difference": float(
                    cohort["fault"]["frozen_sequential_gain_over_fixed4"]
                    - reference["fault"]["frozen_sequential_gain_over_fixed4"]
                ),
                "median_initial_distance_difference": float(
                    cohort["initial_distance"]["median"]
                    - reference["initial_distance"]["median"]
                ),
            }
        )
    report = {
        "version": VERSION,
        "split": "posthoc_descriptive_nonprotected_setup_cohort_shift",
        "protected_set_used": False,
        "selection_or_retuning_on_cohorts": False,
        "frozen_rule": rule,
        "reference_label": args.reference_label,
        "cohorts": cohorts,
        "pairwise_group_id_overlaps": overlaps,
        "comparisons_to_reference": comparisons,
        "interpretation_guard": (
            "This audit quantifies cohort difficulty after all seed-1 outcomes were known. "
            "It is descriptive, uses one unchanged rule, and cannot select setups, seeds, "
            "thresholds, models, or protected evidence."
        ),
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
