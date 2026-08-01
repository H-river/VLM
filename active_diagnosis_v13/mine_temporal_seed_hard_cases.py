#!/usr/bin/env python3
"""Serialize cross-seed temporal recoveries and frozen-rule misses."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from active_diagnosis_v13.analyze_control_step_budget import (
    _assert_replay,
    _key,
    _read_jsonl,
    _switched_row,
)
from active_diagnosis_v13.analyze_temporal_seed_statistics import _paths

VERSION = "active_diagnosis_v13_temporal_seed_hard_cases_v2"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _episode_id(key: tuple[str, float]) -> str:
    return f"{key[0]}__g{key[1]:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-report", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_report = json.loads(args.seed_report.resolve().read_text())
    if seed_report.get("protected_set_used") is not False:
        raise ValueError("seed report must be development-only")
    seeds = [int(row["seed"]) for row in seed_report["seeds"]]
    rule = seed_report["frozen_seed1_visible_rule"]
    maximum_raw = rule["maximum_final_distance"]
    maximum = float("inf") if maximum_raw == "infinity" else float(maximum_raw)
    control = args.output_dir.resolve() / "control"
    records = []
    for seed in seeds:
        baseline_path, treatment_path = _paths(control, seed)
        baseline = {_key(row): row for row in _read_jsonl(baseline_path)}
        treatment = {_key(row): row for row in _read_jsonl(treatment_path)}
        if len(baseline) != 150 or set(baseline) != set(treatment):
            raise ValueError(f"seed {seed} is not a complete matched arm")
        mismatches = _assert_replay(baseline, treatment)
        if mismatches:
            raise ValueError(f"seed {seed} prefix mismatch: {mismatches[:3]}")
        for key in sorted(baseline):
            if key[1] == 1.0:
                continue
            base = baseline[key]
            fixed = treatment[key]
            adaptive = _switched_row(
                base,
                fixed,
                minimum_last_improvement=float(rule["minimum_last_step_improvement"]),
                maximum_final_distance=maximum,
            )
            fixed_recovery = bool(
                not bool(base["strict_success"]) and bool(fixed["strict_success"])
            )
            recovery_step = None
            if fixed_recovery:
                recovery_step = next(
                    index + 1
                    for index, step in enumerate(fixed["trace"])
                    if float(step["actual_target_cost"]) <= 1.0
                )
            last_improvement = (
                0.0
                if not base["trace"]
                else float(base["trace"][-1]["before_target_cost"])
                - float(base["trace"][-1]["actual_target_cost"])
            )
            records.append(
                {
                    "record_id": f"{_episode_id(key)}__seed_{seed}",
                    "episode_id": _episode_id(key),
                    "case_id": key[0],
                    "group_id": str(base["group_id"]),
                    "stratum": str(base["stratum"]),
                    "evaluator_only_true_gain": key[1],
                    "planner_seed": seed,
                    "baseline_strict_success": bool(base["strict_success"]),
                    "baseline_final_distance": float(base["final_normalized_distance"]),
                    "fourth_step_last_improvement": last_improvement,
                    "rule_continuation_selected": bool(
                        adaptive["continuation_selected"]
                    ),
                    "fixed6_strict_success": bool(fixed["strict_success"]),
                    "fixed6_final_distance": float(fixed["final_normalized_distance"]),
                    "fixed6_recovery": fixed_recovery,
                    "fixed6_recovery_step": recovery_step,
                    "adaptive_recovery": bool(
                        not bool(base["strict_success"])
                        and bool(adaptive["strict_success"])
                    ),
                    "fixed6_saturation_count": int(fixed["saturation_count"]),
                    "adaptive_saturation_count": int(adaptive["saturation_count"]),
                }
            )
    by_seed = {}
    for seed in seeds:
        rows = [row for row in records if row["planner_seed"] == seed]
        failures = [row for row in rows if not row["baseline_strict_success"]]
        by_seed[str(seed)] = {
            "fault_episodes": len(rows),
            "baseline_failures": len(failures),
            "fixed6_recoveries": sum(row["fixed6_recovery"] for row in failures),
            "adaptive_recoveries": sum(row["adaptive_recovery"] for row in failures),
            "selected_continuations": sum(
                row["rule_continuation_selected"] for row in failures
            ),
            "recovery_step_counts": {
                str(step): count
                for step, count in sorted(
                    Counter(
                        row["fixed6_recovery_step"]
                        for row in failures
                        if row["fixed6_recovery_step"] is not None
                    ).items()
                )
            },
            "missed_recovery_ids": sorted(
                row["episode_id"]
                for row in failures
                if row["fixed6_recovery"] and not row["adaptive_recovery"]
            ),
        }
    episode_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        episode_rows[row["episode_id"]].append(row)
    cross_seed = []
    for episode_id, rows in sorted(episode_rows.items()):
        failure_seeds = sorted(
            row["planner_seed"] for row in rows if not row["baseline_strict_success"]
        )
        fixed_seeds = sorted(
            row["planner_seed"] for row in rows if row["fixed6_recovery"]
        )
        adaptive_seeds = sorted(
            row["planner_seed"] for row in rows if row["adaptive_recovery"]
        )
        cross_seed.append(
            {
                "episode_id": episode_id,
                "baseline_failure_seeds": failure_seeds,
                "fixed6_recovery_seeds": fixed_seeds,
                "adaptive_recovery_seeds": adaptive_seeds,
                "baseline_fails_all_seeds": len(failure_seeds) == len(seeds),
                "fixed6_recovers_all_failure_seeds": bool(failure_seeds)
                and fixed_seeds == failure_seeds,
                "adaptive_recovers_all_failure_seeds": bool(failure_seeds)
                and adaptive_seeds == failure_seeds,
            }
        )
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "frozen_seed1_visible_rule": rule,
        "planner_seeds": seeds,
        "serialized_fault_seed_records": len(records),
        "by_seed": by_seed,
        "cross_seed_episode_summary": cross_seed,
        "stable_exact_ids": {
            "baseline_fails_all_observed_seeds": sorted(
                row["episode_id"] for row in cross_seed if row["baseline_fails_all_seeds"]
            ),
            "fixed6_recovers_every_failure_seed": sorted(
                row["episode_id"]
                for row in cross_seed
                if row["fixed6_recovers_all_failure_seeds"]
            ),
            "adaptive_recovers_every_failure_seed": sorted(
                row["episode_id"]
                for row in cross_seed
                if row["adaptive_recovers_all_failure_seeds"]
            ),
            "fixed6_never_recovers_any_failure_seed": sorted(
                row["episode_id"]
                for row in cross_seed
                if row["baseline_failure_seeds"] and not row["fixed6_recovery_seeds"]
            ),
        },
        "interpretation_guard": (
            "The rule was selected on seed 1 and held fixed on every additional planner "
            f"seed. Cross-seed stability reflects {len(seeds)} planner samples only; setup "
            "groups are unchanged and no protected data are used."
        ),
    }
    _atomic_jsonl(args.records.resolve(), records)
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
