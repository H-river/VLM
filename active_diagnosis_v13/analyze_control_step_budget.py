#!/usr/bin/env python3
"""Analyze matched four/six-step controls and a group-OOF continuation rule."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "active_diagnosis_v13_control_step_budget_analysis_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.resolve().read_text().splitlines() if line]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _key(row: Mapping[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _trace_projection(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "command_mm": step["command_mm"],
            "predicted_next_metrics": step["predicted_next_metrics"],
            "observed_next_metrics": step["observed_next_metrics"],
            "before_target_cost": step["before_target_cost"],
            "predicted_target_cost": step["predicted_target_cost"],
            "actual_target_cost": step["actual_target_cost"],
            "actual_step_audit": step["actual_step_audit"],
        }
        for step in row["trace"]
    ]


def _assert_replay(
    baseline: Mapping[tuple[str, float], Mapping[str, Any]],
    treatment: Mapping[tuple[str, float], Mapping[str, Any]],
) -> list[str]:
    mismatches = []
    for key, base in baseline.items():
        treatment_prefix = dict(treatment[key])
        treatment_prefix["trace"] = treatment_prefix["trace"][: len(base["trace"])]
        if _trace_projection(base) != _trace_projection(treatment_prefix):
            mismatches.append(f"{key[0]}__g{key[1]:g}")
    return mismatches


def _block(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "episodes": len(rows),
        "strict_success": float(np.mean([bool(row["strict_success"]) for row in rows])),
        "mean_final_distance": float(
            np.mean([float(row["final_normalized_distance"]) for row in rows])
        ),
        "median_final_distance": float(
            np.median([float(row["final_normalized_distance"]) for row in rows])
        ),
        "mean_control_steps": float(np.mean([int(row["control_steps"]) for row in rows])),
        "saturation_episode_rate": float(
            np.mean([int(row["saturation_count"]) > 0 for row in rows])
        ),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_gain: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_stratum: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    fault_by_stratum: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    fault = []
    for row in rows:
        by_gain[f"{float(row['evaluator_only_true_gain']):g}"].append(row)
        by_stratum[str(row["stratum"])].append(row)
        if float(row["evaluator_only_true_gain"]) != 1.0:
            fault.append(row)
            fault_by_stratum[str(row["stratum"])].append(row)
    return {
        "overall": _block(rows),
        "fault": _block(fault),
        "by_gain": {
            key: _block(value)
            for key, value in sorted(by_gain.items(), key=lambda item: float(item[0]))
        },
        "by_stratum": {
            key: _block(value) for key, value in sorted(by_stratum.items())
        },
        "fault_by_stratum": {
            key: _block(value) for key, value in sorted(fault_by_stratum.items())
        },
    }


def _bootstrap_difference(
    baseline: Mapping[tuple[str, float], Mapping[str, Any]],
    treatment: Mapping[tuple[str, float], Mapping[str, Any]],
    *,
    seed: int,
    fault_only: bool,
    samples: int = 4000,
) -> dict[str, float | int]:
    keys = [key for key in baseline if not fault_only or key[1] != 1.0]
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for key in keys:
        grouped[str(baseline[key]["group_id"])].append(
            float(treatment[key]["strict_success"])
            - float(baseline[key]["strict_success"])
        )
    values = np.asarray(
        [float(np.mean(grouped[group])) for group in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(values[rng.integers(0, len(values), len(values))]))
        for _ in range(samples)
    ]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(
            np.mean(
                [
                    float(treatment[key]["strict_success"])
                    - float(baseline[key]["strict_success"])
                    for key in keys
                ]
            )
        ),
        "group_mean_estimate": float(np.mean(values)),
        "low": float(low),
        "high": float(high),
        "independent_groups": len(values),
    }


def _switched_row(
    base: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    minimum_last_improvement: float,
    maximum_final_distance: float,
) -> dict[str, Any]:
    last_improvement = (
        0.0
        if not base["trace"]
        else float(base["trace"][-1]["before_target_cost"])
        - float(base["trace"][-1]["actual_target_cost"])
    )
    continue_control = bool(
        not bool(base["strict_success"])
        and int(base["control_steps"]) == 4
        and last_improvement >= minimum_last_improvement
        and float(base["final_normalized_distance"]) <= maximum_final_distance
    )
    source = treatment if continue_control else base
    row = dict(source)
    row["continuation_selected"] = continue_control
    row["continuation_visible_features"] = {
        "strict_success_after_four": bool(base["strict_success"]),
        "control_steps_after_four": int(base["control_steps"]),
        "last_step_actual_improvement": last_improvement,
        "final_normalized_distance": float(base["final_normalized_distance"]),
    }
    return row


def _candidate_rules() -> list[tuple[float, float]]:
    return [
        (minimum, maximum)
        for minimum in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
        for maximum in (2.0, 4.0, 8.0, 16.0, float("inf"))
    ]


def _rule_rows(
    baseline: Mapping[tuple[str, float], Mapping[str, Any]],
    treatment: Mapping[tuple[str, float], Mapping[str, Any]],
    rule: tuple[float, float],
    keys: Sequence[tuple[str, float]],
) -> dict[tuple[str, float], dict[str, Any]]:
    return {
        key: _switched_row(
            baseline[key],
            treatment[key],
            minimum_last_improvement=rule[0],
            maximum_final_distance=rule[1],
        )
        for key in keys
    }


def _rule_score(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    return (
        float(np.mean([bool(row["strict_success"]) for row in rows])),
        -float(np.mean([int(row["control_steps"]) for row in rows])),
        -float(np.mean([int(row["saturation_count"]) > 0 for row in rows])),
    )


def _rule_dict(rule: tuple[float, float]) -> dict[str, float | str]:
    return {
        "minimum_last_step_improvement": float(rule[0]),
        "maximum_final_distance": "infinity" if np.isinf(rule[1]) else float(rule[1]),
    }


def _group_oof_rule(
    baseline: Mapping[tuple[str, float], Mapping[str, Any]],
    treatment: Mapping[tuple[str, float], Mapping[str, Any]],
) -> tuple[dict[tuple[str, float], dict[str, Any]], dict[str, Any]]:
    groups = sorted({str(row["group_id"]) for row in baseline.values()})
    candidates = _candidate_rules()
    selected_rows: dict[tuple[str, float], dict[str, Any]] = {}
    selections = []
    for held_group in groups:
        training_keys = [
            key for key, row in baseline.items() if str(row["group_id"]) != held_group
        ]
        held_keys = [
            key for key, row in baseline.items() if str(row["group_id"]) == held_group
        ]
        scored = []
        for rule in candidates:
            rows = list(_rule_rows(baseline, treatment, rule, training_keys).values())
            scored.append((_rule_score(rows), rule))
        best_score, best_rule = max(scored, key=lambda item: (item[0], -item[1][0], -item[1][1]))
        selected_rows.update(_rule_rows(baseline, treatment, best_rule, held_keys))
        selections.append(
            {
                "held_out_group": held_group,
                "training_score": {
                    "strict_success": best_score[0],
                    "negative_mean_control_steps": best_score[1],
                    "negative_saturation_rate": best_score[2],
                },
                "selected_rule": _rule_dict(best_rule),
            }
        )
    if set(selected_rows) != set(baseline):
        raise ValueError("group-OOF continuation predictions are incomplete")
    frequencies = Counter(
        (
            row["selected_rule"]["minimum_last_step_improvement"],
            row["selected_rule"]["maximum_final_distance"],
        )
        for row in selections
    )
    return selected_rows, {
        "folds": len(groups),
        "selections": selections,
        "selected_rule_frequencies": [
            {
                "minimum_last_step_improvement": rule[0],
                "maximum_final_distance": rule[1],
                "folds": count,
            }
            for rule, count in sorted(frequencies.items(), key=lambda item: (-item[1], str(item[0])))
        ],
    }


def _arm_analysis(
    name: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    baseline = {_key(row): row for row in baseline_rows}
    treatment = {_key(row): row for row in treatment_rows}
    if len(baseline) != 150 or len(treatment) != 150 or set(baseline) != set(treatment):
        raise ValueError(f"{name} requires exactly 150 matched rows per budget")
    mismatches = _assert_replay(baseline, treatment)
    if mismatches:
        raise ValueError(f"{name} first-four replay mismatch: {mismatches[:3]}")
    oof_rows, oof_selection = _group_oof_rule(baseline, treatment)
    full_frontier = []
    for rule in _candidate_rules():
        rows = _rule_rows(baseline, treatment, rule, sorted(baseline))
        values = list(rows.values())
        full_frontier.append(
            {
                "rule": _rule_dict(rule),
                "strict_success": float(
                    np.mean([bool(row["strict_success"]) for row in values])
                ),
                "fault_success": float(
                    np.mean(
                        [
                            bool(row["strict_success"])
                            for key, row in rows.items()
                            if key[1] != 1.0
                        ]
                    )
                ),
                "mean_control_steps": float(
                    np.mean([int(row["control_steps"]) for row in values])
                ),
                "saturation_episode_rate": float(
                    np.mean([int(row["saturation_count"]) > 0 for row in values])
                ),
                "selected_continuations": sum(
                    bool(row["continuation_selected"]) for row in values
                ),
            }
        )
    full_best = max(
        full_frontier,
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
    recoveries = [
        f"{key[0]}__g{key[1]:g}"
        for key in sorted(baseline)
        if not bool(baseline[key]["strict_success"])
        and bool(treatment[key]["strict_success"])
    ]
    regressions = [
        f"{key[0]}__g{key[1]:g}"
        for key in sorted(baseline)
        if bool(baseline[key]["strict_success"])
        and not bool(treatment[key]["strict_success"])
    ]
    selected_continuations = [
        f"{key[0]}__g{key[1]:g}"
        for key in sorted(oof_rows)
        if bool(oof_rows[key]["continuation_selected"])
    ]
    return {
        "arm": name,
        "first_four_trace_exact_replays": len(baseline),
        "baseline_budget4": _summarize(list(baseline.values())),
        "fixed_budget6": _summarize(list(treatment.values())),
        "fixed_budget6_over_budget4_overall": _bootstrap_difference(
            baseline, treatment, seed=seed, fault_only=False
        ),
        "fixed_budget6_over_budget4_fault": _bootstrap_difference(
            baseline, treatment, seed=seed + 1, fault_only=True
        ),
        "fixed_budget6_recoveries": recoveries,
        "fixed_budget6_regressions": regressions,
        "group_oof_adaptive_budget": {
            "summary": _summarize(list(oof_rows.values())),
            "difference_over_budget4_overall": _bootstrap_difference(
                baseline, oof_rows, seed=seed + 2, fault_only=False
            ),
            "difference_over_budget4_fault": _bootstrap_difference(
                baseline, oof_rows, seed=seed + 3, fault_only=True
            ),
            "selected_continuations": len(selected_continuations),
            "selected_continuation_episode_ids": selected_continuations,
            "selection": oof_selection,
            "features_are_policy_visible": True,
            "visible_features": [
                "observed strict-success state after four controls",
                "observed fourth-step normalized-distance improvement",
                "observed normalized target distance after four controls",
            ],
        },
        "exploratory_full_development_rule": {
            "selection_order": (
                "maximize all-episode strict success, then minimize mean control steps, "
                "then saturation; development-only and not an OOF estimate"
            ),
            "selected": full_best,
            "candidate_frontier": full_frontier,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("direct", "oracle", "probe"):
        parser.add_argument(f"--{arm}-budget4", type=Path, required=True)
        parser.add_argument(f"--{arm}-budget6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "arms": {},
        "interpretation_guard": (
            "The fixed-six ablation is matched causal development evidence. The adaptive "
            "rule is group-out-of-fold cached policy switching and requires a fresh "
            "preregistered study before deployment or protected evaluation."
        ),
    }
    for offset, arm in enumerate(("direct", "oracle", "probe")):
        report["arms"][arm] = _arm_analysis(
            arm,
            _read_jsonl(getattr(args, f"{arm}_budget4")),
            _read_jsonl(getattr(args, f"{arm}_budget6")),
            2026080160 + offset * 10,
        )
    _atomic_json(args.output, report)
    compact = {
        arm: {
            "fixed_fault_difference": row["fixed_budget6_over_budget4_fault"],
            "adaptive_fault_difference": row["group_oof_adaptive_budget"][
                "difference_over_budget4_fault"
            ],
            "fixed_recoveries": len(row["fixed_budget6_recoveries"]),
            "fixed_regressions": len(row["fixed_budget6_regressions"]),
            "adaptive_continuations": row["group_oof_adaptive_budget"][
                "selected_continuations"
            ],
        }
        for arm, row in report["arms"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
