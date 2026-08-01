#!/usr/bin/env python3
"""Serialize budget-four failures, six-step outcomes, and OOF continuation decisions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.analyze_control_step_budget import _key, _read_jsonl

VERSION = "active_diagnosis_v13_temporal_control_case_mining_v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _rule_for_group(arm_analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["held_out_group"]): row["selected_rule"]
        for row in arm_analysis["group_oof_adaptive_budget"]["selection"]["selections"]
    }


def _selected(base: dict[str, Any], rule: dict[str, Any]) -> bool:
    maximum = rule["maximum_final_distance"]
    maximum_value = float("inf") if maximum == "infinity" else float(maximum)
    last = base["trace"][-1]
    improvement = float(last["before_target_cost"]) - float(last["actual_target_cost"])
    return bool(
        not bool(base["strict_success"])
        and int(base["control_steps"]) == 4
        and improvement >= float(rule["minimum_last_step_improvement"])
        and float(base["final_normalized_distance"]) <= maximum_value
    )


def _category(base: dict[str, Any], treatment: dict[str, Any]) -> str:
    if bool(base["strict_success"]):
        return "baseline_success_not_hard_case"
    for step in treatment["trace"][len(base["trace"]) :]:
        if float(step["actual_target_cost"]) <= 1.0:
            return f"recovered_at_step_{int(step['control_step'])}"
    if float(treatment["final_normalized_distance"]) < float(
        base["final_normalized_distance"]
    ) - 1e-12:
        return "unresolved_but_improved"
    if float(treatment["final_normalized_distance"]) > float(
        base["final_normalized_distance"]
    ) + 1e-12:
        return "unresolved_and_worsened"
    return "unresolved_unchanged"


def _arm_rows(
    *,
    name: str,
    arm_analysis: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {_key(row): row for row in baseline_rows}
    treatment = {_key(row): row for row in treatment_rows}
    if set(baseline) != set(treatment) or len(baseline) != 150:
        raise ValueError(f"{name} arms are not 150-row matched sets")
    rules = _rule_for_group(arm_analysis)
    rows = []
    for key in sorted(baseline):
        base, extended = baseline[key], treatment[key]
        if bool(base["strict_success"]):
            continue
        rule = rules[str(base["group_id"])]
        last = base["trace"][-1]
        added_trace = extended["trace"][len(base["trace"]) :]
        rows.append(
            {
                "version": VERSION,
                "arm": name,
                "episode_id": f"{key[0]}__g{key[1]:g}",
                "case_id": key[0],
                "group_id": str(base["group_id"]),
                "stratum": str(base["stratum"]),
                "regime": str(base["regime"]),
                "true_gain_evaluator_only": key[1],
                "budget4_final_distance": float(base["final_normalized_distance"]),
                "budget6_final_distance": float(extended["final_normalized_distance"]),
                "distance_change_budget6_minus_budget4": float(
                    extended["final_normalized_distance"]
                )
                - float(base["final_normalized_distance"]),
                "fourth_step_actual_improvement": float(last["before_target_cost"])
                - float(last["actual_target_cost"]),
                "oof_rule": rule,
                "oof_continuation_selected": _selected(base, rule),
                "fixed6_category": _category(base, extended),
                "fixed6_strict_success": bool(extended["strict_success"]),
                "added_control_steps": len(added_trace),
                "added_actual_target_costs": [
                    float(step["actual_target_cost"]) for step in added_trace
                ],
                "added_commands_mm": [step["command_mm"] for step in added_trace],
                "budget4_saturation_count": int(base["saturation_count"]),
                "budget6_saturation_count": int(extended["saturation_count"]),
                "added_saturation_count": int(extended["saturation_count"])
                - int(base["saturation_count"]),
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    result = {}
    for arm, values in sorted(by_arm.items()):
        result[arm] = {
            "budget4_failures": len(values),
            "fixed6_categories": dict(
                sorted(Counter(str(row["fixed6_category"]) for row in values).items())
            ),
            "oof_continuations": sum(
                bool(row["oof_continuation_selected"]) for row in values
            ),
            "fixed6_recoveries": sum(bool(row["fixed6_strict_success"]) for row in values),
            "mean_distance_change": float(
                np.mean([row["distance_change_budget6_minus_budget4"] for row in values])
            ),
            "median_distance_change": float(
                np.median([row["distance_change_budget6_minus_budget4"] for row in values])
            ),
            "added_saturation_events": int(sum(row["added_saturation_count"] for row in values)),
            "by_stratum": {
                stratum: {
                    "failures": len(subset),
                    "recoveries": sum(bool(row["fixed6_strict_success"]) for row in subset),
                    "oof_continuations": sum(
                        bool(row["oof_continuation_selected"]) for row in subset
                    ),
                }
                for stratum, subset in sorted(
                    {
                        name: [row for row in values if row["stratum"] == name]
                        for name in {row["stratum"] for row in values}
                    }.items()
                )
            },
            "unresolved_episode_ids": [
                row["episode_id"] for row in values if not row["fixed6_strict_success"]
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    for arm in ("direct", "oracle", "probe"):
        parser.add_argument(f"--{arm}-budget4", type=Path, required=True)
        parser.add_argument(f"--{arm}-budget6", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.resolve().read_text())
    if analysis.get("protected_set_used") is not False:
        raise ValueError("temporal case mining requires development-only analysis")
    rows = []
    for arm in ("direct", "oracle", "probe"):
        rows.extend(
            _arm_rows(
                name=arm,
                arm_analysis=analysis["arms"][arm],
                baseline_rows=_read_jsonl(getattr(args, f"{arm}_budget4")),
                treatment_rows=_read_jsonl(getattr(args, f"{arm}_budget6")),
            )
        )
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "records": len(rows),
        "by_arm": _summary(rows),
        "interpretation_guard": (
            "Hard cases are selected by observed budget-four failure. Fixed-six outcomes are "
            "causal matched evidence; conditional recovery rates are not unconditional effects."
        ),
    }
    _write_jsonl(args.records, rows)
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
