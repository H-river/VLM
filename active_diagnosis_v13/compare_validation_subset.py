#!/usr/bin/env python3
"""Compare matched control policies on an anchor policy's development subset."""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.compare_control_policies import _bootstrap, _key


VERSION = "active_diagnosis_v13_validation_subset_comparison_v2"
GAINS = {0.5, 0.75, 1.0, 1.25, 1.5}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _complete_case_ids(rows: list[dict[str, Any]]) -> set[str]:
    by_case: dict[str, set[float]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), set()).add(
            float(row["evaluator_only_true_gain"])
        )
    incomplete = {
        case_id: sorted(gains) for case_id, gains in by_case.items() if gains != GAINS
    }
    if incomplete:
        raise ValueError(f"incomplete gain groups: {incomplete}")
    return set(by_case)


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _summary_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy_values = [row.get("gain_classification_correct") for row in rows]
    return {
        "episodes": len(rows),
        "strict_success": _rate(rows),
        "gain_classification_accuracy": (
            float(np.mean(list(map(bool, accuracy_values))))
            if all(value is not None for value in accuracy_values)
            else None
        ),
        "saturation_episode_rate": float(
            np.mean([int(row["saturation_count"]) > 0 for row in rows])
        ),
        "mean_total_additional_steps": float(
            np.mean([int(row["total_additional_steps"]) for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--anchor-policy", required=True)
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026080114)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    anchor = _jsonl(root / "control" / f"{args.anchor_policy}.jsonl")
    case_ids = _complete_case_ids(anchor)
    if not case_ids:
        raise ValueError("anchor policy is empty")
    if any(int(case_id.rsplit("_", 1)[1]) >= 10 for case_id in case_ids):
        raise ValueError("protected case present in anchor policy")
    expected_keys = {_key(row) for row in anchor}
    selected: dict[str, list[dict[str, Any]]] = {}
    for name in args.policy:
        rows = [
            row
            for row in _jsonl(root / "control" / f"{name}.jsonl")
            if str(row["case_id"]) in case_ids
        ]
        if {_key(row) for row in rows} != expected_keys:
            raise ValueError(f"{name} does not match anchor keys")
        if len(rows) != len(expected_keys):
            raise ValueError(f"{name} contains duplicate anchor keys")
        selected[name] = rows
    nominal = 1.0
    summaries = []
    for name in args.policy:
        rows = selected[name]
        fault = [row for row in rows if float(row["evaluator_only_true_gain"]) != nominal]
        accuracy_values = [row.get("gain_classification_correct") for row in rows]
        summaries.append(
            {
                "policy": name,
                "episodes": len(rows),
                "fault_episodes": len(fault),
                "overall_success": _rate(rows),
                "fault_success": _rate(fault),
                "gain_classification_accuracy": (
                    float(np.mean(list(map(bool, accuracy_values))))
                    if all(value is not None for value in accuracy_values)
                    else None
                ),
                "saturation_episode_rate": float(
                    np.mean([int(row["saturation_count"]) > 0 for row in rows])
                ),
                "mean_total_additional_steps": float(
                    np.mean([int(row["total_additional_steps"]) for row in rows])
                ),
                "by_gain": {
                    f"{gain:g}": _summary_block(
                        [
                            row
                            for row in rows
                            if float(row["evaluator_only_true_gain"]) == gain
                        ]
                    )
                    for gain in sorted(GAINS)
                },
                "by_stratum": {
                    stratum: _summary_block(
                        [row for row in fault if str(row["stratum"]) == stratum]
                    )
                    for stratum in sorted({str(row["stratum"]) for row in fault})
                },
            }
        )
    pairwise = []
    for offset, (reference_name, alternate_name) in enumerate(
        combinations(args.policy, 2)
    ):
        reference = [
            row
            for row in selected[reference_name]
            if float(row["evaluator_only_true_gain"]) != nominal
        ]
        alternate = [
            row
            for row in selected[alternate_name]
            if float(row["evaluator_only_true_gain"]) != nominal
        ]
        reference_by_key = {_key(row): row for row in reference}
        alternate_by_key = {_key(row): row for row in alternate}
        recoveries = sum(
            not bool(reference_by_key[key]["strict_success"])
            and bool(alternate_by_key[key]["strict_success"])
            for key in reference_by_key
        )
        regressions = sum(
            bool(reference_by_key[key]["strict_success"])
            and not bool(alternate_by_key[key]["strict_success"])
            for key in reference_by_key
        )
        pairwise.append(
            {
                "reference_policy": reference_name,
                "alternate_policy": alternate_name,
                "fault_success_difference": _rate(alternate) - _rate(reference),
                "matched_reference_failure_recoveries": int(recoveries),
                "matched_reference_success_regressions": int(regressions),
                "matched_group_bootstrap_95": _bootstrap(
                    reference, alternate, args.seed + offset
                ),
            }
        )
    report = {
        "version": VERSION,
        "split": "development_group_heldout_validation_subset",
        "protected_set_used": False,
        "anchor_policy": args.anchor_policy,
        "cases": len(case_ids),
        "case_ids": sorted(case_ids),
        "episodes": len(expected_keys),
        "groups": len({str(row["group_id"]) for row in anchor}),
        "policies": summaries,
        "pairwise_comparisons": pairwise,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
