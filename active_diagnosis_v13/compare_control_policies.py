#!/usr/bin/env python3
"""Compare matched development control policies against direct H1."""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_matched_policy_comparison_v2"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _bootstrap(
    reference: list[dict[str, Any]], policy: list[dict[str, Any]], seed: int
) -> dict[str, float]:
    def grouped(rows: list[dict[str, Any]]) -> dict[str, float]:
        values: defaultdict[str, list[float]] = defaultdict(list)
        for row in rows:
            values[str(row["group_id"])].append(float(row["strict_success"]))
        return {key: float(np.mean(item)) for key, item in values.items()}

    left, right = grouped(reference), grouped(policy)
    groups = np.asarray(sorted(set(left) & set(right)))
    differences = np.asarray([right[group] - left[group] for group in groups])
    rng = np.random.default_rng(seed)
    estimates = [
        float(np.mean(differences[rng.integers(0, len(groups), len(groups))]))
        for _ in range(4000)
    ]
    low, high = np.quantile(estimates, (0.025, 0.975))
    return {"estimate": float(differences.mean()), "low": float(low), "high": float(high)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026080107)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    direct = _jsonl(root / "control" / "direct.jsonl")
    if len(direct) != 150:
        raise ValueError("direct control must have 150 rows")
    direct_fault = [row for row in direct if float(row["evaluator_only_true_gain"]) != 1.0]
    direct_by_key = {_key(row): row for row in direct_fault}
    comparisons = []
    policy_fault_rows: dict[str, list[dict[str, Any]]] = {}
    for offset, name in enumerate(args.policy):
        rows = _jsonl(root / "control" / f"{name}.jsonl")
        if len(rows) != 150:
            raise ValueError(f"{name} has {len(rows)} rows, expected 150")
        suffixes = {int(str(row["case_id"]).rsplit("_", 1)[1]) for row in rows}
        if any(suffix >= 10 for suffix in suffixes):
            raise ValueError(f"protected case present in {name}")
        fault = [row for row in rows if float(row["evaluator_only_true_gain"]) != 1.0]
        policy_fault_rows[name] = fault
        by_key = {_key(row): row for row in fault}
        if set(by_key) != set(direct_by_key):
            raise ValueError(f"matched fault keys differ for {name}")
        raw_estimates = [row.get("raw_gain_estimate") for row in rows]
        hypotheses = np.asarray([0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64)
        continuous_estimator = bool(
            raw_estimates
            and all(value is not None for value in raw_estimates)
            and any(
                not np.any(
                    np.isclose(float(value), hypotheses, rtol=0.0, atol=1e-12)
                )
                for value in raw_estimates
            )
        )
        rate = lambda values: float(np.mean([bool(row["strict_success"]) for row in values]))
        recoveries = sum(
            not bool(direct_by_key[key]["strict_success"])
            and bool(by_key[key]["strict_success"])
            for key in by_key
        )
        regressions = sum(
            bool(direct_by_key[key]["strict_success"])
            and not bool(by_key[key]["strict_success"])
            for key in by_key
        )
        comparisons.append(
            {
                "policy": name,
                "episodes": len(rows),
                "fault_episodes": len(fault),
                "overall_success": rate(rows),
                "fault_success": rate(fault),
                "fault_success_gain_over_direct": rate(fault) - rate(direct_fault),
                "matched_direct_failure_recoveries": int(recoveries),
                "matched_direct_success_regressions": int(regressions),
                "matched_group_bootstrap_95": _bootstrap(
                    direct_fault, fault, args.seed + offset
                ),
                "continuous_gain_estimator": continuous_estimator,
                "gain_classification_accuracy": (
                    float(np.mean([bool(row["gain_classification_correct"]) for row in rows]))
                    if all(row.get("gain_classification_correct") is not None for row in rows)
                    and not continuous_estimator
                    else None
                ),
                "mean_absolute_raw_gain_error": (
                    float(
                        np.mean(
                            [
                                abs(
                                    float(row["raw_gain_estimate"])
                                    - float(row["evaluator_only_true_gain"])
                                )
                                for row in rows
                            ]
                        )
                    )
                    if all(value is not None for value in raw_estimates)
                    else None
                ),
                "mean_total_additional_steps": float(
                    np.mean([int(row["total_additional_steps"]) for row in rows])
                ),
                "saturation_episode_rate": float(
                    np.mean([int(row["saturation_count"]) > 0 for row in rows])
                ),
            }
        )
    pairwise_comparisons = []
    for offset, (reference_name, alternate_name) in enumerate(
        combinations(args.policy, 2), start=len(args.policy)
    ):
        reference = policy_fault_rows[reference_name]
        alternate = policy_fault_rows[alternate_name]
        reference_by_key = {_key(row): row for row in reference}
        alternate_by_key = {_key(row): row for row in alternate}
        if set(reference_by_key) != set(alternate_by_key):
            raise ValueError(
                f"matched fault keys differ for {reference_name} and {alternate_name}"
            )
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
        reference_success = float(
            np.mean([bool(row["strict_success"]) for row in reference])
        )
        alternate_success = float(
            np.mean([bool(row["strict_success"]) for row in alternate])
        )
        pairwise_comparisons.append(
            {
                "reference_policy": reference_name,
                "alternate_policy": alternate_name,
                "fault_episodes": len(reference),
                "reference_fault_success": reference_success,
                "alternate_fault_success": alternate_success,
                "fault_success_difference": alternate_success - reference_success,
                "matched_reference_failure_recoveries": int(recoveries),
                "matched_reference_success_regressions": int(regressions),
                "matched_group_bootstrap_95": _bootstrap(
                    reference, alternate, args.seed + offset
                ),
            }
        )
    report: dict[str, Any] = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "direct_fault_success": float(
            np.mean([bool(row["strict_success"]) for row in direct_fault])
        ),
        "comparisons": comparisons,
        "pairwise_comparisons": pairwise_comparisons,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
