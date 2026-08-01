#!/usr/bin/env python3
"""Audit paired boundary-candidate coverage records with clustered intervals."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

VERSION = "active_diagnosis_v13_boundary_candidate_coverage_audit_v1"
PROPOSALS = ("uniform_feasible", "boundary_conditioned")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _group_bootstrap(
    values: list[tuple[str, float]], *, seed: int, samples: int = 10000
) -> dict[str, float | int]:
    if not values:
        raise ValueError("bootstrap values must be nonempty")
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for group, value in values:
        grouped[group].append(float(value))
    group_values = np.asarray(
        [float(np.mean(grouped[group])) for group in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            float(
                np.mean(
                    group_values[
                        rng.integers(0, len(group_values), size=len(group_values))
                    ]
                )
            )
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "episode_estimate": float(np.mean([value for _, value in values])),
        "group_mean_estimate": float(np.mean(group_values)),
        "low": float(low),
        "high": float(high),
        "independent_groups": int(len(group_values)),
        "samples": int(samples),
    }


def _best(row: dict[str, Any], proposal: str, budget: int) -> dict[str, Any]:
    candidates = row["proposals"][proposal]["candidates"][:budget]
    if len(candidates) != budget:
        raise ValueError(f"{row['record_id']} has fewer than {budget} candidates")
    return min(candidates, key=lambda candidate: float(candidate["terminal_normalized_distance"]))


def _ids(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[str]:
    return sorted(str(row["record_id"]).removesuffix("__boundary_candidate_coverage") for row in rows if predicate(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.records.resolve().read_text().splitlines()
        if line
    ]
    if len(rows) != 20 or len({row["record_id"] for row in rows}) != 20:
        raise ValueError("coverage audit requires exactly 20 unique records")
    record_versions = sorted({str(row["version"]) for row in rows})
    if len(record_versions) != 1:
        raise ValueError(f"coverage records mix schema versions: {record_versions}")
    for row in rows:
        expected_record_id = (
            f"{row['case_id']}__g{float(row['evaluator_only_true_gain']):g}"
            "__boundary_candidate_coverage"
        )
        if row["record_id"] != expected_record_id:
            raise ValueError(f"candidate record ID mismatch: {row['record_id']}")
        if (
            row["stratum"] != "reachable_boundary_or_clipping"
            or float(row["evaluator_only_true_gain"]) == 1.0
            or bool(row["source_budget4_strict_success"])
            or int(row["sequence_horizon"]) != 2
            or set(row["proposals"]) != set(PROPOSALS)
        ):
            raise ValueError(f"candidate record violates selected-failure contract: {row['record_id']}")
        maximum_budget = int(row["maximum_budget"])
        for proposal in PROPOSALS:
            candidates = row["proposals"][proposal]["candidates"]
            if len(candidates) != maximum_budget:
                raise ValueError(
                    f"{row['record_id']} {proposal} has incomplete candidate trace"
                )
            for candidate in candidates:
                distance = float(candidate["terminal_normalized_distance"])
                if (
                    not np.isfinite(distance)
                    or bool(candidate["strict_success"]) != (distance <= 1.0)
                    or len(candidate["desired_physical_sequence"]) != 2
                    or len(candidate["issued_command_sequence"]) != 2
                ):
                    raise ValueError(
                        f"{row['record_id']} {proposal} has invalid candidate semantics"
                    )
    budgets = sorted(set(int(value) for value in args.budgets))
    if budgets[0] <= 0 or budgets[-1] > min(int(row["maximum_budget"]) for row in rows):
        raise ValueError("invalid nested candidate budgets")
    if not all(bool(row["proposal_raw_uniforms_matched"]) for row in rows):
        raise ValueError("proposal raw uniforms are not matched")
    curve = []
    for index, budget in enumerate(budgets):
        best = {
            proposal: {row["record_id"]: _best(row, proposal, budget) for row in rows}
            for proposal in PROPOSALS
        }
        per_proposal = {}
        for proposal in PROPOSALS:
            candidates = list(best[proposal].values())
            distances = np.asarray(
                [float(candidate["terminal_normalized_distance"]) for candidate in candidates]
            )
            per_proposal[proposal] = {
                "strict_success": float(np.mean(distances <= 1.0)),
                "strict_successes": int(np.sum(distances <= 1.0)),
                "strict_success_ids": sorted(
                    str(record_id).removesuffix("__boundary_candidate_coverage")
                    for record_id, candidate in best[proposal].items()
                    if bool(candidate["strict_success"])
                ),
                "mean_best_terminal_distance": float(np.mean(distances)),
                "median_best_terminal_distance": float(np.median(distances)),
            }
        success_differences = []
        distance_differences = []
        for row in rows:
            uniform = best["uniform_feasible"][row["record_id"]]
            conditioned = best["boundary_conditioned"][row["record_id"]]
            group = str(row["group_id"])
            success_differences.append(
                (
                    group,
                    float(conditioned["strict_success"])
                    - float(uniform["strict_success"]),
                )
            )
            distance_differences.append(
                (
                    group,
                    float(conditioned["terminal_normalized_distance"])
                    - float(uniform["terminal_normalized_distance"]),
                )
            )
        curve.append(
            {
                "candidate_budget": budget,
                "proposals": per_proposal,
                "conditioned_minus_uniform_success": _group_bootstrap(
                    success_differences, seed=2026080181 + index
                ),
                "conditioned_minus_uniform_best_distance": _group_bootstrap(
                    distance_differences, seed=2026080191 + index
                ),
            }
        )
    maximum = budgets[-1]
    uniform_success = {
        row["record_id"]: bool(_best(row, "uniform_feasible", maximum)["strict_success"])
        for row in rows
    }
    conditioned_success = {
        row["record_id"]: bool(
            _best(row, "boundary_conditioned", maximum)["strict_success"]
        )
        for row in rows
    }
    conditioning_counts = [
        int(row["proposals"]["boundary_conditioned"]["conditioning_active_candidates"])
        for row in rows
    ]
    report = {
        "version": VERSION,
        "split": "development_only_selected_failures",
        "protected_set_used": False,
        "episodes": len(rows),
        "record_schema_versions": record_versions,
        "planner_root_seed_field_complete": all(
            "planner_root_seed" in row for row in rows
        ),
        "planner_root_seed_values": sorted(
            {
                int(row["planner_root_seed"])
                for row in rows
                if "planner_root_seed" in row
            }
        ),
        "budgets": budgets,
        "nested_curve": curve,
        "maximum_budget_pairing": {
            "both_success": _ids(
                rows,
                lambda row: uniform_success[row["record_id"]]
                and conditioned_success[row["record_id"]],
            ),
            "uniform_only_success": _ids(
                rows,
                lambda row: uniform_success[row["record_id"]]
                and not conditioned_success[row["record_id"]],
            ),
            "conditioned_only_success": _ids(
                rows,
                lambda row: not uniform_success[row["record_id"]]
                and conditioned_success[row["record_id"]],
            ),
            "neither_success": _ids(
                rows,
                lambda row: not uniform_success[row["record_id"]]
                and not conditioned_success[row["record_id"]],
            ),
        },
        "conditioning_reach": {
            "episodes_with_any_conditioning": int(np.sum(np.asarray(conditioning_counts) > 0)),
            "mean_active_candidate_fraction": float(
                np.mean(np.asarray(conditioning_counts, dtype=np.float64) / maximum)
            ),
            "mean_first_action_pairwise_normalized_distance": {
                proposal: float(
                    np.mean(
                        [
                            row["proposals"][proposal][
                                "mean_first_action_pairwise_normalized_distance"
                            ]
                            for row in rows
                        ]
                    )
                )
                for proposal in PROPOSALS
            },
        },
        "interpretation_guard": (
            "All intervals cluster the twenty selected fault episodes by setup group. "
            "Best-of-k is an evaluator-only simulator coverage bound, not deployable "
            "policy performance; no protected data are used."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
