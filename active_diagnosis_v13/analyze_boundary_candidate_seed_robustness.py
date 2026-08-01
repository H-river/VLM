#!/usr/bin/env python3
"""Compare boundary-candidate coverage across independent proposal seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "active_diagnosis_v13_boundary_candidate_seed_robustness_v3"
PROPOSALS = ("uniform_feasible", "boundary_conditioned")
FAULT_GAIN_LABELS = ("0.5", "0.75", "1.25", "1.5")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_audit(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("audit must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("audit must be LABEL=PATH")
    return label, Path(path)


def _counts_by_gain(episode_ids: set[str]) -> dict[str, int]:
    counts = {label: 0 for label in FAULT_GAIN_LABELS}
    for episode_id in episode_ids:
        if "__g" not in episode_id:
            raise ValueError(f"candidate success ID lacks gain suffix: {episode_id}")
        gain = episode_id.rsplit("__g", 1)[1]
        if gain not in counts:
            raise ValueError(f"unexpected candidate success gain {gain}: {episode_id}")
        counts[gain] += 1
    return counts


def _group_id(episode_id: str) -> str:
    if "__g" not in episode_id:
        raise ValueError(f"candidate success ID lacks gain suffix: {episode_id}")
    return episode_id.rsplit("__g", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", type=_parse_audit, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.audit) < 2:
        raise ValueError("at least two independent proposal-seed audits are required")
    audits = {label: json.loads(path.resolve().read_text()) for label, path in args.audit}
    if len(audits) != len(args.audit):
        raise ValueError("proposal-seed audit labels must be unique")
    for label, audit in audits.items():
        if audit.get("protected_set_used") is not False or int(audit["episodes"]) != 20:
            raise ValueError(f"{label} is not a 20-episode development-only audit")
    common_budgets = sorted(set.intersection(*(set(audit["budgets"]) for audit in audits.values())))
    if not common_budgets:
        raise ValueError("proposal-seed audits share no candidate budget")
    curves = {
        label: {int(row["candidate_budget"]): row for row in audit["nested_curve"]}
        for label, audit in audits.items()
    }
    curve = []
    for budget in common_budgets:
        proposal_blocks = {}
        for proposal in PROPOSALS:
            successes = {
                label: int(curves[label][budget]["proposals"][proposal]["strict_successes"])
                for label in sorted(curves)
            }
            success_sets = {
                label: set(
                    curves[label][budget]["proposals"][proposal]["strict_success_ids"]
                )
                for label in sorted(curves)
            }
            proposal_blocks[proposal] = {
                "successes_by_seed": successes,
                "mean_successes": float(np.mean(list(successes.values()))),
                "minimum_successes": min(successes.values()),
                "maximum_successes": max(successes.values()),
                "stable_success_ids_all_seeds": sorted(
                    set.intersection(*success_sets.values())
                ),
                "success_ids_any_seed": sorted(set.union(*success_sets.values())),
            }
        union_sets_at_budget = {
            label: set(
                curves[label][budget]["proposals"]["uniform_feasible"][
                    "strict_success_ids"
                ]
            )
            | set(
                curves[label][budget]["proposals"]["boundary_conditioned"][
                    "strict_success_ids"
                ]
            )
            for label in sorted(curves)
        }
        stable_union = set.intersection(*union_sets_at_budget.values())
        any_union = set.union(*union_sets_at_budget.values())
        pairwise_jaccard = {}
        union_labels = sorted(union_sets_at_budget)
        for index, label in enumerate(union_labels):
            for other in union_labels[index + 1 :]:
                intersection = union_sets_at_budget[label] & union_sets_at_budget[other]
                combined = union_sets_at_budget[label] | union_sets_at_budget[other]
                pairwise_jaccard[f"{label}__{other}"] = (
                    1.0 if not combined else len(intersection) / len(combined)
                )
        curve.append(
            {
                "candidate_budget": budget,
                "proposals": proposal_blocks,
                "proposal_union": {
                    "successes_by_seed": {
                        label: len(values)
                        for label, values in union_sets_at_budget.items()
                    },
                    "mean_successes": float(
                        np.mean([len(values) for values in union_sets_at_budget.values()])
                    ),
                    "minimum_successes": min(
                        len(values) for values in union_sets_at_budget.values()
                    ),
                    "maximum_successes": max(
                        len(values) for values in union_sets_at_budget.values()
                    ),
                    "stable_success_ids_all_seeds": sorted(stable_union),
                    "success_ids_any_seed": sorted(any_union),
                    "stable_to_any_success_ratio": (
                        1.0 if not any_union else len(stable_union) / len(any_union)
                    ),
                    "pairwise_success_jaccard": pairwise_jaccard,
                },
            }
        )
    maximum = curve[-1]
    union_sets = {}
    for label in sorted(curves):
        row = curves[label][maximum["candidate_budget"]]
        union_sets[label] = set(
            row["proposals"]["uniform_feasible"]["strict_success_ids"]
        ) | set(row["proposals"]["boundary_conditioned"]["strict_success_ids"])
    recovered_any = set.union(*union_sets.values())
    recovery_frequency = {
        episode_id: sum(episode_id in values for values in union_sets.values())
        for episode_id in sorted(recovered_any)
    }
    seed_count = len(union_sets)
    report = {
        "version": VERSION,
        "split": "development_only_selected_failures",
        "protected_set_used": False,
        "proposal_seeds": sorted(audits),
        "audit_provenance": {
            label: {
                "record_schema_versions": audit.get("record_schema_versions", []),
                "planner_root_seed_field_complete": bool(
                    audit.get("planner_root_seed_field_complete", False)
                ),
                "planner_root_seed_values": audit.get("planner_root_seed_values", []),
            }
            for label, audit in sorted(audits.items())
        },
        "episodes": 20,
        "common_budgets": common_budgets,
        "seed_robustness_curve": curve,
        "maximum_common_budget": maximum["candidate_budget"],
        "proposal_union_successes_by_seed": {
            label: len(values) for label, values in union_sets.items()
        },
        "stable_union_success_ids_all_seeds": sorted(
            set.intersection(*union_sets.values())
        ),
        "union_success_ids_any_seed": sorted(recovered_any),
        "maximum_budget_seed_recovery_frequency": {
            "recovery_seed_count_by_episode": recovery_frequency,
            "episode_count_by_recovery_seed_count": {
                str(count): sum(value == count for value in recovery_frequency.values())
                for count in range(1, seed_count + 1)
            },
            "stable_recovery_ids_all_seeds": sorted(
                episode_id
                for episode_id, count in recovery_frequency.items()
                if count == seed_count
            ),
            "seed_variable_recovery_ids": sorted(
                episode_id
                for episode_id, count in recovery_frequency.items()
                if count < seed_count
            ),
        },
        "maximum_budget_recovery_slices": {
            "successes_by_gain_per_seed": {
                label: _counts_by_gain(values)
                for label, values in union_sets.items()
            },
            "stable_successes_by_gain_all_seeds": _counts_by_gain(
                set.intersection(*union_sets.values())
            ),
            "successes_by_gain_any_seed": _counts_by_gain(
                recovered_any
            ),
            "stable_success_group_ids_all_seeds": sorted(
                {
                    _group_id(value)
                    for value in set.intersection(*union_sets.values())
                }
            ),
            "success_group_ids_any_seed": sorted(
                {_group_id(value) for value in recovered_any}
            ),
        },
        "minimum_seed_candidate_union_fault_upper_bound_points": 100.0
        * min(len(values) for values in union_sets.values())
        / 120.0,
        "interpretation_guard": (
            "This tests evaluator best-of-k sampling variability on the same selected "
            "development failures. It is not new-setup or deployable-policy validation."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
