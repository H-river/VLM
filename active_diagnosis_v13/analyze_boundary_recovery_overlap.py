#!/usr/bin/env python3
"""Compare temporal and simulator-oracle recovery on exact boundary failures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

VERSION = "active_diagnosis_v13_boundary_recovery_overlap_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _partition(
    universe: set[str], temporal: set[str], candidates: set[str]
) -> dict[str, list[str]]:
    if not temporal <= universe or not candidates <= universe:
        raise ValueError("recovery sets must be subsets of the exact failure universe")
    return {
        "both": sorted(temporal & candidates),
        "temporal_only": sorted(temporal - candidates),
        "candidate_only": sorted(candidates - temporal),
        "neither": sorted(universe - temporal - candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-budget", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    step = json.loads(args.step_budget.resolve().read_text())
    candidate = json.loads(args.candidate_audit.resolve().read_text())
    if step.get("protected_set_used") is not False or candidate.get(
        "protected_set_used"
    ) is not False:
        raise ValueError("overlap analysis requires development-only evidence")
    universe = set(str(value) for value in step["exact_episode_ids"])
    if len(universe) != 20:
        raise ValueError("expected exactly 20 boundary failures")
    temporal6 = set(str(value) for value in step["arms"]["6"]["recovered_episode_ids"])
    temporal8 = set(str(value) for value in step["arms"]["8"]["recovered_episode_ids"])
    pairing = candidate["maximum_budget_pairing"]
    both_candidates = set(str(value) for value in pairing["both_success"])
    uniform_only = set(str(value) for value in pairing["uniform_only_success"])
    conditioned_only = set(str(value) for value in pairing["conditioned_only_success"])
    candidate_union = both_candidates | uniform_only | conditioned_only
    partitions = {
        "budget6_temporal_vs_max_candidate_union": _partition(
            universe, temporal6, candidate_union
        ),
        "budget8_temporal_vs_max_candidate_union": _partition(
            universe, temporal8, candidate_union
        ),
    }
    maximum_budget = int(candidate["budgets"][-1])
    report = {
        "version": VERSION,
        "split": "development_only_selected_failures",
        "protected_set_used": False,
        "exact_failure_episodes": len(universe),
        "temporal_budget6_recoveries": len(temporal6),
        "temporal_budget8_recoveries": len(temporal8),
        "candidate_maximum_budget": maximum_budget,
        "candidate_union_recoveries": len(candidate_union),
        "uniform_candidate_recoveries": len(both_candidates | uniform_only),
        "conditioned_candidate_recoveries": len(both_candidates | conditioned_only),
        "partitions": partitions,
        "combined_temporal8_or_candidate_union_recoveries": len(
            temporal8 | candidate_union
        ),
        "combined_selected_failure_coverage": len(temporal8 | candidate_union)
        / len(universe),
        "combined_fault_set_upper_bound_points": 100.0
        * len(temporal8 | candidate_union)
        / 120.0,
        "mechanism_interpretation": {
            "candidate_only_recoveries": len(candidate_union - temporal8),
            "temporal_only_recoveries": len(temporal8 - candidate_union),
            "shared_recoveries": len(temporal8 & candidate_union),
            "guard": (
                "Distinct recovery sets indicate complementary evaluator headroom; they do "
                "not establish that a visible policy can choose simulator-best candidates."
            ),
        },
        "interpretation_guard": (
            "Both mechanisms start from the exact budget-four oracle failure set. Temporal "
            "recovery is an executed oracle-gain controller result; candidate recovery is an "
            "evaluator-only best-of-k bound on selected development failures."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
