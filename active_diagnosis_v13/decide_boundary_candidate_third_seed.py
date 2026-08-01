#!/usr/bin/env python3
"""Apply a preregistered third proposal-seed uncertainty trigger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

VERSION = "active_diagnosis_v13_boundary_candidate_third_seed_decision_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-seed-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.two_seed_analysis.resolve().read_text())
    if analysis.get("protected_set_used") is not False:
        raise ValueError("third-seed decision requires development-only evidence")
    if len(analysis["proposal_seeds"]) != 2 or int(
        analysis["maximum_common_budget"]
    ) != 96:
        raise ValueError("third-seed decision requires exactly two seeds through budget 96")
    counts = {
        str(label): int(value)
        for label, value in analysis["proposal_union_successes_by_seed"].items()
    }
    spread = max(counts.values()) - min(counts.values())
    stable = len(analysis["stable_union_success_ids_all_seeds"])
    any_count = len(analysis["union_success_ids_any_seed"])
    stable_to_any = 1.0 if any_count == 0 else stable / any_count
    count_spread_threshold = 2
    stable_to_any_threshold = 0.75
    launch = bool(
        spread >= count_spread_threshold or stable_to_any < stable_to_any_threshold
    )
    report = {
        "version": VERSION,
        "protected_set_used": False,
        "decision_budget": 96,
        "proposal_union_successes_by_seed": counts,
        "proposal_union_count_spread": spread,
        "stable_union_successes": stable,
        "union_successes_any_seed": any_count,
        "stable_to_any_recovery_ratio": stable_to_any,
        "launch_if_count_spread_at_least": count_spread_threshold,
        "launch_if_stable_to_any_below": stable_to_any_threshold,
        "launch_seed3": launch,
        "seed3_root_seed": 2026080103,
        "seed3_maximum_budget": 96,
        "interpretation_guard": (
            "This trigger was registered before proposal seed 2 completed. A third seed "
            "quantifies unresolved evaluator best-of-k sampling variability only and cannot "
            "select a policy, proposal family, primary branch, or protected result."
        ),
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
