#!/usr/bin/env python3
"""Describe exact fresh-suite failure stability across planner seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from active_diagnosis_v13.analyze_fresh_holdout_seed_setup_robustness import (
    _parse_artifact,
)

VERSION = "active_diagnosis_v13_fresh_holdout_failure_stability_v1"
FAULT_GAINS = (0.5, 0.75, 1.25, 1.5)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _group_id(episode_id: str) -> str:
    return episode_id.rsplit("__g", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=_parse_artifact, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifacts: dict[tuple[str, int], dict[str, Any]] = {}
    for suite, seed, path in args.artifact:
        artifact = json.loads(path.resolve().read_text())
        if (suite, seed) in artifacts:
            raise ValueError(f"duplicate artifact {(suite, seed)}")
        if artifact.get("planner_root_seed") != seed:
            raise ValueError(f"planner seed mismatch for {(suite, seed)}")
        if artifact.get("protected_set_used") is not False or artifact.get(
            "selection_or_retuning_on_fresh_suite"
        ) is not False:
            raise ValueError(f"{(suite, seed)} is not frozen non-protected evidence")
        artifacts[(suite, seed)] = artifact

    suites = sorted({suite for suite, _ in artifacts})
    seeds = sorted({seed for _, seed in artifacts})
    expected_cells = {(suite, seed) for suite in suites for seed in seeds}
    if len(suites) < 2 or len(seeds) != 2 or set(artifacts) != expected_cells:
        raise ValueError(
            "failure audit requires a complete matrix of at least two suites by two seeds"
        )
    rules = {
        json.dumps(artifact["frozen_rule"], sort_keys=True)
        for artifact in artifacts.values()
    }
    if len(rules) != 1:
        raise ValueError("artifacts use different frozen rules")

    all_expected: set[str] = set()
    per_suite = []
    all_failure_sets: dict[int, set[str]] = {seed: set() for seed in seeds}
    all_miss_sets: dict[int, set[str]] = {seed: set() for seed in seeds}
    for suite in suites:
        group_sets = {
            tuple(sorted(artifacts[(suite, seed)]["setup_independence"]["fresh_group_ids"]))
            for seed in seeds
        }
        if len(group_sets) != 1:
            raise ValueError(f"suite {suite} setups changed across seeds")
        groups = next(iter(group_sets))
        expected_episodes = {
            f"{group}__g{gain:g}" for group in groups for gain in FAULT_GAINS
        }
        if len(expected_episodes) != 120 or all_expected & expected_episodes:
            raise ValueError(f"suite {suite} is not a distinct 120-fault-episode suite")
        all_expected |= expected_episodes
        failure_sets = {
            seed: set(
                artifacts[(suite, seed)]["exact_ids"][
                    "frozen_sequential_fault_failures"
                ]
            )
            for seed in seeds
        }
        miss_sets = {
            seed: set(
                artifacts[(suite, seed)]["exact_ids"]["fixed8_recoveries_missed"]
            )
            for seed in seeds
        }
        if any(not values <= expected_episodes for values in failure_sets.values()):
            raise ValueError(f"suite {suite} contains an out-of-suite failure ID")
        stable_failures = set.intersection(*failure_sets.values())
        any_failures = set.union(*failure_sets.values())
        discordant = failure_sets[seeds[0]] ^ failure_sets[seeds[1]]
        stable_misses = set.intersection(*miss_sets.values())
        per_suite.append(
            {
                "suite": suite,
                "fault_episodes": len(expected_episodes),
                "failures_by_seed": {
                    str(seed): len(failure_sets[seed]) for seed in seeds
                },
                "stable_failure_ids_all_seeds": sorted(stable_failures),
                "stable_failure_group_ids_all_seeds": sorted(
                    {_group_id(value) for value in stable_failures}
                ),
                "failure_ids_any_seed": sorted(any_failures),
                "planner_seed_discordant_failure_ids": sorted(discordant),
                "fault_outcome_agreement_rate": 1.0
                - len(discordant) / len(expected_episodes),
                "stable_fixed8_recovery_misses_all_seeds": sorted(stable_misses),
            }
        )
        for seed in seeds:
            all_failure_sets[seed] |= failure_sets[seed]
            all_miss_sets[seed] |= miss_sets[seed]

    stable_all = set.intersection(*all_failure_sets.values())
    any_all = set.union(*all_failure_sets.values())
    discordant_all = all_failure_sets[seeds[0]] ^ all_failure_sets[seeds[1]]
    stable_misses_all = set.intersection(*all_miss_sets.values())
    report = {
        "version": VERSION,
        "split": "posthoc_descriptive_fresh_nonprotected_failure_stability",
        "protected_set_used": False,
        "selection_or_retuning_on_fresh_suites": False,
        "frozen_rule": json.loads(next(iter(rules))),
        "suites": suites,
        "planner_root_seeds": seeds,
        "fault_episodes_per_seed": len(all_expected),
        "per_suite": per_suite,
        "failures_by_seed": {
            str(seed): len(all_failure_sets[seed]) for seed in seeds
        },
        "stable_failure_ids_all_seeds": sorted(stable_all),
        "stable_failure_group_ids_all_seeds": sorted(
            {_group_id(value) for value in stable_all}
        ),
        "failure_ids_any_seed": sorted(any_all),
        "planner_seed_discordant_failure_ids": sorted(discordant_all),
        "fault_outcome_agreement_rate": 1.0
        - len(discordant_all) / len(all_expected),
        "stable_fixed8_recovery_misses_all_seeds": sorted(stable_misses_all),
        "interpretation_guard": (
            "This exact-ID audit was registered before either second-seed result. It is a "
            "posthoc descriptive stability analysis and cannot select or retune the frozen "
            "model, stopping rule, setup suites, planner seeds, or matched-cohort selection."
        ),
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
