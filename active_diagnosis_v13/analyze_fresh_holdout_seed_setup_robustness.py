#!/usr/bin/env python3
"""Audit a frozen rule across independent fresh setups and planner seeds."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.analyze_temporal_seed_statistics import _two_way_bootstrap

VERSION = "active_diagnosis_v13_fresh_holdout_seed_setup_robustness_v1"


def _parse_artifact(value: str) -> tuple[str, int, Path]:
    """Parse SUITE:PLANNER_SEED=PATH."""
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("artifact must be SUITE:PLANNER_SEED=PATH")
    label, path = value.split("=", 1)
    suite, seed_raw = label.rsplit(":", 1)
    try:
        seed = int(seed_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("planner seed must be an integer") from exc
    if not suite or not path:
        raise argparse.ArgumentTypeError("artifact must be SUITE:PLANNER_SEED=PATH")
    return suite, seed, Path(path)


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
    parser.add_argument("--artifact", action="append", type=_parse_artifact, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifacts: dict[tuple[str, int], dict[str, Any]] = {}
    for suite, seed, path in args.artifact:
        key = (suite, seed)
        if key in artifacts:
            raise ValueError(f"duplicate suite/seed artifact: {key}")
        artifact = json.loads(path.resolve().read_text())
        if artifact.get("planner_root_seed") != seed:
            raise ValueError(
                f"declared seed {seed} disagrees with {path}: "
                f"{artifact.get('planner_root_seed')}"
            )
        if artifact.get("protected_set_used") is not False or artifact.get(
            "selection_or_retuning_on_fresh_suite"
        ) is not False:
            raise ValueError(f"{key} is not frozen one-shot non-protected evidence")
        artifacts[key] = artifact

    suites = sorted({suite for suite, _ in artifacts})
    seeds = sorted({seed for _, seed in artifacts})
    expected = {(suite, seed) for suite in suites for seed in seeds}
    if len(suites) < 2 or len(seeds) < 2 or set(artifacts) != expected:
        raise ValueError("artifacts must form a complete matrix of at least two suites and seeds")

    rules = {
        json.dumps(artifact["frozen_rule"], sort_keys=True)
        for artifact in artifacts.values()
    }
    if len(rules) != 1:
        raise ValueError("all artifacts must use one identical frozen rule")

    suite_groups: dict[str, list[str]] = {}
    suite_hashes: dict[str, set[str]] = {}
    for suite in suites:
        group_sets = {
            tuple(sorted(artifacts[(suite, seed)]["setup_independence"]["fresh_group_ids"]))
            for seed in seeds
        }
        hash_sets = {
            tuple(sorted(artifacts[(suite, seed)]["setup_independence"]["fresh_setup_hashes"]))
            for seed in seeds
        }
        if len(group_sets) != 1 or len(hash_sets) != 1:
            raise ValueError(f"fresh setups changed across planner seeds in suite {suite}")
        suite_groups[suite] = list(next(iter(group_sets)))
        suite_hashes[suite] = set(next(iter(hash_sets)))
        if len(suite_groups[suite]) != 30 or len(suite_hashes[suite]) != 30:
            raise ValueError(f"suite {suite} must contain thirty unique setups")

    for index, suite in enumerate(suites):
        for other in suites[index + 1 :]:
            if set(suite_groups[suite]) & set(suite_groups[other]):
                raise ValueError(f"group overlap between suites {suite} and {other}")
            if suite_hashes[suite] & suite_hashes[other]:
                raise ValueError(f"setup-hash overlap between suites {suite} and {other}")

    ordered_groups = [group for suite in suites for group in suite_groups[suite]]
    matrix = np.empty((len(seeds), len(ordered_groups)), dtype=np.float64)
    per_cell = []
    recoveries_by_seed: dict[int, int] = defaultdict(int)
    regressions_by_seed: dict[int, int] = defaultdict(int)
    for seed_index, seed in enumerate(seeds):
        offset = 0
        for suite in suites:
            artifact = artifacts[(suite, seed)]
            effects = {
                row["group_id"]: float(row["frozen_sequential_minus_fixed4"])
                for row in artifact["group_effects"]
            }
            if set(effects) != set(suite_groups[suite]):
                raise ValueError(f"group effects do not match suite {suite} at seed {seed}")
            suite_values = np.asarray(
                [effects[group] for group in suite_groups[suite]], dtype=np.float64
            )
            matrix[seed_index, offset : offset + len(suite_values)] = suite_values
            offset += len(suite_values)
            recoveries_by_seed[seed] += int(artifact["recoveries"])
            regressions_by_seed[seed] += int(artifact["regressions"])
            per_cell.append(
                {
                    "suite": suite,
                    "planner_root_seed": seed,
                    "setup_groups": len(suite_values),
                    "fault_gain_over_fixed4": float(np.mean(suite_values)),
                    "recoveries": int(artifact["recoveries"]),
                    "regressions": int(artifact["regressions"]),
                    "probe_gain_accuracy": float(
                        artifact["probe_gain_classification"]["overall_accuracy"]
                    ),
                }
            )

    per_seed = [
        {
            "planner_root_seed": seed,
            "fault_gain_over_fixed4": float(np.mean(matrix[index])),
            "recoveries": int(recoveries_by_seed[seed]),
            "regressions": int(regressions_by_seed[seed]),
        }
        for index, seed in enumerate(seeds)
    ]
    report = {
        "version": VERSION,
        "split": "balanced_fresh_nonprotected_setups_by_planner_seed",
        "protected_set_used": False,
        "selection_or_retuning_on_fresh_suites": False,
        "frozen_rule": json.loads(next(iter(rules))),
        "suites": suites,
        "planner_root_seeds": seeds,
        "independent_setup_groups": len(ordered_groups),
        "balanced_seed_by_group_shape": list(matrix.shape),
        "per_suite_seed": per_cell,
        "per_seed": per_seed,
        "all_suite_seed_cells_positive_over_fixed4": all(
            row["fault_gain_over_fixed4"] > 0.0 for row in per_cell
        ),
        "all_planner_seeds_positive_over_fixed4": all(
            row["fault_gain_over_fixed4"] > 0.0 for row in per_seed
        ),
        "two_way_seed_group_bootstrap_gain_over_fixed4": _two_way_bootstrap(
            matrix, seed=2026080222
        ),
        "cross_suite_group_id_overlap": 0,
        "cross_suite_setup_hash_overlap": 0,
        "interpretation_guard": (
            f"The frozen probe model and stopping rule were unchanged. {len(seeds)} planner "
            f"seeds were evaluated on the same {len(ordered_groups)} non-protected setups, "
            f"balanced across {len(suites)} mutually group/hash-disjoint suites. Suite A's "
            "positive outcome motivated the planner-seed expansion, and the later matched "
            "suite was selected only from pre-treatment distance/stratum covariates; neither "
            "expansion caused model or rule retuning."
        ),
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
