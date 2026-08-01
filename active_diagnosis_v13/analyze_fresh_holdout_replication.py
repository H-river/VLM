#!/usr/bin/env python3
"""Pool frozen-rule effects across independent fresh non-protected setup suites."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "active_diagnosis_v13_fresh_holdout_replication_v2"


def _parse_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    return label, Path(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _bootstrap(values: np.ndarray, *, seed: int, samples: int = 10000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            float(np.mean(values[rng.integers(0, len(values), len(values))]))
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(np.mean(values)),
        "low": float(low),
        "high": float(high),
        "independent_setup_groups": int(len(values)),
        "samples": int(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=_parse_artifact, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.artifact) < 2:
        raise ValueError("at least two independent fresh-suite artifacts are required")
    artifacts = {
        label: json.loads(path.resolve().read_text())
        for label, path in args.artifact
    }
    if len(artifacts) != len(args.artifact):
        raise ValueError("fresh-suite labels must be unique")
    all_groups: set[str] = set()
    all_hashes: set[str] = set()
    rules = set()
    planner_root_seeds = set()
    rows = []
    per_suite = []
    for label, artifact in sorted(artifacts.items()):
        if artifact.get("protected_set_used") is not False or artifact.get(
            "selection_or_retuning_on_fresh_suite"
        ) is not False:
            raise ValueError(f"{label} is not one-shot non-protected evidence")
        groups = set(artifact["setup_independence"]["fresh_group_ids"])
        hashes = set(artifact["setup_independence"]["fresh_setup_hashes"])
        if all_groups & groups or all_hashes & hashes:
            raise ValueError(f"fresh suite {label} overlaps another fresh suite")
        all_groups |= groups
        all_hashes |= hashes
        rules.add(json.dumps(artifact["frozen_rule"], sort_keys=True))
        planner_root_seeds.add(int(artifact["planner_root_seed"]))
        effects = [
            float(row["frozen_sequential_minus_fixed4"])
            for row in artifact["group_effects"]
        ]
        if len(groups) != 30 or len(effects) != 30:
            raise ValueError(f"fresh suite {label} does not contain thirty group effects")
        rows.extend(effects)
        per_suite.append(
            {
                "suite": label,
                "setup_groups": len(groups),
                "fault_gain_over_fixed4": float(np.mean(effects)),
                "recoveries": int(artifact["recoveries"]),
                "regressions": int(artifact["regressions"]),
                "fault_success": artifact["policies"]["frozen_sequential"]["fault"][
                    "strict_success"
                ],
                "fixed4_fault_success": artifact["policies"]["fixed4"]["fault"][
                    "strict_success"
                ],
                "probe_gain_accuracy": artifact["probe_gain_classification"][
                    "overall_accuracy"
                ],
                "boundary_fault_gain_accuracy": artifact[
                    "probe_gain_classification"
                ]["boundary_fault_accuracy"],
            }
        )
    if len(rules) != 1:
        raise ValueError("fresh suites do not use one identical frozen rule")
    if len(planner_root_seeds) != 1:
        raise ValueError("fresh suites do not use one identical planner root seed")
    report = {
        "version": VERSION,
        "split": "independent_fresh_nonprotected_setup_holdouts",
        "protected_set_used": False,
        "selection_or_retuning_on_fresh_suites": False,
        "frozen_rule": json.loads(next(iter(rules))),
        "planner_root_seed": next(iter(planner_root_seeds)),
        "suites": per_suite,
        "suite_count": len(per_suite),
        "independent_setup_groups": len(all_groups),
        "all_suites_positive_over_fixed4": all(
            row["fault_gain_over_fixed4"] > 0.0 for row in per_suite
        ),
        "mean_probe_gain_accuracy": float(
            np.mean([float(row["probe_gain_accuracy"]) for row in per_suite])
        ),
        "minimum_probe_gain_accuracy": min(
            float(row["probe_gain_accuracy"]) for row in per_suite
        ),
        "pooled_group_bootstrap_gain_over_fixed4": _bootstrap(
            np.asarray(rows, dtype=np.float64), seed=2026080211
        ),
        "total_recoveries": sum(row["recoveries"] for row in per_suite),
        "total_regressions": sum(row["regressions"] for row in per_suite),
        "cross_suite_group_id_overlap": 0,
        "cross_suite_setup_hash_overlap": 0,
        "interpretation_guard": (
            "Both suites were generated and the analysis was preregistered before the first "
            "fresh-suite result completed. The frozen probe model and stopping rule are "
            "unchanged; all setups are non-protected and mutually group/hash disjoint."
        ),
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
