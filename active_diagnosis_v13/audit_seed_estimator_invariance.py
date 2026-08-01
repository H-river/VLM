#!/usr/bin/env python3
"""Verify that matched planner seeds do not change probe features or estimates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_seed_estimator_invariance_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _feature(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        row["probe_record"]["policy_record"]["feature_vector"], dtype=np.float64
    )


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--base-policy-name", required=True)
    parser.add_argument("--seed-policy-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve() / "control"
    base_seed = int(json.loads(args.config.resolve().read_text())["root_seed"])
    seeds = list(dict.fromkeys(args.seeds))
    if base_seed not in seeds or len(seeds) < 2:
        raise ValueError("at least two seeds including the configured base are required")
    names = {
        seed: (
            args.base_policy_name
            if seed == base_seed
            else f"{args.seed_policy_prefix}_{seed}"
        )
        for seed in seeds
    }
    rows_by_seed = {
        seed: _jsonl(root / f"{name}.jsonl") for seed, name in names.items()
    }
    maps = {
        seed: {_key(row): row for row in rows}
        for seed, rows in rows_by_seed.items()
    }
    reference_keys = set(maps[base_seed])
    for seed, rows in rows_by_seed.items():
        if len(rows) != 150 or len(maps[seed]) != 150 or set(maps[seed]) != reference_keys:
            raise ValueError(f"seed {seed} is incomplete, duplicated, or unmatched")
    if any(int(case_id.rsplit("_", 1)[1]) >= 10 for case_id, _ in reference_keys):
        raise ValueError("protected case present in estimator invariance audit")

    mismatches: list[dict[str, Any]] = []
    maximum_feature_delta = 0.0
    maximum_gain_belief_delta = 0.0
    for seed in seeds:
        if seed == base_seed:
            continue
        for key in sorted(reference_keys):
            reference = maps[base_seed][key]
            current = maps[seed][key]
            feature_delta = float(np.max(np.abs(_feature(reference) - _feature(current))))
            gain_delta = abs(float(reference["gain_belief"]) - float(current["gain_belief"]))
            maximum_feature_delta = max(maximum_feature_delta, feature_delta)
            maximum_gain_belief_delta = max(maximum_gain_belief_delta, gain_delta)
            scalar_equal = all(
                _optional_float(reference, field) == _optional_float(current, field)
                for field in (
                    "raw_gain_estimate",
                    "gain_estimate_confidence",
                    "gain_belief",
                )
            )
            if feature_delta > 0 or not scalar_equal:
                mismatches.append(
                    {
                        "seed": seed,
                        "episode_id": f"{key[0]}__g{key[1]:g}",
                        "maximum_feature_delta": feature_delta,
                        "reference_gain_belief": reference["gain_belief"],
                        "current_gain_belief": current["gain_belief"],
                    }
                )
    report = {
        "version": VERSION,
        "role": "development_only_matched_planner_seed_estimator_invariance",
        "split": "development_only",
        "protected_set_used": False,
        "base_seed": base_seed,
        "seeds": seeds,
        "policies": names,
        "episodes_per_seed": 150,
        "comparisons": 150 * (len(seeds) - 1),
        "probe_feature_and_estimator_outputs_exactly_invariant": not mismatches,
        "maximum_absolute_probe_feature_delta": maximum_feature_delta,
        "maximum_absolute_gain_belief_delta": maximum_gain_belief_delta,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "interpretation": (
            "when this passes, cross-seed outcome variation isolates planner sampling rather than estimator variation"
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
