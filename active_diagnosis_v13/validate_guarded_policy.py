#!/usr/bin/env python3
"""Verify a guarded hybrid replay exactly selects one of two cached policies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_guarded_policy_replay_validation_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--discrete-policy", required=True)
    parser.add_argument("--continuous-policy", required=True)
    parser.add_argument("--hybrid-policy", required=True)
    parser.add_argument("--discrete-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve() / "control"
    names = {
        "discrete": args.discrete_policy,
        "continuous": args.continuous_policy,
        "hybrid": args.hybrid_policy,
    }
    rows = {
        role: {_key(row): row for row in _jsonl(root / f"{name}.jsonl")}
        for role, name in names.items()
    }
    if any(len(value) != 150 for value in rows.values()):
        raise ValueError("all three guarded-policy arms must contain 150 episodes")
    key_sets = [set(value) for value in rows.values()]
    if any(value != key_sets[0] for value in key_sets[1:]):
        raise ValueError("guarded-policy arms are not episode-matched")
    mismatches = []
    expected_source_counts = {"discrete": 0, "continuous": 0}
    fields = (
        "strict_success",
        "total_additional_steps",
        "saturation_count",
        "constraint_violation_count",
    )
    float_fields = ("gain_belief", "final_normalized_distance")
    for key in sorted(key_sets[0]):
        discrete = rows["discrete"][key]
        source = (
            "continuous"
            if float(discrete["gain_belief"]) >= args.discrete_threshold
            else "discrete"
        )
        expected_source_counts[source] += 1
        expected = rows[source][key]
        actual = rows["hybrid"][key]
        different = [field for field in fields if actual[field] != expected[field]]
        different.extend(
            field
            for field in float_fields
            if not np.isclose(
                float(actual[field]),
                float(expected[field]),
                rtol=0.0,
                atol=1e-12,
            )
        )
        if different:
            mismatches.append(
                {
                    "case_id": key[0],
                    "true_gain_evaluator_only": key[1],
                    "expected_source": source,
                    "different_fields": different,
                }
            )
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "discrete_policy": args.discrete_policy,
        "continuous_policy": args.continuous_policy,
        "hybrid_policy": args.hybrid_policy,
        "discrete_threshold": args.discrete_threshold,
        "matched_episodes": 150,
        "expected_source_counts": expected_source_counts,
        "mismatches": mismatches,
        "passes": not mismatches,
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
