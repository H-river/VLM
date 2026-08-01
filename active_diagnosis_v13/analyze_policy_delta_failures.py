#!/usr/bin/env python3
"""Taxonomize matched success and failure changes between two dev policies."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_policy_delta_failure_taxonomy_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _category(reference: dict[str, Any], alternate: dict[str, Any]) -> str:
    left = bool(reference["strict_success"])
    right = bool(alternate["strict_success"])
    if left and right:
        return "both_success"
    if not left and not right:
        return "both_fail"
    return "alternate_recovery" if right else "alternate_regression"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(str(row["category"]) for row in rows)
    return {
        "episodes": len(rows),
        "categories": dict(sorted(categories.items())),
        "reference_success": float(np.mean([row["reference_success"] for row in rows])),
        "alternate_success": float(np.mean([row["alternate_success"] for row in rows])),
        "alternate_success_difference": float(
            np.mean([row["alternate_success"] for row in rows])
            - np.mean([row["reference_success"] for row in rows])
        ),
        "reference_saturation_episode_rate": float(
            np.mean([row["reference_saturation_count"] > 0 for row in rows])
        ),
        "alternate_saturation_episode_rate": float(
            np.mean([row["alternate_saturation_count"] > 0 for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--reference-policy", required=True)
    parser.add_argument("--alternate-policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    reference_rows = _jsonl(root / "control" / f"{args.reference_policy}.jsonl")
    alternate_rows = _jsonl(root / "control" / f"{args.alternate_policy}.jsonl")
    reference = {_key(row): row for row in reference_rows}
    alternate = {_key(row): row for row in alternate_rows}
    if len(reference_rows) != len(reference) or len(alternate_rows) != len(alternate):
        raise ValueError("duplicate control keys")
    if set(reference) != set(alternate):
        raise ValueError("control policies are not matched")
    if any(int(case_id.rsplit("_", 1)[1]) >= 10 for case_id, _ in reference):
        raise ValueError("protected case present")
    episodes = []
    for key in sorted(reference):
        left, right = reference[key], alternate[key]
        true_gain = float(left["evaluator_only_true_gain"])
        reference_gain = left.get("raw_gain_estimate", left.get("gain_belief"))
        if reference_gain is None:
            reference_gain = left.get("gain_belief")
        alternate_gain = right.get("raw_gain_estimate", right.get("gain_belief"))
        if alternate_gain is None:
            alternate_gain = right.get("gain_belief")
        episodes.append(
            {
                "episode_id": f"{key[0]}__g{key[1]:g}",
                "case_id": key[0],
                "group_id": str(left["group_id"]),
                "true_gain_evaluator_only": true_gain,
                "stratum": str(left["stratum"]),
                "category": _category(left, right),
                "reference_success": bool(left["strict_success"]),
                "alternate_success": bool(right["strict_success"]),
                "reference_final_distance": float(left["final_normalized_distance"]),
                "alternate_final_distance": float(right["final_normalized_distance"]),
                "reference_gain_estimate": (
                    None if reference_gain is None else float(reference_gain)
                ),
                "alternate_gain_estimate": (
                    None if alternate_gain is None else float(alternate_gain)
                ),
                "reference_absolute_gain_error": (
                    None if reference_gain is None else abs(float(reference_gain) - true_gain)
                ),
                "alternate_absolute_gain_error": (
                    None if alternate_gain is None else abs(float(alternate_gain) - true_gain)
                ),
                "reference_saturation_count": int(left["saturation_count"]),
                "alternate_saturation_count": int(right["saturation_count"]),
            }
        )
    fault = [row for row in episodes if row["true_gain_evaluator_only"] != 1.0]
    by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_gain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fault:
        by_stratum[row["stratum"]].append(row)
        by_gain[f"{row['true_gain_evaluator_only']:g}"].append(row)
    categories = sorted({str(row["category"]) for row in episodes})
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "reference_policy": args.reference_policy,
        "alternate_policy": args.alternate_policy,
        "overall": _summary(episodes),
        "fault": _summary(fault),
        "fault_by_stratum": {
            key: _summary(value) for key, value in sorted(by_stratum.items())
        },
        "fault_by_gain": {key: _summary(value) for key, value in sorted(by_gain.items())},
        "exact_episode_ids_by_category": {
            category: [
                row["episode_id"] for row in episodes if row["category"] == category
            ]
            for category in categories
        },
        "episodes": episodes,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
