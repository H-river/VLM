#!/usr/bin/env python3
"""Validate completeness, matching, finiteness, and leakage of v13 artifacts."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.contracts import assert_policy_visible

VERSION = "active_diagnosis_v13_artifact_validation_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _probe_rows(root: Path) -> list[dict[str, Any]]:
    by_id = {}
    for path in sorted((root / "probes").glob("records*.jsonl")):
        for row in _jsonl(path):
            if row["record_id"] in by_id:
                raise ValueError(f"duplicate probe record: {row['record_id']}")
            by_id[row["record_id"]] = row
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    probes = _probe_rows(root)
    feature_lengths: defaultdict[tuple[str, float], set[int]] = defaultdict(set)
    combo_counts: Counter[tuple[str, float]] = Counter()
    group_ids = set()
    invalid_visible = []
    finite = True
    for row in probes:
        try:
            assert_policy_visible(row["policy_record"])
            for step in row["step_records"]:
                assert_policy_visible(step["visible_input"])
        except ValueError as error:
            invalid_visible.append(str(error))
        vector = np.asarray(row["policy_record"]["feature_vector"], dtype=np.float64)
        finite = finite and bool(np.isfinite(vector).all())
        key = (str(row["design"]), float(row["fraction"]))
        feature_lengths[key].add(len(vector))
        combo_counts[key] += 1
        group_ids.add(str(row["group_id"]))
    controls = {
        mode: _jsonl(root / "control" / f"{mode}.jsonl")
        for mode in ("direct", "oracle_known", "probe_replan")
    }
    duplicate_controls = {}
    for mode, rows in controls.items():
        ids = [row["record_id"] for row in rows]
        duplicate_controls[mode] = len(ids) - len(set(ids))
        for row in rows:
            for step in row["trace"]:
                try:
                    assert_policy_visible(step["visible_plan_input"])
                except ValueError as error:
                    invalid_visible.append(str(error))
    direct_nominal = {
        row["case_id"]: row
        for row in controls["direct"]
        if float(row["evaluator_only_true_gain"]) == 1.0
    }
    oracle_nominal = {
        row["case_id"]: row
        for row in controls["oracle_known"]
        if float(row["evaluator_only_true_gain"]) == 1.0
    }
    matched_nominal_cases = sorted(set(direct_nominal) & set(oracle_nominal))
    nominal_max_distance_difference = max(
        [
            abs(
                float(direct_nominal[case]["final_normalized_distance"])
                - float(oracle_nominal[case]["final_normalized_distance"])
            )
            for case in matched_nominal_cases
        ]
        or [0.0]
    )
    report = {
        "version": VERSION,
        "development_groups": len(group_ids),
        "probe_records": len(probes),
        "probe_combo_counts": {
            f"{design}@{fraction:g}": count
            for (design, fraction), count in sorted(combo_counts.items())
        },
        "feature_lengths_consistent_per_design": all(
            len(lengths) == 1 for lengths in feature_lengths.values()
        ),
        "all_features_finite": finite,
        "policy_visibility_violations": invalid_visible,
        "control_records": {mode: len(rows) for mode, rows in controls.items()},
        "duplicate_control_records": duplicate_controls,
        "matched_nominal_cases": len(matched_nominal_cases),
        "direct_oracle_nominal_max_final_distance_difference": nominal_max_distance_difference,
        "protected_case_suffixes_present": sorted(
            {
                int(group.rsplit("_", 1)[1])
                for group in group_ids
                if int(group.rsplit("_", 1)[1]) >= 10
            }
        ),
        "gate_a_present": (root / "gate_a_diagnosis.json").exists(),
    }
    errors = []
    if invalid_visible or not finite:
        errors.append("visible features failed leakage or finiteness validation")
    if any(len(lengths) != 1 for lengths in feature_lengths.values()):
        errors.append("feature length varies inside a probe design")
    if report["protected_case_suffixes_present"]:
        errors.append("protected cases are present in development artifacts")
    if any(duplicate_controls.values()):
        errors.append("duplicate control records")
    if nominal_max_distance_difference > 1e-12:
        errors.append("direct and oracle-known gain=1 controls are not matched")
    if args.require_complete:
        if len(probes) != 2400 or len(group_ids) != 30:
            errors.append("development probe sweep is incomplete")
        if any(count != 150 for count in combo_counts.values()) or len(combo_counts) != 16:
            errors.append("probe design/fraction cells are incomplete")
        if any(len(rows) != 150 for rows in controls.values()):
            errors.append("one or more development control modes are incomplete")
        if len(matched_nominal_cases) != 30:
            errors.append("nominal matching does not cover all development cases")
    if args.require_gate and not report["gate_a_present"]:
        errors.append("Gate A diagnosis is absent")
    report["passes"] = not errors
    report["errors"] = errors
    output = root / "artifact_validation.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
