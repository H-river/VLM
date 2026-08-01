#!/usr/bin/env python3
"""Evaluate the frozen visible stopping rule once on fresh non-protected setups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from active_diagnosis_v13.analyze_control_horizon_curve import (
    _bootstrap,
    _cap,
    _summary,
)
from active_diagnosis_v13.analyze_control_step_budget import _key, _read_jsonl
from active_diagnosis_v13.analyze_sequential_horizon_rule import _sequential_cap

VERSION = "active_diagnosis_v13_frozen_sequential_holdout_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _suite_identity(path: Path) -> tuple[dict[str, Any], set[str], set[str]]:
    suite = json.loads(path.resolve().read_text(encoding="utf-8"))
    groups = {str(row["group_id"]) for row in suite["cases"]}
    hashes = {str(row["setup_hash"]) for row in suite["cases"]}
    if len(groups) != len(suite["cases"]) or len(hashes) != len(suite["cases"]):
        raise ValueError(f"suite does not contain independent setups: {path}")
    return suite, groups, hashes


def _fault_keys(rows: dict[tuple[str, float], dict[str, Any]]) -> list[tuple[str, float]]:
    return [key for key in rows if key[1] != 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-budget8", type=Path, required=True)
    parser.add_argument("--rule-report", type=Path, required=True)
    parser.add_argument("--probe-model", type=Path, required=True)
    parser.add_argument("--fresh-suite", type=Path, required=True)
    parser.add_argument("--reference-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rule_report = json.loads(args.rule_report.resolve().read_text())
    if rule_report.get("protected_set_used") is not False:
        raise ValueError("the frozen rule report must be development-only")
    rule_block = rule_report["frozen_full_selection_seed_rule"]
    minimum = float(rule_block["minimum_last_step_improvement"])
    maximum_raw = rule_block["maximum_final_distance"]
    maximum = float("inf") if maximum_raw == "infinity" else float(maximum_raw)

    fresh_suite, fresh_groups, fresh_hashes = _suite_identity(args.fresh_suite)
    _, reference_groups, reference_hashes = _suite_identity(args.reference_suite)
    if fresh_groups & reference_groups or fresh_hashes & reference_hashes:
        raise ValueError("fresh suite overlaps the rule-selection reference suite")
    validation = fresh_suite.get("validation", {})
    if int(validation.get("prior_group_id_overlap", -1)) != 0 or int(
        validation.get("prior_setup_hash_overlap", -1)
    ) != 0:
        raise ValueError("fresh suite reports overlap with an excluded source")

    source_rows = _read_jsonl(args.source_budget8.resolve())
    source = {_key(row): row for row in source_rows}
    if len(source_rows) != 150 or len(source) != 150:
        raise ValueError("fresh holdout requires exactly 150 unique gain episodes")
    if {str(row["group_id"]) for row in source.values()} != fresh_groups:
        raise ValueError("source arm does not exactly cover the fresh suite")
    if any(str(row["mode"]) != "probe_replan" for row in source.values()):
        raise ValueError("fresh holdout source is not a probe-replan arm")
    if any(
        str(row["gain_prediction_source"])
        != "frozen_full_classifier_fresh_nonprotected"
        for row in source.values()
    ):
        raise ValueError("fresh holdout source does not use the frozen full classifier")
    policy_names = {str(row["policy_name"]) for row in source.values()}
    if len(policy_names) != 1 or "budget8" not in next(iter(policy_names)):
        raise ValueError(f"fresh holdout source lacks budget-eight identity: {policy_names}")
    if max(len(row["trace"]) for row in source.values()) != 8:
        raise ValueError("fresh holdout source contains no complete eight-control trace")
    planner_root_seeds = {int(row["planner_root_seed"]) for row in source.values()}
    if len(planner_root_seeds) != 1:
        raise ValueError("fresh holdout source mixes planner root seeds")

    bundle = joblib.load(args.probe_model.resolve())
    prediction_mismatches = []
    for key, row in source.items():
        features = np.asarray(
            row["probe_record"]["policy_record"]["feature_vector"], dtype=np.float64
        )
        prediction = float(bundle["full_model"].predict(features[None, :])[0])
        if not np.isclose(prediction, float(row["raw_gain_estimate"]), rtol=0.0, atol=0.0):
            prediction_mismatches.append(f"{key[0]}__g{key[1]:g}")
    if prediction_mismatches:
        raise ValueError(
            f"fresh arm does not replay the frozen probe model: {prediction_mismatches[:3]}"
        )

    fixed = {
        horizon: {key: _cap(dict(row), horizon) for key, row in source.items()}
        for horizon in (4, 6, 8)
    }
    adaptive = {
        key: _sequential_cap(
            row,
            minimum_improvement=minimum,
            maximum_distance=maximum,
        )
        for key, row in source.items()
    }
    fault = _fault_keys(source)
    boundary_fault = [
        key
        for key in fault
        if str(source[key]["stratum"]) == "reachable_boundary_or_clipping"
    ]
    fixed8_recoveries = {
        key
        for key in fault
        if not bool(fixed[4][key]["strict_success"])
        and bool(fixed[8][key]["strict_success"])
    }
    sequential_recoveries = {
        key
        for key in fault
        if not bool(fixed[4][key]["strict_success"])
        and bool(adaptive[key]["strict_success"])
    }
    regressions = {
        key
        for key in fault
        if bool(fixed[4][key]["strict_success"])
        and not bool(adaptive[key]["strict_success"])
    }
    episode_id = lambda key: f"{key[0]}__g{key[1]:g}"
    group_effects = []
    for group in sorted(fresh_groups):
        keys = [
            key
            for key in fault
            if str(source[key]["group_id"]) == group
        ]
        if len(keys) != 4:
            raise ValueError(f"fresh group does not contain four fault episodes: {group}")
        group_effects.append(
            {
                "group_id": group,
                "stratum": str(source[keys[0]]["stratum"]),
                "fixed4_success": float(
                    np.mean([bool(fixed[4][key]["strict_success"]) for key in keys])
                ),
                "fixed6_success": float(
                    np.mean([bool(fixed[6][key]["strict_success"]) for key in keys])
                ),
                "fixed8_success": float(
                    np.mean([bool(fixed[8][key]["strict_success"]) for key in keys])
                ),
                "frozen_sequential_success": float(
                    np.mean([bool(adaptive[key]["strict_success"]) for key in keys])
                ),
                "frozen_sequential_minus_fixed4": float(
                    np.mean(
                        [
                            float(adaptive[key]["strict_success"])
                            - float(fixed[4][key]["strict_success"])
                            for key in keys
                        ]
                    )
                ),
            }
        )
    report = {
        "version": VERSION,
        "split": "fresh_nonprotected_setup_holdout",
        "protected_set_used": False,
        "selection_or_retuning_on_fresh_suite": False,
        "frozen_rule": rule_block,
        "rule_source": str(args.rule_report.resolve()),
        "rule_source_sha256": _sha256(args.rule_report),
        "probe_model": str(args.probe_model.resolve()),
        "probe_model_sha256": _sha256(args.probe_model),
        "probe_prediction_replay_exact": True,
        "probe_gain_classification": {
            "overall_accuracy": float(
                np.mean(
                    [bool(row["gain_classification_correct"]) for row in source.values()]
                )
            ),
            "fault_accuracy": float(
                np.mean(
                    [bool(source[key]["gain_classification_correct"]) for key in fault]
                )
            ),
            "boundary_fault_accuracy": float(
                np.mean(
                    [
                        bool(source[key]["gain_classification_correct"])
                        for key in boundary_fault
                    ]
                )
            ),
        },
        "source_budget8": str(args.source_budget8.resolve()),
        "source_budget8_sha256": _sha256(args.source_budget8),
        "planner_root_seed": next(iter(planner_root_seeds)),
        "setup_independence": {
            "fresh_groups": len(fresh_groups),
            "reference_groups": len(reference_groups),
            "group_id_overlap": 0,
            "setup_hash_overlap": 0,
            "fresh_group_ids": sorted(fresh_groups),
            "fresh_setup_hashes": sorted(fresh_hashes),
            "fresh_suite_validation": validation,
        },
        "group_effects": group_effects,
        "policies": {
            "fixed4": _summary(list(fixed[4].values())),
            "fixed6": _summary(list(fixed[6].values())),
            "fixed8": _summary(list(fixed[8].values())),
            "frozen_sequential": _summary(list(adaptive.values())),
        },
        "frozen_sequential_gain_over_fixed4": _bootstrap(
            fixed[4], adaptive, seed=2026080201
        ),
        "frozen_sequential_gain_over_fixed6": _bootstrap(
            fixed[6], adaptive, seed=2026080202
        ),
        "recoveries": len(sequential_recoveries),
        "regressions": len(regressions),
        "fixed8_recoveries": len(fixed8_recoveries),
        "fixed8_recoveries_retained": len(fixed8_recoveries & sequential_recoveries),
        "exact_ids": {
            "fixed4_fault_failures": sorted(
                episode_id(key)
                for key in fault
                if not bool(fixed[4][key]["strict_success"])
            ),
            "frozen_sequential_fault_failures": sorted(
                episode_id(key)
                for key in fault
                if not bool(adaptive[key]["strict_success"])
            ),
            "frozen_sequential_recoveries": sorted(map(episode_id, sequential_recoveries)),
            "frozen_sequential_regressions": sorted(map(episode_id, regressions)),
            "fixed8_recoveries_missed": sorted(
                map(episode_id, fixed8_recoveries - sequential_recoveries)
            ),
        },
        "interpretation_guard": (
            "The rule and probe model were frozen on the original development suite. "
            "This one-shot fresh-suite evaluation uses group- and setup-hash-disjoint, "
            "non-protected setups and performs no threshold or policy selection."
        ),
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
