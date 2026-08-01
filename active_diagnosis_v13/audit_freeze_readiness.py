#!/usr/bin/env python3
"""Audit development evidence and freeze-chain readiness without protected reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.run_gate_a import _load_config, select_gate_branch


VERSION = "active_diagnosis_v13_freeze_readiness_audit_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    root = args.gate_dir.resolve()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    probe_summary_path = root / "probes/probe_summary.json"
    probe_summary = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    selected = probe_summary["selected"]
    selected_model = Path(selected["classifier_bundle"]).resolve()
    controls = {
        mode: _jsonl(root / "control" / f"{mode}.jsonl")
        for mode in ("direct", "oracle_known", "probe_replan")
    }
    nominal_gain = float(config["fault"]["nominal_gain"])
    direct_nominal = [
        row
        for row in controls["direct"]
        if float(row["evaluator_only_true_gain"]) == nominal_gain
    ]
    fault = {
        mode: [
            row
            for row in rows
            if float(row["evaluator_only_true_gain"]) != nominal_gain
        ]
        for mode, rows in controls.items()
    }
    rate = lambda rows: float(np.mean([bool(row["strict_success"]) for row in rows]))
    impact = rate(direct_nominal) - rate(fault["direct"])
    recovery = rate(fault["oracle_known"]) - rate(fault["direct"])
    control_value = rate(fault["probe_replan"]) - rate(fault["direct"])
    accuracy = float(selected["development_gain_classification_accuracy"])
    thresholds = config["gate_a"]
    checks = (
        impact >= float(thresholds["meaningful_fault_impact_success_drop"]),
        recovery >= float(thresholds["meaningful_oracle_recovery_success_gain"]),
        accuracy >= float(thresholds["minimum_probe_gain_classification_accuracy"]),
        control_value >= float(thresholds["minimum_probe_replan_success_gain"]),
    )
    predicted_branch, predicted_reason = select_gate_branch(*checks)
    errors = []
    if any(len(rows) != 150 for rows in controls.values()):
        errors.append("primary development control arms are incomplete")
    if _sha256(selected_model) != selected["classifier_bundle_sha256"]:
        errors.append("selected preregistered probe model hash mismatch")
    validation_path = root / "artifact_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passes"):
        errors.append("development artifact validation does not pass")
    if (root / "gate_a_diagnosis.json").exists() or (root / "frozen_decision.json").exists():
        errors.append("formal Gate artifacts already exist before one-shot formalization")
    protected_path = run / "protected"
    if protected_path.exists() and any(protected_path.rglob("*")):
        errors.append("protected artifacts exist before formal freeze")

    ablations = json.loads(
        (root / "ablations/probe_feature_ablation_report.json").read_text(
            encoding="utf-8"
        )
    )
    nores = ablations["ablations"]["no_residual_history"]
    nores_model = Path(nores["classifier_bundle"]).resolve()
    comparison = json.loads(
        (root / "estimator_policy_comparison.json").read_text(encoding="utf-8")
    )
    nores_control = next(
        row
        for row in comparison["comparisons"]
        if row["policy"] == "probe_symmetric_pair_f0p1_no_residual"
    )
    if _sha256(nores_model) != nores["classifier_bundle_sha256"]:
        errors.append("Branch-A no-residual model hash mismatch")
    if float(nores["group_out_of_fold_accuracy_95"]["estimate"]) < 0.8:
        errors.append("Branch-A refinement does not clear observability")
    if float(nores_control["fault_success_gain_over_direct"]) < 0.05:
        errors.append("Branch-A refinement does not clear control value")
    robustness_path = root / "branch_a_refinement_seven_seed.json"
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    if robustness.get("protected_set_used", True) or not robustness[
        "control_value"
    ].get("passes_five_points_every_seed", False):
        errors.append("required seven-seed Branch-A control robustness does not pass")
    for component in ("fault_impact", "oracle_recovery"):
        if not robustness.get("gate_components", {}).get(component, {}).get(
            "passes_five_points_every_seed", False
        ):
            errors.append(f"seven-seed Branch-A {component} robustness does not pass")
    matrix_path = root / "probe_design_control_summary.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if (
        matrix.get("protected_set_used", True)
        or int(matrix.get("completed_cells", -1)) != 16
        or int(matrix.get("expected_cells", -2)) != 16
    ):
        errors.append("strict 16-cell development probe matrix is incomplete")
    command_audit_path = root / "reproduction_command_audit.json"
    command_audit = json.loads(command_audit_path.read_text(encoding="utf-8"))
    if command_audit.get("protected_set_used", True) or not command_audit.get(
        "passes", False
    ):
        errors.append("reproduction-command audit does not pass")
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_read": False,
        "protected_artifacts_present": protected_path.exists(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_hashes_revalidated": True,
        "critical_evidence_hashes": {
            "probe_summary": _sha256(probe_summary_path),
            "preregistered_probe_model": _sha256(selected_model),
            "artifact_validation": _sha256(validation_path),
            "branch_a_ablation_report": _sha256(
                root / "ablations/probe_feature_ablation_report.json"
            ),
            "branch_a_policy_comparison": _sha256(
                root / "estimator_policy_comparison.json"
            ),
            "branch_a_seven_seed_robustness": _sha256(robustness_path),
            "probe_design_control_summary": _sha256(matrix_path),
            "reproduction_command_audit": _sha256(command_audit_path),
        },
        "primary_control_records": {
            mode: len(rows) for mode, rows in controls.items()
        },
        "preregistered_gate_preview": {
            "A_fault_impact": impact,
            "B_oracle_recovery": recovery,
            "C_observability": accuracy,
            "D_control_value": control_value,
            "passes": {key: bool(value) for key, value in zip("ABCD", checks)},
            "predicted_branch": predicted_branch,
            "predicted_reason": predicted_reason,
            "role": "readiness preview only; formal classification remains clock-guarded",
        },
        "branch_a_refinement": {
            "gain_accuracy": float(
                nores["group_out_of_fold_accuracy_95"]["estimate"]
            ),
            "control_value": float(nores_control["fault_success_gain_over_direct"]),
            "model": str(nores_model),
            "model_sha256": _sha256(nores_model),
            "seven_seed_control_value": robustness["control_value"],
            "seven_seed_gate_components": robustness["gate_components"],
        },
        "probe_control_matrix": {
            "completed_cells": int(matrix["completed_cells"]),
            "expected_cells": int(matrix["expected_cells"]),
        },
        "reproduction_command_audit": {
            "commands": int(command_audit["command_count"]),
            "passes": bool(command_audit["passes"]),
        },
        "formal_gate_artifacts_absent": not (root / "gate_a_diagnosis.json").exists()
        and not (root / "frozen_decision.json").exists(),
        "ready": not errors,
        "errors": errors,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
