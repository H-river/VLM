#!/usr/bin/env python3
"""Freeze the successful development-only Branch-A estimator refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


VERSION = "active_diagnosis_v13_branch_a_resolution_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_formal_gate(
    root: Path, diagnosis_path: Path, diagnosis: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    if diagnosis.get("selected_primary_branch") != "A":
        raise ValueError("Branch-A resolution requires the formal Gate A branch to be A")
    if diagnosis.get("split") != "development_only" or diagnosis.get(
        "protected_set_used", True
    ):
        raise ValueError("Branch-A resolution requires development-only Gate evidence")
    formal_path = (root / "frozen_decision.json").resolve()
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    if not formal.get("frozen") or formal.get("selected_primary_branch") != "A":
        raise ValueError("formal frozen decision does not consistently select Branch A")
    if formal.get("protected_set_used_for_selection", True):
        raise ValueError("formal frozen decision is not development-only")
    if formal.get("gate_a_diagnosis_sha256") != _sha256(diagnosis_path):
        raise ValueError("formal frozen decision does not match Gate A diagnosis")
    return formal_path, formal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--ablation-report", type=Path, required=True)
    parser.add_argument("--policy-comparison", type=Path, required=True)
    parser.add_argument("--robustness-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-name", default="probe_symmetric_pair_f0p1_no_residual")
    parser.add_argument("--ablation", default="no_residual_history")
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    diagnosis_path = root / "gate_a_diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    formal_path, formal_freeze = _validate_formal_gate(root, diagnosis_path, diagnosis)
    ablations = json.loads(args.ablation_report.resolve().read_text(encoding="utf-8"))
    if ablations.get("protected_set_used", True):
        raise ValueError("Branch-A ablation report is not development-only")
    candidate = ablations["ablations"][args.ablation]
    comparison = json.loads(args.policy_comparison.resolve().read_text(encoding="utf-8"))
    if comparison.get("protected_set_used", True):
        raise ValueError("Branch-A policy comparison is not development-only")
    control = next(
        row for row in comparison["comparisons"] if row["policy"] == args.policy_name
    )
    robustness_path = args.robustness_report.resolve()
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    if robustness.get("protected_set_used", True):
        raise ValueError("Branch-A robustness report is not development-only")
    if not robustness["control_value"].get("passes_five_points_every_seed", False):
        raise ValueError("Branch-A refinement does not clear control value in every seed")
    accuracy = float(candidate["group_out_of_fold_accuracy_95"]["estimate"])
    control_value = float(control["fault_success_gain_over_direct"])
    if accuracy < 0.8 or control_value < 0.05:
        raise ValueError("Branch-A refinement does not clear observability and control gates")
    model_path = Path(candidate["classifier_bundle"]).resolve()
    if _sha256(model_path) != candidate["classifier_bundle_sha256"]:
        raise ValueError("Branch-A classifier hash mismatch")
    selected_probe = {
        **formal_freeze["selected_probe"],
        "classifier_bundle": str(model_path),
        "classifier_bundle_sha256": _sha256(model_path),
        "development_gain_classification_accuracy": accuracy,
        "estimator_refinement": args.ablation,
        "retained_feature_count": int(candidate["retained_feature_count"]),
        "removed_feature_count": int(candidate["removed_feature_count"]),
    }
    resolution: dict[str, Any] = {
        "version": VERSION,
        "frozen": True,
        "selected_primary_branch": "A",
        "branch_reason": (
            "formal full-feature probe was borderline; development-only no-residual-history "
            "refinement clears both observability and closed-loop control gates"
        ),
        "formal_gate_a_diagnosis": str(diagnosis_path),
        "formal_gate_a_diagnosis_sha256": _sha256(diagnosis_path),
        "formal_freeze": str(formal_path),
        "formal_freeze_sha256": _sha256(formal_path),
        "config_sha256": formal_freeze["config_sha256"],
        "selected_probe": selected_probe,
        "development_evidence": {
            "gain_classification_accuracy": accuracy,
            "gain_classification_accuracy_95": candidate[
                "group_out_of_fold_accuracy_95"
            ],
            "fault_success": float(control["fault_success"]),
            "fault_success_gain_over_direct": control_value,
            "fault_success_gain_bootstrap_95": control[
                "matched_group_bootstrap_95"
            ],
            "matched_recoveries": int(control["matched_direct_failure_recoveries"]),
            "matched_regressions": int(control["matched_direct_success_regressions"]),
            "planner_seed_robustness": robustness["control_value"],
            "planner_seed_robustness_report": str(robustness_path),
            "planner_seed_robustness_report_sha256": _sha256(robustness_path),
        },
        "protected_set_used_for_selection": False,
        "protected_test_permitted_once": True,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(resolution, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(resolution, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
