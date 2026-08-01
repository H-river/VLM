#!/usr/bin/env python3
"""Audit every required v4 artifact, provenance invariant, and final result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.scripts.verify_frozen_baseline import (
    DEFAULT_MANIFEST as QWEN_FREEZE_MANIFEST,
)
from Qwen_orchestration.scripts.verify_frozen_baseline import verify as verify_qwen
from control_rebuild_v4.assess_validation import THRESHOLDS as VALIDATION_THRESHOLDS
from control_rebuild_v4.orchestrated_runtime import ROUTE_ARTIFACTS, ROUTE_BACKENDS
from control_rebuild_v4.write_candidate_manifest import (
    DEFAULT_MEASUREMENT_V3_RUN,
    DIRECTION_ARTIFACT,
    QWEN_REGISTRY,
    QWEN_SCHEMA,
)

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v4_numerical"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_MEASUREMENT_V4_RUN = (
    REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v4_one_seed"
)
PRIMARY_RATES = (
    "strict_all_five_success",
    "target_success_feasible",
    "physical_target_success_feasible",
    "status_accuracy",
    "status_macro_f1",
    "minimum_movement_exact_feasible",
    "final_physical_success",
    "rate",
)
ROUTE_TASK_TYPES = {
    "measure_beam_profile_v1": "beam_profile_measurement",
    "predict_direction_from_state_v1": "direction_prediction",
    "predict_direction_from_image_v1": "direction_prediction",
    "predict_forward_from_state_v1": "forward_prediction",
    "predict_forward_from_image_v1": "forward_prediction",
    "select_inverse_action_from_states_v1": "inverse_control",
    "select_inverse_action_from_images_v1": "inverse_control",
}
EXPECTED_VALIDATION_GATE_COUNT = 26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--measurement-v4-run",
        type=Path,
        default=DEFAULT_MEASUREMENT_V4_RUN,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_contract_evidence(data_dir: Path) -> dict[str, Any]:
    failures = []
    checksum_path = data_dir / "checksums.sha256"
    checksum_text = checksum_path.read_text(encoding="utf-8")
    relatives = []
    for line_number, line in enumerate(checksum_text.splitlines(), start=1):
        if not line:
            continue
        if "  " not in line:
            failures.append(f"malformed checksum line: {line_number}")
            continue
        expected, relative = line.split("  ", 1)
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not relative
        ):
            failures.append(f"malformed checksum fields: {line_number}")
            continue
        relatives.append(relative)
        path = data_dir / relative
        if not path.is_file():
            failures.append(f"missing checksum target: {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
    relative_counts = Counter(relatives)
    duplicate_relatives = sorted(
        relative for relative, count in relative_counts.items() if count > 1
    )
    expected_relatives = {
        str(path.relative_to(data_dir))
        for path in data_dir.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256", "manifest.json"}
    }
    listed_relatives = set(relatives)
    missing_relatives = sorted(expected_relatives - listed_relatives)
    extra_relatives = sorted(listed_relatives - expected_relatives)
    if duplicate_relatives:
        failures.append(f"duplicate checksum targets: {duplicate_relatives[:5]}")
    if missing_relatives:
        failures.append(f"unlisted dataset files: {missing_relatives[:5]}")
    if extra_relatives:
        failures.append(f"extra checksum targets: {extra_relatives[:5]}")
    return {
        "failures": failures,
        "line_count": len(relatives),
        "unique_target_count": len(listed_relatives),
        "expected_file_count": len(expected_relatives),
        "duplicate_targets": duplicate_relatives,
        "missing_targets": missing_relatives,
        "extra_targets": extra_relatives,
        "checksum_manifest_sha256": hashlib.sha256(
            checksum_text.encode("utf-8")
        ).hexdigest(),
    }


def rates_in_range(value: Any, path: str = "root") -> list[str]:
    failures = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in PRIMARY_RATES and isinstance(item, (int, float)):
                if not math.isfinite(float(item)) or not 0.0 <= float(item) <= 1.0:
                    failures.append(f"invalid rate at {child}: {item}")
            failures.extend(rates_in_range(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(rates_in_range(item, f"{path}[{index}]"))
    return failures


def pinned_entry_failure(
    label: str,
    entry: Any,
    expected_path: Path,
) -> str | None:
    if not isinstance(entry, dict):
        return f"{label}: pin is not an object"
    if set(entry) != {"path", "size", "sha256"}:
        return f"{label}: pin fields differ"
    path_value = entry.get("path")
    if not isinstance(path_value, str) or not path_value:
        return f"{label}: pin path is missing"
    path = Path(path_value).resolve()
    expected = expected_path.resolve()
    if path != expected:
        return f"{label}: path differs"
    if not path.is_file():
        return f"{label}: target is missing"
    if not isinstance(entry.get("size"), int) or path.stat().st_size != entry["size"]:
        return f"{label}: size differs"
    if not isinstance(entry.get("sha256"), str) or sha256(path) != entry["sha256"]:
        return f"{label}: sha256 differs"
    return None


def candidate_manifest_contract_failures(
    manifest: Mapping[str, Any],
    run_dir: Path,
    measurement_run: Path,
) -> list[str]:
    failures = []
    expected_artifacts = {
        "measurement_v3": DEFAULT_MEASUREMENT_V3_RUN / "measurement_v3.pt",
        "measurement_calibrator_v4": measurement_run / "measurement_calibrator_v4.pt",
        "forward_v4": run_dir / "forward_physics_residual_v4.pt",
        "inverse_v4": run_dir / "inverse_control_v4.pt",
        "visual_scorer_v4": run_dir / "visual_sensor_scorer_v4_integrated.pt",
        "direction_v1": DIRECTION_ARTIFACT,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        failures.append("artifacts: expected an object")
        artifacts = {}
    if set(artifacts) != set(expected_artifacts):
        failures.append("artifacts: exact six-key set differs")
    for key, expected_path in expected_artifacts.items():
        failure = pinned_entry_failure(
            f"artifacts.{key}",
            artifacts.get(key),
            expected_path,
        )
        if failure is not None:
            failures.append(failure)

    frozen_qwen = manifest.get("frozen_qwen")
    if not isinstance(frozen_qwen, dict):
        failures.append("frozen_qwen: expected an object")
        frozen_qwen = {}
    for key, expected_path in (
        ("registry", QWEN_REGISTRY),
        ("decision_schema", QWEN_SCHEMA),
    ):
        failure = pinned_entry_failure(
            f"frozen_qwen.{key}",
            frozen_qwen.get(key),
            expected_path,
        )
        if failure is not None:
            failures.append(failure)

    expected_routes = {
        route: {
            "backend": ROUTE_BACKENDS[route],
            "artifacts": list(ROUTE_ARTIFACTS[route]),
        }
        for route in ROUTE_BACKENDS
    }
    if manifest.get("routes") != expected_routes:
        failures.append("routes: exact route, backend, or artifact mapping differs")
    return failures


def real_example_contract_failures(
    examples: Mapping[str, Any],
    expected_splits: set[str],
) -> list[str]:
    failures = []
    contracts = {
        "forward": ("strict_all_five_success", "group_id"),
        "inverse": ("physical_target_reached", "request_id"),
        "measurement": ("strict_all_five_success", "state_id"),
        "visual_inverse": ("physical_target_reached", "group_id"),
    }
    definitions = examples.get("definitions")
    if not isinstance(definitions, dict) or set(definitions) != set(contracts):
        failures.append("definitions: exact four-task set differs")
        definitions = {}
    tasks = examples.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(contracts):
        failures.append("tasks: exact four-task set differs")
        tasks = {}
    for task, (flag, identifier) in contracts.items():
        if not isinstance(definitions.get(task), str) or not definitions[task]:
            failures.append(f"{task}: definition is missing")
        outcomes = tasks.get(task)
        if not isinstance(outcomes, dict) or set(outcomes) != {"success", "failure"}:
            failures.append(f"{task}: exact success/failure set differs")
            outcomes = {}
        for outcome, expected_flag in (("success", True), ("failure", False)):
            record = outcomes.get(outcome)
            label = f"{task}.{outcome}"
            if not isinstance(record, dict):
                failures.append(f"{label}: record is missing")
                continue
            if record.get(flag) is not expected_flag:
                failures.append(f"{label}: metric outcome differs")
            if not isinstance(record.get(identifier), (str, int)):
                failures.append(f"{label}: identifier is missing")
            if record.get("source_split") not in expected_splits:
                failures.append(f"{label}: source split differs")
    return failures


def orchestrated_details_contract_evidence(
    path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    failures = []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            failures.append(f"empty details line: {line_number}")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            failures.append(f"invalid details JSON: {line_number}")
            continue
        if not isinstance(row, dict):
            failures.append(f"details row is not an object: {line_number}")
            continue
        rows.append(row)

    required_fields = {
        "example_id",
        "target_status",
        "predicted_status",
        "target_route",
        "predicted_route",
        "executed",
        "dispatch_error",
    }
    identifiers = []
    target_ready_count = 0
    successful_ready_execution_count = 0
    target_status_counts: Counter[str] = Counter()
    predicted_status_counts: Counter[str] = Counter()
    invalid_predicted_status_count = 0
    task_counts: Counter[str] = Counter()
    direction_physical_count = 0
    forward_physical_count = 0
    for index, row in enumerate(rows):
        missing = required_fields - set(row)
        if missing:
            failures.append(f"details row {index}: missing fields {sorted(missing)}")
        identifier = row.get("example_id")
        if not isinstance(identifier, str) or not identifier:
            failures.append(f"details row {index}: invalid example_id")
        else:
            identifiers.append(identifier)
        if not isinstance(row.get("executed"), bool):
            failures.append(f"details row {index}: executed is not Boolean")
        target_status = row.get("target_status")
        target_route = row.get("target_route")
        if target_status not in {"ready", "needs_clarification", "unsupported"}:
            failures.append(f"details row {index}: target status is invalid")
        else:
            target_status_counts[str(target_status)] += 1
        if target_status != "ready" and target_route is not None:
            failures.append(f"details row {index}: non-ready target has a route")
        predicted_status = row.get("predicted_status")
        valid_predicted_statuses = {
            None,
            "ready",
            "needs_clarification",
            "unsupported",
        }
        if predicted_status in valid_predicted_statuses:
            predicted_status_counts[
                "<null>" if predicted_status is None else str(predicted_status)
            ] += 1
        else:
            # A schema-invalid status is a real Qwen failure, not corruption of
            # the diagnostic record. Preserve and count it, while requiring
            # the runtime to have rejected it without execution.
            invalid_predicted_status_count += 1
            predicted_status_counts["<invalid>"] += 1
            if row.get("executed") is True:
                failures.append(
                    f"details row {index}: invalid predicted status was executed"
                )
            dispatch_error = row.get("dispatch_error")
            if not isinstance(dispatch_error, str) or not dispatch_error:
                failures.append(
                    f"details row {index}: invalid predicted status lacks "
                    "a dispatch error"
                )
        dispatch_error = row.get("dispatch_error")
        if dispatch_error is not None and (
            not isinstance(dispatch_error, str) or not dispatch_error
        ):
            failures.append(f"details row {index}: dispatch error is invalid")
        if row.get("executed") is True:
            if predicted_status != "ready":
                failures.append(
                    f"details row {index}: executed decision is not predicted ready"
                )
            if row.get("predicted_route") not in ROUTE_TASK_TYPES:
                failures.append(
                    f"details row {index}: executed route is not registered"
                )
            if row.get("dispatch_error") is not None:
                failures.append(
                    f"details row {index}: executed decision has a dispatch error"
                )
        if row.get("dispatch_error") is not None and row.get("executed") is True:
            failures.append(f"details row {index}: dispatch error was executed")
        if target_status == "ready":
            target_ready_count += 1
            if row.get("executed") is True:
                successful_ready_execution_count += 1
            task_type = ROUTE_TASK_TYPES.get(target_route)
            if task_type is None:
                failures.append(f"details row {index}: unknown ready target route")
            else:
                task_counts[task_type] += 1
        if target_route in {
            "predict_direction_from_state_v1",
            "predict_direction_from_image_v1",
        }:
            direction_physical_count += 1
            if row.get("physical_metric") != "all_five_directions_exact":
                failures.append(f"details row {index}: direction metric differs")
            if not isinstance(row.get("physical_success"), bool) or not isinstance(
                row.get("correctly_routed_specialist_physical_success"),
                bool,
            ):
                failures.append(f"details row {index}: direction outcomes are missing")
        if target_route in {
            "predict_forward_from_state_v1",
            "predict_forward_from_image_v1",
        }:
            forward_physical_count += 1
            if (
                row.get("physical_metric")
                != "all_five_numerical_changes_within_tolerance"
            ):
                failures.append(f"details row {index}: forward metric differs")
            if not isinstance(row.get("physical_success"), bool) or not isinstance(
                row.get("correctly_routed_specialist_physical_success"),
                bool,
            ):
                failures.append(f"details row {index}: forward outcomes are missing")

    duplicate_ids = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicate_ids:
        failures.append(f"duplicate details example IDs: {duplicate_ids[:5]}")
    metrics = report.get("metrics", {})
    expected_task_counts = report.get("task_ready_counts")
    observed_task_counts = dict(sorted(task_counts.items()))
    comparisons = {
        "record_count": (len(rows), report.get("record_count")),
        "target_ready_count": (
            target_ready_count,
            metrics.get("target_ready_count"),
        ),
        "task_ready_counts": (observed_task_counts, expected_task_counts),
        "direction_physical_count": (
            direction_physical_count,
            metrics.get("direction_physical_count"),
        ),
        "forward_physical_count": (
            forward_physical_count,
            metrics.get("forward_physical_count"),
        ),
    }
    for label, (observed, expected) in comparisons.items():
        if observed != expected:
            failures.append(f"{label}: details disagree with summary")
    expected_execution_rate = successful_ready_execution_count / max(
        target_ready_count,
        1,
    )
    reported_execution_rate = metrics.get("successful_valid_execution_rate")
    if (
        not isinstance(reported_execution_rate, (int, float))
        or isinstance(reported_execution_rate, bool)
        or not math.isfinite(float(reported_execution_rate))
        or not math.isclose(
            float(reported_execution_rate),
            expected_execution_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        failures.append("successful execution rate: details disagree with summary")
    if len(set(identifiers)) != len(rows):
        failures.append("unique example ID count differs from details row count")
    return {
        "failures": failures,
        "record_count": len(rows),
        "unique_example_id_count": len(set(identifiers)),
        "target_ready_count": target_ready_count,
        "successful_ready_execution_count": successful_ready_execution_count,
        "target_status_counts": dict(sorted(target_status_counts.items())),
        "predicted_status_counts": dict(sorted(predicted_status_counts.items())),
        "invalid_predicted_status_count": invalid_predicted_status_count,
        "task_ready_counts": observed_task_counts,
        "direction_physical_count": direction_physical_count,
        "forward_physical_count": forward_physical_count,
    }


def validation_decision_contract_failures(
    decision: Mapping[str, Any],
) -> list[str]:
    failures = []
    if (
        decision.get("decision_rule_version")
        != "control_rebuild_v4_preregistered_validation_gates"
    ):
        failures.append("decision rule version differs")
    if decision.get("thresholds") != VALIDATION_THRESHOLDS:
        failures.append("registered threshold table differs")
    gates = decision.get("gates")
    if not isinstance(gates, list):
        failures.append("gates are not a list")
        gates = []
    if len(gates) != EXPECTED_VALIDATION_GATE_COUNT:
        failures.append("gate record count differs")
    labels = []
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            failures.append(f"gate {index}: record is not an object")
            continue
        if set(gate) != {
            "label",
            "comparison",
            "observed",
            "threshold",
            "passed",
        }:
            failures.append(f"gate {index}: fields differ")
        if not isinstance(gate.get("label"), str) or not gate["label"]:
            failures.append(f"gate {index}: label is missing")
        else:
            labels.append(gate["label"])
        if gate.get("comparison") not in {">=", "<=", "=="}:
            failures.append(f"gate {index}: comparison is invalid")
        if gate.get("passed") is not True:
            failures.append(f"gate {index}: did not explicitly pass")
    if len(set(labels)) != len(labels):
        failures.append("gate labels are not unique")
    if decision.get("gate_count") != EXPECTED_VALIDATION_GATE_COUNT:
        failures.append("declared gate count differs")
    if decision.get("passed_gate_count") != EXPECTED_VALIDATION_GATE_COUNT:
        failures.append("declared passed-gate count differs")
    if decision.get("failed_gates") != []:
        failures.append("failed-gate list is not empty")
    return failures


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def require(self, label: str, condition: bool, evidence: Any) -> None:
        self.checks.append(
            {
                "label": label,
                "passed": bool(condition),
                "evidence": evidence,
            }
        )

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [row for row in self.checks if not row["passed"]]


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    measurement_run = args.measurement_v4_run.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "completion_audit.json"
    )
    audit = Audit()

    qwen_failures = verify_qwen(QWEN_FREEZE_MANIFEST)
    audit.require(
        "frozen Qwen baseline integrity",
        not qwen_failures,
        {"failures": qwen_failures},
    )

    required_paths = [
        data_dir / "manifest.json",
        data_dir / "checksums.sha256",
        measurement_run / "measurement_calibrator_v4.pt",
        measurement_run / "measurement_calibrator_v4_summary.json",
        run_dir / "measurement_error_bank_v4.npz",
        run_dir / "measurement_error_bank_v4_summary.json",
        run_dir / "forward_physics_residual_v4.pt",
        run_dir / "forward_physics_residual_v4_summary.json",
        run_dir / "inverse_control_v4.pt",
        run_dir / "inverse_control_v4_summary.json",
        run_dir / "v3_v4_selection_validation_comparison.json",
        run_dir / "visual_sensor_scorer_v4_integrated.pt",
        run_dir / "visual_sensor_scorer_v4_integrated_summary.json",
        run_dir / "candidate_overlay_manifest.json",
        run_dir / "controlled_validation.json",
        run_dir / "closed_loop_val.json",
        run_dir / "orchestrated_runtime_validation.json",
        run_dir / "orchestrated_system_validation.json",
        run_dir / "orchestrated_system_validation.details.jsonl",
        run_dir / "validation_decision.json",
        run_dir / "validation_performance_summary.json",
        run_dir / "VALIDATION_REPORT.md",
        run_dir / "controlled_evaluation.json",
        run_dir / "closed_loop_test_iid.json",
        run_dir / "closed_loop_test_ood_physics.json",
        run_dir / "final_performance_summary.json",
        run_dir / "FINAL_REPORT.md",
        run_dir / "final_examples.json",
        run_dir / "FINAL_EXAMPLES.md",
        run_dir / "post_generation_pipeline.complete",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    audit.require("all declared artifacts exist", not missing, {"missing": missing})
    if missing:
        result = {
            "audit_version": "control_rebuild_v4_completion_audit",
            "complete": False,
            "checks": audit.checks,
            "failure_count": len(audit.failures),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(1)

    manifest = read_json(data_dir / "manifest.json")
    verification = manifest.get("verification", {})
    splits = verification.get("splits", {})
    audit.require(
        "numerical dataset verified with exact split sizes",
        bool(verification.get("complete"))
        and splits.get("train", {}).get("groups") == 3000
        and splits.get("train", {}).get("transitions") == 243000
        and splits.get("val", {}).get("groups") == 450
        and splits.get("val", {}).get("transitions") == 36450,
        {"splits": splits},
    )
    audit.require(
        "seeded category allocation matches the declared failure-focused mix",
        splits.get("train", {}).get("category_counts")
        == {
            "high_nonlinearity": 1165,
            "iid_expanded": 646,
            "ood_boundary": 1189,
        }
        and splits.get("val", {}).get("category_counts")
        == {
            "high_nonlinearity": 189,
            "iid_expanded": 78,
            "ood_boundary": 183,
        },
        {
            split: splits.get(split, {}).get("category_counts")
            for split in ("train", "val")
        },
    )
    audit.require(
        "dataset split, action, finite-value, and test-isolation invariants",
        verification.get("cross_split_group_overlap") == 0
        and verification.get("candidate_count_per_group") == 81
        and verification.get("finite_numerical_values") is True
        and verification.get("action_grid_order_exact") is True
        and verification.get("held_out_v2_test_files_opened") is False,
        verification,
    )
    checksum_evidence = checksum_contract_evidence(data_dir)
    checksum_manifest = manifest.get("checksum_contract", {})
    audit.require(
        "all numerical dataset checksums match",
        not checksum_evidence["failures"]
        and checksum_evidence["line_count"] == 3453
        and checksum_evidence["unique_target_count"] == 3453
        and checksum_evidence["expected_file_count"] == 3453
        and checksum_manifest.get("file_count") == 3453
        and checksum_manifest.get("checksum_manifest_sha256")
        == checksum_evidence["checksum_manifest_sha256"],
        {
            **checksum_evidence,
            "manifest_file_count": checksum_manifest.get("file_count"),
            "manifest_checksum_manifest_sha256": checksum_manifest.get(
                "checksum_manifest_sha256"
            ),
        },
    )

    measurement_summary = read_json(
        measurement_run / "measurement_calibrator_v4_summary.json"
    )
    error_summary = read_json(run_dir / "measurement_error_bank_v4_summary.json")
    audit.require(
        "measurement calibrator used no held-out test data",
        measurement_summary.get("held_out_test_used") is False,
        {
            "held_out_test_used": measurement_summary.get("held_out_test_used"),
            "artifact": measurement_summary.get("artifact"),
        },
    )
    audit.require(
        "measurement error bank is training-only with 70,000 samples",
        error_summary.get("source_split") == "train"
        and error_summary.get("validation_used") is False
        and error_summary.get("held_out_test_used") is False
        and error_summary.get("sample_count") == 70000,
        {
            key: error_summary.get(key)
            for key in (
                "source_split",
                "validation_used",
                "held_out_test_used",
                "sample_count",
            )
        },
    )

    import torch

    forward_artifact = torch.load(
        run_dir / "forward_physics_residual_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    inverse_artifact = torch.load(
        run_dir / "inverse_control_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    visual_artifact = torch.load(
        run_dir / "visual_sensor_scorer_v4_integrated.pt",
        map_location="cpu",
        weights_only=False,
    )
    forward_summary = read_json(run_dir / "forward_physics_residual_v4_summary.json")
    inverse_summary = read_json(run_dir / "inverse_control_v4_summary.json")
    selection_comparison = read_json(
        run_dir / "v3_v4_selection_validation_comparison.json"
    )
    visual_summary = read_json(
        run_dir / "visual_sensor_scorer_v4_integrated_summary.json"
    )
    candidate_manifest = read_json(run_dir / "candidate_overlay_manifest.json")
    audit.require(
        "full forward checkpoint has exact zero anchor and full training counts",
        forward_artifact.get("zero_action_exact") is True
        and forward_artifact.get("seed") == 20260726
        and len(forward_artifact.get("action_grid", [])) == 81
        and all(
            float(value) == 0.0
            for value in forward_artifact["action_grid"][40].values()
        )
        and forward_summary.get("train_groups") == 5500
        and forward_summary.get("train_transitions") == 445500
        and forward_summary.get("held_out_test_used") is False,
        {
            "zero_action_exact": forward_artifact.get("zero_action_exact"),
            "train_groups": forward_summary.get("train_groups"),
            "train_transitions": forward_summary.get("train_transitions"),
            "held_out_test_used": forward_summary.get("held_out_test_used"),
        },
    )
    inverse_training = inverse_summary.get("training", {})
    audit.require(
        "full inverse checkpoint uses v2 plus derived v4 and paired measurement errors",
        inverse_training.get("v2_pairs") == 60000
        and inverse_artifact.get("seed") == 20260726
        and inverse_training.get("v4_derived_pairs") == 18000
        and inverse_training.get("total_pairs") == 78000
        and inverse_summary.get("measurement_error_condition_pairing")
        == "current_and_desired_same_condition"
        and inverse_summary.get("held_out_test_used") is False
        and len(inverse_artifact.get("action_grid", [])) == 81,
        {
            "training": inverse_training,
            "pairing": inverse_summary.get("measurement_error_condition_pairing"),
            "held_out_test_used": inverse_summary.get("held_out_test_used"),
        },
    )
    audit.require(
        "v3 and v4 were compared on identical difficult validation groups",
        selection_comparison.get("complete") is True
        and selection_comparison.get("category_counts")
        == {
            "high_nonlinearity": 189,
            "iid_expanded": 78,
            "ood_boundary": 183,
        }
        and selection_comparison.get("models", {})
        .get("v3", {})
        .get("all", {})
        .get("group_count")
        == 450
        and selection_comparison.get("models", {})
        .get("v4", {})
        .get("all", {})
        .get("transition_count")
        == 36450
        and selection_comparison.get("held_out_used_for_training_or_selection") == 0,
        {
            "complete": selection_comparison.get("complete"),
            "category_counts": selection_comparison.get("category_counts"),
            "v3_all": selection_comparison.get("models", {})
            .get("v3", {})
            .get("all", {}),
            "v4_all": selection_comparison.get("models", {})
            .get("v4", {})
            .get("all", {}),
        },
    )
    comparison_examples = selection_comparison.get("examples", {})
    overall_comparison_examples = comparison_examples.get("all", {})
    expected_example_outcomes = {
        "forward": {
            "v4_improvement": (False, True),
            "v4_regression": (True, False),
        },
        "inverse": {
            "v4_improvement": (False, True),
            "v4_regression": (True, False),
        },
    }
    example_metric_contracts = {
        "forward": (
            "strict_all_five_forward_prediction",
            "v3_strict_all_five_success",
            "v4_strict_all_five_success",
        ),
        "inverse": (
            "physical_target_success_feasible",
            "v3_physical_target_success",
            "v4_physical_target_success",
        ),
    }
    example_contract_failures = []
    for task, outcomes in expected_example_outcomes.items():
        metric, v3_key, v4_key = example_metric_contracts[task]
        for outcome, expected_flags in outcomes.items():
            record = overall_comparison_examples.get(task, {}).get(outcome)
            if (
                not isinstance(record, dict)
                or record.get("metric") != metric
                or not record.get("metric_definition")
                or (
                    record.get(v3_key),
                    record.get(v4_key),
                )
                != expected_flags
            ):
                example_contract_failures.append(f"{task}:{outcome}")
    audit.require(
        "same-distribution comparison records real v4 fixes and regressions",
        set(overall_comparison_examples) == {"forward", "inverse"}
        and all(
            set(value) == {"v4_improvement", "v4_regression"}
            for value in overall_comparison_examples.values()
        )
        and set(comparison_examples.get("by_category", {}))
        == {"high_nonlinearity", "iid_expanded", "ood_boundary"}
        and not example_contract_failures,
        {
            "all": {
                task: {
                    outcome: row.get(outcome) is not None
                    for outcome in ("v4_improvement", "v4_regression")
                }
                for task, row in overall_comparison_examples.items()
            },
            "categories": sorted(comparison_examples.get("by_category", {})),
            "contract_failures": example_contract_failures,
        },
    )
    audit.require(
        "integrated visual scorer uses full forward and inverse initialization",
        Path(str(visual_artifact.get("forward_artifact"))).resolve()
        == (run_dir / "forward_physics_residual_v4.pt").resolve()
        and visual_artifact.get("seed") == 20260726
        and Path(str(visual_artifact.get("inverse_initialization"))).resolve()
        == (run_dir / "inverse_control_v4.pt").resolve()
        and visual_summary.get("held_out_test_used") is False
        and visual_summary.get("train_groups") == 2500
        and visual_summary.get("train_pairs") == 52500,
        {
            "forward_artifact": visual_artifact.get("forward_artifact"),
            "inverse_initialization": visual_artifact.get("inverse_initialization"),
            "train_groups": visual_summary.get("train_groups"),
            "train_pairs": visual_summary.get("train_pairs"),
            "held_out_test_used": visual_summary.get("held_out_test_used"),
        },
    )
    manifest_artifacts = candidate_manifest.get("artifacts", {})
    manifest_contract_failures = candidate_manifest_contract_failures(
        candidate_manifest,
        run_dir,
        measurement_run,
    )
    audit.require(
        "candidate execution manifest pins all seven routes and artifact hashes",
        candidate_manifest.get("complete") is True
        and candidate_manifest.get("seed") == 20260726
        and candidate_manifest.get("action_grid_size") == 81
        and candidate_manifest.get("simulator_at_inference") is False
        and candidate_manifest.get("held_out_test_used_for_training_or_selection") == 0
        and len(candidate_manifest.get("routes", {})) == 7
        and not manifest_contract_failures,
        {
            "route_count": len(candidate_manifest.get("routes", {})),
            "artifact_count": len(manifest_artifacts),
            "contract_failures": manifest_contract_failures,
        },
    )

    validation = read_json(run_dir / "controlled_validation.json")
    loop_val = read_json(run_dir / "closed_loop_val.json")
    overlay = read_json(run_dir / "orchestrated_runtime_validation.json")
    system = read_json(run_dir / "orchestrated_system_validation.json")
    validation_decision = read_json(run_dir / "validation_decision.json")
    audit.require(
        "controlled validation is complete and test-isolated",
        validation.get("complete") is True
        and validation.get("seed") == 20260726
        and set(validation.get("split_results", {})) == {"val"}
        and validation.get("held_out_examples_used_for_training_or_selection") == 0,
        {
            "complete": validation.get("complete"),
            "splits": sorted(validation.get("split_results", {})),
            "held_out_used": validation.get(
                "held_out_examples_used_for_training_or_selection"
            ),
        },
    )
    audit.require(
        "validation closed loop contains 50 three-step requests",
        loop_val.get("complete") is True
        and loop_val.get("metrics", {}).get("request_count") == 50
        and loop_val.get("metrics", {}).get("max_steps") == 3,
        loop_val.get("metrics", {}),
    )
    direct = overlay.get("direct_measurement_validation", {})
    audit.require(
        "all seven Qwen-to-v4 routes and 150 direct images were validated",
        overlay.get("complete") is True
        and overlay.get("all_routes_passed") is True
        and overlay.get("routes_passed") == 7
        and overlay.get("simulator_inference_calls") == 0
        and direct.get("count") == 150
        and Path(str(overlay.get("overlay_manifest"))).resolve()
        == (run_dir / "candidate_overlay_manifest.json").resolve(),
        {
            "routes_passed": overlay.get("routes_passed"),
            "all_routes_passed": overlay.get("all_routes_passed"),
            "simulator_inference_calls": overlay.get("simulator_inference_calls"),
            "direct_measurement_count": direct.get("count"),
        },
    )
    system_metrics = system.get("metrics", {})
    audit.require(
        "saved Qwen checkpoint was executed through v4 on all 1,600 records",
        system.get("complete") is True
        and system.get("record_count") == 1600
        and system_metrics.get("target_ready_count") == 1050
        and system_metrics.get("simulator_calls_during_inference_count") == 0
        and Path(str(system.get("overlay_manifest"))).resolve()
        == (run_dir / "candidate_overlay_manifest.json").resolve()
        and system.get("task_ready_counts")
        == {
            "beam_profile_measurement": 150,
            "direction_prediction": 300,
            "forward_prediction": 300,
            "inverse_control": 300,
        },
        {
            "record_count": system.get("record_count"),
            "target_ready_count": system_metrics.get("target_ready_count"),
            "task_ready_counts": system.get("task_ready_counts"),
            "simulator_calls_during_inference": system_metrics.get(
                "simulator_calls_during_inference_count"
            ),
        },
    )
    audit.require(
        "combined system direction and forward metrics use physical ground truth",
        system_metrics.get("direction_physical_count") == 300
        and system_metrics.get("forward_physical_count") == 300
        and system_metrics.get("private_forward_ground_truth_simulator_calls") == 150
        and system_metrics.get("private_simulator_scoring_calls_total") == 750
        and isinstance(
            system_metrics.get("end_to_end_direction_physical_macro_f1"),
            (int, float),
        )
        and isinstance(
            system_metrics.get("end_to_end_forward_physical_strict_all_five_success"),
            (int, float),
        ),
        {
            key: system_metrics.get(key)
            for key in (
                "direction_physical_count",
                "forward_physical_count",
                "private_forward_ground_truth_simulator_calls",
                "private_simulator_scoring_calls_total",
                "end_to_end_direction_physical_macro_f1",
                "end_to_end_forward_physical_strict_all_five_success",
            )
        },
    )
    system_details = orchestrated_details_contract_evidence(
        run_dir / "orchestrated_system_validation.details.jsonl",
        system,
    )
    audit.require(
        "combined-system details contain one valid record per frozen Qwen example",
        not system_details["failures"]
        and system_details["record_count"] == 1600
        and system_details["unique_example_id_count"] == 1600
        and system_details["target_ready_count"] == 1050
        and system_details["direction_physical_count"] == 300
        and system_details["forward_physical_count"] == 300,
        system_details,
    )
    validation_decision_failures = validation_decision_contract_failures(
        validation_decision
    )
    audit.require(
        "pre-registered validation gates authorized the held-out evaluation",
        validation_decision.get("complete") is True
        and validation_decision.get("proceed_to_heldout") is True
        and validation_decision.get("held_out_evaluation_opened") is False
        and not validation_decision_failures,
        {
            "gate_count": validation_decision.get("gate_count"),
            "passed_gate_count": validation_decision.get("passed_gate_count"),
            "failed_gates": validation_decision.get("failed_gates"),
            "proceed_to_heldout": validation_decision.get("proceed_to_heldout"),
            "contract_failures": validation_decision_failures,
        },
    )

    final = read_json(run_dir / "controlled_evaluation.json")
    expected_final_splits = {
        "test_iid",
        "test_ood_physics",
        "test_visual_stress",
    }
    audit.require(
        "one final controlled evaluation is complete on all held-out splits",
        final.get("complete") is True
        and final.get("seed") == 20260726
        and set(final.get("split_results", {})) == expected_final_splits
        and final.get("held_out_examples_used_for_training_or_selection") == 0
        and int(final.get("held_out_state_examples_evaluated", 0)) > 0,
        {
            "complete": final.get("complete"),
            "splits": sorted(final.get("split_results", {})),
            "held_out_used_for_training_or_selection": final.get(
                "held_out_examples_used_for_training_or_selection"
            ),
            "held_out_state_examples_evaluated": final.get(
                "held_out_state_examples_evaluated"
            ),
        },
    )
    loop_results = {}
    for split in ("test_iid", "test_ood_physics"):
        value = read_json(run_dir / f"closed_loop_{split}.json")
        loop_results[split] = value.get("metrics", {})
    audit.require(
        "both held-out closed-loop evaluations contain 50 three-step requests",
        all(
            read_json(run_dir / f"closed_loop_{split}.json").get("complete") is True
            and loop_results[split].get("request_count") == 50
            and loop_results[split].get("max_steps") == 3
            for split in loop_results
        ),
        loop_results,
    )

    final_summary = read_json(run_dir / "final_performance_summary.json")
    audit.require(
        "final report preserves validation versus held-out evidence scope",
        final_summary.get("phase") == "final"
        and final_summary.get("evidence_scope", {}).get(
            "held_out_used_for_training_or_selection"
        )
        == 0
        and final_summary.get("evidence_scope", {}).get(
            "same_denominator_for_qwen_and_specialists"
        )
        is False,
        final_summary.get("evidence_scope", {}),
    )
    failure_focused = final_summary.get("failure_focused_selection_validation", {})
    audit.require(
        "report exposes the 450-group failure-focused selection validation",
        failure_focused.get("group_count") == 450
        and failure_focused.get("category_counts")
        == {
            "high_nonlinearity": 189,
            "iid_expanded": 78,
            "ood_boundary": 183,
        }
        and failure_focused.get("forward", {}).get("transition_count") == 36450
        and failure_focused.get("forward_only_target_retrieval", {}).get(
            "request_count"
        )
        == 1350
        and failure_focused.get("inverse_clean", {}).get("pair_count") == 2700
        and failure_focused.get("inverse_clean", {}).get("reachable_pair_count") == 1800
        and failure_focused.get("inverse_paired_measurement_errors", {}).get(
            "pair_count"
        )
        == 2700
        and failure_focused.get("held_out_used_for_training_or_selection") == 0,
        failure_focused,
    )
    examples = read_json(run_dir / "final_examples.json")
    example_tasks = examples.get("tasks", {})
    example_failures = real_example_contract_failures(
        examples,
        expected_final_splits,
    )
    audit.require(
        "final real-example report covers all four rebuilt specialist tasks",
        examples.get("phase") == "final"
        and examples.get("report_version") == "control_rebuild_v4_real_examples"
        and not example_failures,
        {
            task: {
                outcome: isinstance(value, dict) and value.get(outcome) is not None
                for outcome in ("success", "failure")
            }
            for task, value in example_tasks.items()
        }
        | {"contract_failures": example_failures},
    )
    rate_failures = [
        *rates_in_range(validation, "controlled_validation"),
        *rates_in_range(system, "orchestrated_system_validation"),
        *rates_in_range(final, "controlled_evaluation"),
        *rates_in_range(final_summary, "final_performance_summary"),
        *rates_in_range(loop_val, "closed_loop_val"),
        *[
            failure
            for split in loop_results
            for failure in rates_in_range(loop_results[split], f"closed_loop_{split}")
        ],
    ]
    audit.require(
        "all reported primary rates are finite and inside zero to one",
        not rate_failures,
        {"failures": rate_failures},
    )

    post_stages = run_dir / "post_generation_stages"
    final_stages = run_dir / "final_evaluation_stages"
    required_stage_markers = [
        post_stages / "verify_frozen_qwen.complete",
        post_stages / "finalize_numerical_dataset.complete",
        post_stages / "train_forward_v4.complete",
        post_stages / "train_inverse_v4.complete",
        post_stages / "compare_v3_v4_selection_validation.complete",
        post_stages / "train_visual_integrated_v4.complete",
        post_stages / "candidate_overlay_manifest.complete",
        post_stages / "controlled_validation.complete",
        post_stages / "closed_loop_val.complete",
        post_stages / "orchestrated_runtime_validation.complete",
        post_stages / "orchestrated_system_validation.complete",
        post_stages / "assess_validation.complete",
        post_stages / "summarize_validation.complete",
        final_stages / "controlled_evaluation.complete",
        final_stages / "closed_loop_test_iid.complete",
        final_stages / "closed_loop_test_ood_physics.complete",
        final_stages / "summarize_final.complete",
    ]
    missing_markers = [
        str(path) for path in required_stage_markers if not path.is_file()
    ]
    audit.require(
        "every restart-safe pipeline stage has a completion marker",
        not missing_markers,
        {"missing": missing_markers},
    )

    result = {
        "audit_version": "control_rebuild_v4_completion_audit",
        "complete": not audit.failures,
        "checks": audit.checks,
        "check_count": len(audit.checks),
        "failure_count": len(audit.failures),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
