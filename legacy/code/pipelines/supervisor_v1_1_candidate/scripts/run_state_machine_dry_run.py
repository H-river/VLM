#!/usr/bin/env python3
"""Synthetic and candidate-train/dev-only supervisor safety dry run."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_vl_supervisor_v1.closed_loop_adapter import (
    FROZEN_MAX_HORIZON,
    FROZEN_TO_SUPERVISOR_POLICY,
    SUPERVISOR_TO_FROZEN_POLICY,
    adapt_supervisor_decision_or_safe_stop,
    from_frozen_policy,
    to_frozen_policy,
)

BASE = ROOT / "supervisor_v1_1_candidate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def payload(diagnosis: str, policy: str, action: str) -> str:
    return json.dumps(
        {"diagnosis": diagnosis, "measurement_policy": policy, "supervisor_action": action},
        separators=(",", ":"),
    )


def adapt_case(name: str, text: str, *, steps: int = 0, budget: int = 8, allow: bool = True) -> dict[str, Any]:
    result = adapt_supervisor_decision_or_safe_stop(
        text, executed_steps=steps, remaining_step_budget=budget,
        frozen_continuation_allows_next=allow,
    )
    return {"name": name, "decision": asdict(result.decision), "audit": asdict(result.audit)}


def audit_visible_contract() -> dict[str, Any]:
    counts = {"train": 0, "dev": 0}
    violations: list[dict[str, str]] = []
    frame_counts: dict[str, int] = {}
    for split in ("train", "dev"):
        for row in read_jsonl(BASE / f"manifests/manifest_{split}.jsonl"):
            counts[split] += 1
            sample_id = row["sample_id"]
            current = row["model_input"]["current_metrics"]
            goal = row["model_input"]["goal_metrics"]
            constraints = row["model_input"]["actuator_constraints"]
            current_frame = current.get("coordinate_frame")
            goal_frame = goal.get("coordinate_frame")
            frame_counts[f"current:{current_frame}"] = frame_counts.get(f"current:{current_frame}", 0) + 1
            frame_counts[f"goal:{goal_frame}"] = frame_counts.get(f"goal:{goal_frame}", 0) + 1
            checks = {
                "current_frame": current_frame == "diagnostic_image_128px",
                "current_centroid_range": all(0.0 <= float(current[key]) <= 127.0 for key in ("centroid_x", "centroid_y")),
                "current_width_positive": all(float(current[key]) > 0 for key in ("width_x", "width_y")),
                "goal_frame": goal_frame == "lab_sensor_1024px_and_raw_peak",
                "goal_centroid_range": all(0.0 <= float(goal[key]) <= 1023.0 for key in ("centroid_x", "centroid_y")),
                "goal_width_positive": all(float(goal[key]) > 0 for key in ("width_x", "width_y")),
                "actuator_units_mm": constraints.get("units") == "mm",
                "repository_not_hardware": constraints.get("absolute_limit_source") == "repository_sampling_domain_not_hardware_limit",
                "continuous_owner_h1": constraints.get("continuous_actions_selected_by") == "frozen_h1_one_step_cem",
                "budget_integer_bounded": isinstance(row["model_input"]["remaining_step_budget"], int) and 0 <= row["model_input"]["remaining_step_budget"] <= FROZEN_MAX_HORIZON,
            }
            image_path = ROOT / row["assets"]["current_image_path"]
            with Image.open(image_path) as image:
                checks["image_128"] = image.size == (128, 128)
            checks["image_hash"] = sha256(image_path) == row["assets"]["current_image_sha256"]
            for check, passed in checks.items():
                if not passed:
                    violations.append({"sample_id": sample_id, "check": check})
    return {
        "records_checked": counts,
        "frame_counts": frame_counts,
        "violations": violations,
        "passed": not violations,
        "protected_data_used": False,
    }


def main() -> None:
    nominal = adapt_case("nominal_to_standard_to_h1", payload("nominal", "standard", "execute"))
    saturation = adapt_case(
        "saturation_to_lower_exposure_reacquire", payload("sensor_saturation", "lower_exposure_reacquire", "reacquire")
    )
    reflection = adapt_case(
        "reflection_to_primary_spot_to_h1", payload("secondary_reflection", "primary_spot", "switch_measurement")
    )
    invalid = adapt_case("invalid_json_safe_stop", "{not json")
    injection = adapt_case(
        "numeric_continuous_action_injection_safe_stop",
        '{"diagnosis":"nominal","measurement_policy":"standard","supervisor_action":"execute","camera_x_delta_mm":0.02}',
    )
    numeric_field = adapt_case(
        "numeric_enum_injection_safe_stop",
        '{"diagnosis":1,"measurement_policy":"standard","supervisor_action":"execute"}',
    )
    continuation_denied = adapt_case(
        "frozen_continuation_denied", payload("nominal", "standard", "execute"), allow=False
    )
    horizon_clamped = adapt_case(
        "frozen_horizon_clamps_caller_budget", payload("nominal", "standard", "execute"), steps=8, budget=100
    )

    # This is a contract-level orchestration trace, not an observed recovery.
    repeated_trace = []
    for executed_steps, budget in enumerate((3, 2, 1, 0)):
        row = adapt_case(
            f"repeated_saturation_{executed_steps}",
            payload("sensor_saturation", "lower_exposure_reacquire", "reacquire"),
            steps=executed_steps, budget=budget,
        )
        repeated_trace.append({
            "decision": row,
            "orchestrator_next_state": "STOPPED" if row["decision"]["should_stop"] else "REACQUIRE_THEN_REDIAGNOSE",
            "uses_real_sequential_observation": False,
        })

    mapping_rows = []
    mapping_passed = True
    for supervisor_policy, frozen_policy in SUPERVISOR_TO_FROZEN_POLICY.items():
        row = {
            "supervisor_policy": supervisor_policy,
            "frozen_policy": frozen_policy,
            "forward": to_frozen_policy(supervisor_policy),
            "reverse": from_frozen_policy(frozen_policy),
        }
        row["round_trip_passed"] = row["forward"] == frozen_policy and row["reverse"] == supervisor_policy
        mapping_passed = mapping_passed and row["round_trip_passed"]
        mapping_rows.append(row)
    mapping_passed = mapping_passed and dict(FROZEN_TO_SUPERVISOR_POLICY) == {
        value: key for key, value in SUPERVISOR_TO_FROZEN_POLICY.items()
    }

    safety_assertions = {
        "nominal_standard_h1_route": nominal["decision"]["controller_operation"] == "frozen_cem_step" and nominal["decision"]["execute_frozen_cem"],
        "saturation_reacquire_route": saturation["decision"]["controller_operation"] == "reacquire_then_frozen_cem",
        "saturation_orchestrator_requires_rediagnosis_after_reacquire": repeated_trace[0]["orchestrator_next_state"] == "REACQUIRE_THEN_REDIAGNOSE",
        "reflection_primary_spot_h1_route": reflection["decision"]["controller_operation"] == "switch_measurement_then_frozen_cem" and reflection["decision"]["execute_frozen_cem"],
        "invalid_json_non_dispatched_stop": not invalid["audit"]["parse_valid"] and invalid["decision"]["should_stop"] and not invalid["decision"]["execute_frozen_cem"],
        "continuous_injection_rejected": not injection["audit"]["parse_valid"] and injection["audit"]["error_code"] == "invalid_decision_keys",
        "numeric_enum_rejected": not numeric_field["audit"]["parse_valid"] and numeric_field["audit"]["error_code"] == "invalid_field_type",
        "repeated_anomaly_budget_terminates": repeated_trace[-1]["decision"]["decision"]["should_stop"] and repeated_trace[-1]["orchestrator_next_state"] == "STOPPED",
        "continuation_gate_authoritative": continuation_denied["decision"]["should_stop"] and not continuation_denied["decision"]["execute_frozen_cem"],
        "horizon_authoritative": horizon_clamped["decision"]["should_stop"] and horizon_clamped["decision"]["budget_was_clamped"],
        "policy_mapping_reversible": mapping_passed,
        "supervisor_has_no_actuator_fields": injection["audit"]["error_code"] == "invalid_decision_keys",
        "dry_run_controller_backend_h1_only": all("h3" not in case["decision"]["controller_operation"].lower() for case in (nominal, saturation, reflection)),
    }
    visible = audit_visible_contract()
    safety_assertions["candidate_visible_coordinate_and_unit_contract"] = visible["passed"]

    report = {
        "version": "supervisor_v1_1_candidate_state_machine_dry_run_v1",
        "status": "NOT SEALED — FROZEN EVALUATION DISABLED",
        "scope": "synthetic fixtures and candidate train/dev only",
        "cases": [nominal, saturation, reflection, invalid, injection, numeric_field, continuation_denied, horizon_clamped],
        "repeated_anomaly_trace": repeated_trace,
        "policy_mapping": mapping_rows,
        "safety_assertions": safety_assertions,
        "all_safety_assertions_passed": all(safety_assertions.values()),
        "visible_contract_audit": visible,
        "dispatch_contract": {
            "continuous_action_owner": "frozen H1 one-step CEM",
            "h3_registered_or_called_by_this_dry_run": False,
            "direct_actuator_authority": False,
            "note": "The dry run validates enum routing and guards; it does not execute a controller or actuator callback.",
        },
        "known_repository_semantics": {
            "camera_sampling_discontinuity": "The legacy v12.0 sensor extraction path retained clipped left-searchsorted sampling; corrected v12.1 uses continuous finite-pixel irradiance sampling. This dry run does not establish physical camera behavior.",
            "lab_sensor_coordinate_correction": "Current supervisor metrics are explicitly diagnostic-image 128px, while goals are labelled lab_sensor_1024px_and_raw_peak. Corrected v12.1 documents centroid_sensor_px = centroid_lab_px - camera_pose_m/pixel_pitch_m; this dry run checks labels and ranges, not calibration accuracy.",
            "power_w_dead_input": "In the legacy path power_w was not causal because the source stayed peak-normalized. Corrected v12.1 makes power_w causal. The static supervisor prompt does not expose setup power_w directly, so no supervisor claim may rely on learning that setup input.",
            "action_limits": "Visible lens +/-0.05 mm and camera +/-0.02 mm values are per-step bounds. +/-3 mm is a repository sampling-domain limit, not a hardware limit.",
        },
        "temporal_evidence": {
            "real_paired_sequential_anomaly_observations_available": False,
            "conclusion": "temporal recovery performance has not been verified",
        },
        "frozen_or_protected_data_used": False,
    }
    destination = BASE / "reports" / "state_machine_dry_run.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "passed": report["all_safety_assertions_passed"], "visible_violations": len(visible["violations"])}, indent=2))


if __name__ == "__main__":
    main()
