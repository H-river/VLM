"""Development-only actuator-free wiring smoke for the closed-loop adapter."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .closed_loop_adapter import (
    SupervisorDecisionError,
    adapt_and_dispatch,
    adapt_supervisor_decision_or_safe_stop,
)


CASES = (
    {
        "diagnosis": "nominal",
        "measurement_policy": "standard",
        "supervisor_action": "execute",
    },
    {
        "diagnosis": "sensor_saturation",
        "measurement_policy": "lower_exposure_reacquire",
        "supervisor_action": "reacquire",
    },
    {
        "diagnosis": "secondary_reflection",
        "measurement_policy": "primary_spot",
        "supervisor_action": "switch_measurement",
    },
)


def run() -> dict[str, Any]:
    dispatched: list[dict[str, Any]] = []

    def frozen_owner_mock(decision: Any) -> dict[str, Any]:
        payload = asdict(decision)
        serialized = json.dumps(payload, sort_keys=True)
        forbidden = ("lens_x", "lens_y", "camera_x", "camera_y", "delta_mm", "position_mm")
        if any(value in serialized for value in forbidden):
            raise RuntimeError("adapter leaked a continuous actuator field")
        result = {
            "diagnosis": decision.supervisor.diagnosis,
            "frozen_measurement_policy": decision.frozen_measurement_policy,
            "controller_operation": decision.controller_operation,
            "execute_frozen_cem": decision.execute_frozen_cem,
            "guard_reason": decision.guard_reason,
        }
        dispatched.append(result)
        return result

    for case in CASES:
        adapt_and_dispatch(
            json.dumps(case, separators=(",", ":")),
            frozen_owner_mock,
            executed_steps=0,
            remaining_step_budget=8,
            frozen_continuation_allows_next=True,
        )

    invalid_rejected = False
    try:
        adapt_and_dispatch(
            {
                **CASES[0],
                "lens_x_delta_mm": 0.05,
            },
            frozen_owner_mock,
            executed_steps=0,
            remaining_step_budget=8,
            frozen_continuation_allows_next=True,
        )
    except SupervisorDecisionError:
        invalid_rejected = True
    if not invalid_rejected:
        raise RuntimeError("numeric actuator injection was not rejected")

    audited_safe_stop = adapt_supervisor_decision_or_safe_stop(
        {**CASES[0], "lens_x_delta_mm": 0.05},
        executed_steps=0,
        remaining_step_budget=8,
        frozen_continuation_allows_next=True,
    )
    if (
        audited_safe_stop.audit.parse_valid
        or not audited_safe_stop.audit.fail_safe_applied
        or not audited_safe_stop.decision.should_stop
        or audited_safe_stop.decision.execute_frozen_cem
    ):
        raise RuntimeError("invalid output did not become an audited conservative stop")

    return {
        "status": "pass",
        "kind": "mock_wiring_only_not_closed_loop_scientific_evaluation",
        "families_exercised": ["nominal", "sensor_saturation", "secondary_reflection"],
        "decisions_dispatched": dispatched,
        "continuous_actuator_fields_emitted": 0,
        "numeric_actuator_injection_rejected": invalid_rejected,
        "invalid_output_fail_safe": {
            "parse_valid": audited_safe_stop.audit.parse_valid,
            "fail_safe_applied": audited_safe_stop.audit.fail_safe_applied,
            "error_code": audited_safe_stop.audit.error_code,
            "payload_sha256": audited_safe_stop.audit.payload_sha256,
            "controller_operation": audited_safe_stop.decision.controller_operation,
            "execute_frozen_cem": audited_safe_stop.decision.execute_frozen_cem,
            "guard_reason": audited_safe_stop.decision.guard_reason,
            "callback_dispatched": False,
        },
        "frozen_controller_modified_or_invoked": False,
        "protected_or_frozen_predictions_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts/smoke/controller_adapter_wiring.json",
    )
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
