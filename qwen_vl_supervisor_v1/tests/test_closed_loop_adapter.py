"""Unit and mock-wiring tests for the actuator-free supervisor adapter."""

from __future__ import annotations

import json

import pytest

from qwen_vl_supervisor_v1.closed_loop_adapter import (
    FROZEN_MAX_HORIZON,
    SUPERVISOR_TO_FROZEN_POLICY,
    SupervisorDecisionError,
    adapt_and_dispatch,
    adapt_supervisor_decision,
    adapt_supervisor_decision_or_safe_stop,
    from_frozen_policy,
    parse_supervisor_decision,
    to_frozen_policy,
)


NOMINAL = {
    "diagnosis": "nominal",
    "measurement_policy": "standard",
    "supervisor_action": "execute",
}
SATURATION = {
    "diagnosis": "sensor_saturation",
    "measurement_policy": "lower_exposure_reacquire",
    "supervisor_action": "reacquire",
}
REFLECTION = {
    "diagnosis": "secondary_reflection",
    "measurement_policy": "primary_spot",
    "supervisor_action": "switch_measurement",
}


def test_policy_name_mapping_is_exact_and_reversible() -> None:
    assert dict(SUPERVISOR_TO_FROZEN_POLICY) == {
        "standard": "standard_metrics",
        "lower_exposure_reacquire": "reduce_exposure_reacquire",
        "primary_spot": "primary_spot_specialist",
    }
    for external, frozen in SUPERVISOR_TO_FROZEN_POLICY.items():
        assert to_frozen_policy(external) == frozen
        assert from_frozen_policy(frozen) == external

    with pytest.raises(SupervisorDecisionError):
        to_frozen_policy("new_unfrozen_policy")
    with pytest.raises(SupervisorDecisionError):
        from_frozen_policy("new_unfrozen_policy")


@pytest.mark.parametrize("payload", [NOMINAL, SATURATION, REFLECTION])
def test_strict_parser_accepts_only_canonical_combinations(payload: dict[str, str]) -> None:
    parsed = parse_supervisor_decision(json.dumps(payload))
    assert parsed.to_dict() == payload


@pytest.mark.parametrize(
    "payload",
    [
        {**NOMINAL, "lens_x_mm": 0.05},
        {**NOMINAL, "continuous_action": [0.05, 0.0, 0.0, 0.0]},
        {**NOMINAL, "confidence": 0.9},
        {key: value for key, value in NOMINAL.items() if key != "diagnosis"},
    ],
)
def test_parser_rejects_extra_numeric_actuator_or_missing_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(SupervisorDecisionError):
        parse_supervisor_decision(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {**NOMINAL, "diagnosis": "unknown"},
        {**NOMINAL, "measurement_policy": "standard_metrics"},
        {**NOMINAL, "supervisor_action": "move_lens"},
        {**NOMINAL, "measurement_policy": "primary_spot"},
        {**SATURATION, "supervisor_action": "continue"},
        {**REFLECTION, "supervisor_action": "execute"},
    ],
)
def test_parser_rejects_invalid_enums_and_mismatched_combinations(
    payload: dict[str, str],
) -> None:
    with pytest.raises(SupervisorDecisionError):
        parse_supervisor_decision(payload)


def test_parser_rejects_duplicate_keys_and_trailing_json() -> None:
    duplicated = (
        '{"diagnosis":"nominal","diagnosis":"sensor_saturation",'
        '"measurement_policy":"standard","supervisor_action":"execute"}'
    )
    with pytest.raises(SupervisorDecisionError, match="duplicate"):
        parse_supervisor_decision(duplicated)
    with pytest.raises(SupervisorDecisionError):
        parse_supervisor_decision(json.dumps(NOMINAL) + json.dumps(NOMINAL))


def test_frozen_continuation_denial_cannot_be_overridden() -> None:
    decision = adapt_supervisor_decision(
        {**NOMINAL, "supervisor_action": "continue"},
        executed_steps=2,
        remaining_step_budget=6,
        frozen_continuation_allows_next=False,
    )
    assert decision.should_stop is True
    assert decision.execute_frozen_cem is False
    assert decision.controller_operation == "stop"
    assert decision.guard_reason == "frozen_continuation_denied"


@pytest.mark.parametrize("executed_steps", [FROZEN_MAX_HORIZON, 9])
def test_frozen_horizon_eight_cannot_be_extended(executed_steps: int) -> None:
    decision = adapt_supervisor_decision(
        NOMINAL,
        executed_steps=executed_steps,
        remaining_step_budget=100,
        frozen_continuation_allows_next=True,
    )
    assert decision.effective_remaining_step_budget == 0
    assert decision.budget_was_clamped is True
    assert decision.should_stop is True
    assert decision.controller_operation == "stop"


def test_reported_budget_is_clamped_to_frozen_horizon() -> None:
    decision = adapt_supervisor_decision(
        NOMINAL,
        executed_steps=6,
        remaining_step_budget=99,
        frozen_continuation_allows_next=True,
    )
    assert decision.effective_remaining_step_budget == 2
    assert decision.budget_was_clamped is True
    assert decision.execute_frozen_cem is True


def test_supervisor_may_stop_conservatively_even_when_frozen_rule_allows() -> None:
    decision = adapt_supervisor_decision(
        {**REFLECTION, "supervisor_action": "stop"},
        executed_steps=0,
        remaining_step_budget=8,
        frozen_continuation_allows_next=True,
    )
    assert decision.should_stop is True
    assert decision.guard_reason == "supervisor_conservative_stop"
    assert decision.controller_operation == "stop"


@pytest.mark.parametrize(
    ("payload", "expected_frozen_policy", "expected_operation"),
    [
        (NOMINAL, "standard_metrics", "frozen_cem_step"),
        (
            SATURATION,
            "reduce_exposure_reacquire",
            "reacquire_then_frozen_cem",
        ),
        (
            REFLECTION,
            "primary_spot_specialist",
            "switch_measurement_then_frozen_cem",
        ),
    ],
)
def test_lightweight_mock_wiring_keeps_measurement_and_cem_in_callback(
    payload: dict[str, str],
    expected_frozen_policy: str,
    expected_operation: str,
) -> None:
    events: list[tuple[str, object]] = []

    def callback(decision):
        # This mock represents the integration owner.  The adapter only passes
        # a named policy/operation; the callback would own real measurement
        # switching and the frozen CEM invocation.
        if decision.controller_operation == "reacquire_then_frozen_cem":
            events.append(("reacquire", decision.frozen_measurement_policy))
        elif decision.controller_operation == "switch_measurement_then_frozen_cem":
            events.append(("switch_measurement", decision.frozen_measurement_policy))
        events.append(("run_frozen_cem", decision.execute_frozen_cem))
        return {"accepted": True, "operation": decision.controller_operation}

    result = adapt_and_dispatch(
        payload,
        callback,
        executed_steps=0,
        remaining_step_budget=8,
        frozen_continuation_allows_next=True,
    )

    assert result == {"accepted": True, "operation": expected_operation}
    assert events[-1] == ("run_frozen_cem", True)
    if len(events) == 2:
        assert events[0][1] == expected_frozen_policy


def test_stop_dispatch_does_not_invoke_mock_measurement_or_cem() -> None:
    events: list[str] = []

    def callback(decision):
        if decision.should_stop:
            events.append("stop")
            return "stopped"
        events.append("unexpected_control_call")
        return "ran"

    result = adapt_and_dispatch(
        {**SATURATION, "supervisor_action": "stop"},
        callback,
        executed_steps=1,
        remaining_step_budget=7,
        frozen_continuation_allows_next=True,
    )
    assert result == "stopped"
    assert events == ["stop"]


def test_budget_and_guard_arguments_are_strictly_typed() -> None:
    with pytest.raises(TypeError):
        adapt_supervisor_decision(
            NOMINAL,
            executed_steps=True,
            remaining_step_budget=8,
            frozen_continuation_allows_next=True,
        )
    with pytest.raises(ValueError):
        adapt_supervisor_decision(
            NOMINAL,
            executed_steps=0,
            remaining_step_budget=-1,
            frozen_continuation_allows_next=True,
        )
    with pytest.raises(TypeError):
        adapt_supervisor_decision(
            NOMINAL,
            executed_steps=0,
            remaining_step_budget=8,
            frozen_continuation_allows_next=1,
        )


@pytest.mark.parametrize(
    ("payload", "expected_error_code"),
    [
        ('{"diagnosis":', "invalid_json"),
        (["not", "an", "object"], "invalid_payload_type"),
        ({**NOMINAL, "lens_x_mm": 0.05}, "invalid_decision_keys"),
        (
            {**NOMINAL, "measurement_policy": "primary_spot"},
            "inconsistent_decision",
        ),
    ],
)
def test_fail_safe_adapter_turns_invalid_output_into_audited_stop(
    payload: object,
    expected_error_code: str,
) -> None:
    result = adapt_supervisor_decision_or_safe_stop(
        payload,
        executed_steps=2,
        remaining_step_budget=6,
        frozen_continuation_allows_next=True,
    )

    assert result.audit.parse_valid is False
    assert result.audit.fail_safe_applied is True
    assert result.audit.error_code == expected_error_code
    assert result.audit.error_message
    assert len(result.audit.payload_sha256) == 64
    assert result.decision.should_stop is True
    assert result.decision.execute_frozen_cem is False
    assert result.decision.controller_operation == "stop"
    assert result.decision.guard_reason == "invalid_supervisor_output_fail_safe_stop"


def test_fail_safe_adapter_preserves_valid_decision_and_audit_identity() -> None:
    payload = json.dumps(NOMINAL, separators=(",", ":"), sort_keys=True)
    first = adapt_supervisor_decision_or_safe_stop(
        payload,
        executed_steps=0,
        remaining_step_budget=8,
        frozen_continuation_allows_next=True,
    )
    second = adapt_supervisor_decision_or_safe_stop(
        payload,
        executed_steps=0,
        remaining_step_budget=8,
        frozen_continuation_allows_next=True,
    )

    assert first.audit.parse_valid is True
    assert first.audit.fail_safe_applied is False
    assert first.audit.error_code is None
    assert first.decision.supervisor.to_dict() == NOMINAL
    assert first.decision.execute_frozen_cem is True
    assert first.audit.payload_sha256 == second.audit.payload_sha256


def test_strict_parser_still_rejects_when_fail_safe_api_is_available() -> None:
    invalid = {**NOMINAL, "tip_rad": 0.001}
    with pytest.raises(SupervisorDecisionError):
        parse_supervisor_decision(invalid)
    with pytest.raises(SupervisorDecisionError):
        adapt_supervisor_decision(
            invalid,
            executed_steps=0,
            remaining_step_budget=8,
            frozen_continuation_allows_next=True,
        )
