from __future__ import annotations

import copy

import pytest

from qwen_h1_meta_v0_candidate.integration import (
    INTEGRATION_OPERATIONS,
    LoopBudget,
    integrate_supervisor_step,
)
from qwen_vl_supervisor_v1.closed_loop_adapter import (
    adapt_supervisor_decision,
    adapt_supervisor_decision_or_safe_stop,
)


def supervisor(
    diagnosis="nominal",
    policy="standard",
    action="execute",
    *,
    executed=0,
    remaining=4,
    continuation=True,
):
    return adapt_supervisor_decision(
        {
            "diagnosis": diagnosis,
            "measurement_policy": policy,
            "supervisor_action": action,
        },
        executed_steps=executed,
        remaining_step_budget=remaining,
        frozen_continuation_allows_next=continuation,
    )


def budget(measurement=2, control=4, executed=0) -> LoopBudget:
    return LoopBudget(measurement, control, executed)


def test_supervisor_stop_and_invalid_output_have_absolute_precedence(valid_output) -> None:
    stopped = integrate_supervisor_step(
        supervisor(action="stop"),
        feature_mode="guarded",
        budget=budget(),
        measurement_state="valid",
        meta_payload={"mu": [0.1]},
    )
    assert stopped.operation == "stop"
    assert not stopped.call_meta
    assert not stopped.security_event
    assert stopped.budget_after_authorization == budget()

    invalid_supervisor = adapt_supervisor_decision_or_safe_stop(
        '{"diagnosis":"nominal","selected_action":[0.1]}',
        executed_steps=0,
        remaining_step_budget=4,
        frozen_continuation_allows_next=True,
    )
    invalid = integrate_supervisor_step(
        invalid_supervisor,
        feature_mode="guarded",
        budget=budget(),
        measurement_state="valid",
        meta_payload=valid_output,
    )
    assert invalid.operation == "stop"
    assert invalid.reason == "invalid_supervisor_output_fail_safe_stop"
    assert not invalid.call_meta


@pytest.mark.parametrize(
    ("diagnosis", "policy", "action"),
    [
        ("sensor_saturation", "lower_exposure_reacquire", "reacquire"),
        ("secondary_reflection", "primary_spot", "switch_measurement"),
    ],
)
def test_supervisor_recovery_precedes_meta_and_consumes_only_measurement(
    valid_output, diagnosis, policy, action
) -> None:
    before = budget(measurement=2, control=4)
    directive = integrate_supervisor_step(
        supervisor(diagnosis, policy, action),
        feature_mode="guarded",
        budget=before,
        measurement_state="valid",
        meta_payload=valid_output,
    )
    assert directive.operation == "reobserve"
    assert not directive.call_meta
    assert directive.budget_after_authorization == budget(measurement=1, control=4)


def test_invalid_measurement_and_post_action_state_force_reobservation(valid_output) -> None:
    for measurement_state, awaiting in (
        ("requires_recovery", False),
        ("invalid", False),
        ("valid", True),
    ):
        directive = integrate_supervisor_step(
            supervisor(),
            feature_mode="guarded",
            budget=budget(),
            measurement_state=measurement_state,
            awaiting_reobservation=awaiting,
            meta_payload=valid_output,
        )
        assert directive.operation == "reobserve"
        assert not directive.call_meta
        assert directive.budget_after_authorization.measurement_steps == 1


def test_measurement_or_control_budget_exhaustion_stops(valid_output) -> None:
    no_measurements = integrate_supervisor_step(
        supervisor(),
        feature_mode="guarded",
        budget=budget(measurement=0),
        measurement_state="invalid",
        meta_payload=valid_output,
    )
    assert no_measurements.operation == "stop"
    assert no_measurements.reason == "measurement_budget_exhausted"

    no_control = integrate_supervisor_step(
        supervisor(),
        feature_mode="guarded",
        budget=budget(control=0),
        measurement_state="valid",
        meta_payload=valid_output,
    )
    assert no_control.operation == "stop"
    assert no_control.reason == "control_budget_exhausted"


def test_repeated_anomaly_recovery_exhausts_only_measurement_budget() -> None:
    current_budget = budget(measurement=3, control=4)
    recovery_supervisor = supervisor(
        "sensor_saturation", "lower_exposure_reacquire", "reacquire"
    )
    for expected_remaining in (2, 1, 0):
        directive = integrate_supervisor_step(
            recovery_supervisor,
            feature_mode="guarded",
            budget=current_budget,
            measurement_state="requires_recovery",
            meta_payload={"mu": [0.1, 0.2]},
        )
        assert directive.operation == "reobserve"
        assert not directive.call_meta
        assert not directive.security_event
        assert directive.budget_after_authorization.control_steps == 4
        assert directive.budget_after_authorization.measurement_steps == expected_remaining
        current_budget = directive.budget_after_authorization

    exhausted = integrate_supervisor_step(
        recovery_supervisor,
        feature_mode="guarded",
        budget=current_budget,
        measurement_state="requires_recovery",
        meta_payload={"mu": [0.1, 0.2]},
    )
    assert exhausted.operation == "stop"
    assert exhausted.reason == "measurement_budget_exhausted"
    assert exhausted.budget_after_authorization == current_budget


def test_four_step_candidate_control_budget_is_monotone_and_then_stops() -> None:
    current_budget = budget(measurement=2, control=4, executed=0)
    for executed in range(4):
        directive = integrate_supervisor_step(
            supervisor(executed=executed, remaining=4 - executed),
            feature_mode="off",
            budget=current_budget,
            measurement_state="valid",
        )
        assert directive.operation == "dispatch_default_h1"
        assert directive.budget_after_authorization.control_steps == 3 - executed
        assert directive.budget_after_authorization.executed_control_steps == executed + 1
        current_budget = directive.budget_after_authorization

    exhausted = integrate_supervisor_step(
        supervisor(executed=4, remaining=0),
        feature_mode="off",
        budget=current_budget,
        measurement_state="valid",
    )
    assert exhausted.operation == "stop"
    assert not exhausted.dispatch_h1


def test_off_never_parses_meta_and_authorizes_unchanged_default() -> None:
    directive = integrate_supervisor_step(
        supervisor(),
        feature_mode="off",
        budget=budget(),
        measurement_state="valid",
        meta_payload='prefix {"mu":[0.1]} suffix',
    )
    assert directive.operation == "dispatch_default_h1"
    assert not directive.call_meta
    assert not directive.security_event
    assert directive.budget_after_authorization == budget(control=3, executed=1)


def test_shadow_audits_security_injection_but_cannot_override_default(valid_output) -> None:
    injection = copy.deepcopy(valid_output)
    injection["covariance"] = [[1.0]]
    directive = integrate_supervisor_step(
        supervisor(),
        feature_mode="shadow",
        budget=budget(),
        measurement_state="valid",
        meta_payload=injection,
    )
    assert directive.operation == "dispatch_default_h1"
    assert directive.call_meta
    assert directive.security_event
    assert directive.security_event_code == "direct_continuous_action_injection"
    assert directive.meta_parse_error_code == "direct_continuous_action_injection"


def test_guarded_invalid_json_and_continuous_injection_fallback_to_default(valid_output) -> None:
    invalid = integrate_supervisor_step(
        supervisor(),
        feature_mode="guarded",
        budget=budget(),
        measurement_state="valid",
        meta_payload="not json",
    )
    assert invalid.operation == "dispatch_default_h1"
    assert invalid.meta_parse_valid is False
    assert invalid.meta_parse_error_code == "invalid_json"
    assert not invalid.security_event

    injection = copy.deepcopy(valid_output)
    injection["bounds"] = [-1.0, 1.0]
    security = integrate_supervisor_step(
        supervisor(),
        feature_mode="guarded",
        budget=budget(),
        measurement_state="valid",
        meta_payload=injection,
    )
    assert security.operation == "dispatch_default_h1"
    assert security.security_event
    assert security.reason == "security_rejection_then_default_h1"


@pytest.mark.parametrize(
    ("decision", "observation", "operation"),
    [
        ("stop", "reuse_current", "stop"),
        ("stop", "revalidate", "stop"),
        ("reobserve", "reuse_current", "reobserve"),
        ("run_default_h1", "revalidate", "reobserve"),
        ("run_guided_h1", "revalidate", "reobserve"),
        ("run_default_h1", "reuse_current", "dispatch_meta_h1"),
        ("run_guided_h1", "reuse_current", "dispatch_meta_h1"),
    ],
)
def test_guarded_decision_observation_precedence(
    valid_output, decision, observation, operation
) -> None:
    payload = copy.deepcopy(valid_output)
    payload["decision"] = decision
    payload["observation_request"] = observation
    directive = integrate_supervisor_step(
        supervisor(),
        feature_mode="guarded",
        budget=budget(),
        measurement_state="valid",
        meta_payload=payload,
    )
    assert directive.operation == operation
    if operation.startswith("dispatch_"):
        assert directive.budget_after_authorization.control_steps == 3
        assert directive.budget_after_authorization.executed_control_steps == 1
    elif operation == "reobserve":
        assert directive.budget_after_authorization.measurement_steps == 1
    else:
        assert directive.budget_after_authorization == budget()


def test_supervisor_and_loop_step_counters_must_agree() -> None:
    with pytest.raises(ValueError, match="disagree"):
        integrate_supervisor_step(
            supervisor(executed=1, remaining=3),
            feature_mode="off",
            budget=budget(executed=0),
            measurement_state="valid",
        )


def test_integration_operation_space_contains_no_h3() -> None:
    assert all("h3" not in operation.lower() for operation in INTEGRATION_OPERATIONS)
