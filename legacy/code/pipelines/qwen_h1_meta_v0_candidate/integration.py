"""Supervisor-first integration and explicit candidate loop budgets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from qwen_vl_supervisor_v1.closed_loop_adapter import (
    AuditedClosedLoopDecision,
    ClosedLoopDecision,
)

from .contracts import MetaContractError, MetaControllerDecision, parse_meta_output
from .controller import FEATURE_MODES


INTEGRATION_OPERATIONS = frozenset(
    {"dispatch_default_h1", "dispatch_meta_h1", "reobserve", "stop"}
)
MEASUREMENT_STATES = frozenset({"valid", "requires_recovery", "invalid"})


@dataclass(frozen=True)
class LoopBudget:
    measurement_steps: int
    control_steps: int
    executed_control_steps: int = 0

    def __post_init__(self) -> None:
        for name in ("measurement_steps", "control_steps", "executed_control_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

    def consume_measurement(self) -> "LoopBudget":
        if self.measurement_steps == 0:
            raise ValueError("measurement budget is exhausted")
        return LoopBudget(
            measurement_steps=self.measurement_steps - 1,
            control_steps=self.control_steps,
            executed_control_steps=self.executed_control_steps,
        )

    def consume_control(self) -> "LoopBudget":
        if self.control_steps == 0:
            raise ValueError("control budget is exhausted")
        return LoopBudget(
            measurement_steps=self.measurement_steps,
            control_steps=self.control_steps - 1,
            executed_control_steps=self.executed_control_steps + 1,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "measurement_steps": self.measurement_steps,
            "control_steps": self.control_steps,
            "executed_control_steps": self.executed_control_steps,
        }


@dataclass(frozen=True)
class IntegrationDirective:
    feature_mode: str
    operation: str
    call_meta: bool
    dispatch_h1: bool
    controller_request: str | None
    parsed_meta_decision: MetaControllerDecision | None
    reason: str
    budget_before: LoopBudget
    budget_after_authorization: LoopBudget
    measurement_policy: str
    meta_parse_valid: bool | None
    meta_parse_error_code: str | None
    security_event: bool
    security_event_code: str | None

    def __post_init__(self) -> None:
        if self.operation not in INTEGRATION_OPERATIONS:
            raise ValueError(f"invalid integration operation {self.operation!r}")
        if self.dispatch_h1 != self.operation.startswith("dispatch_"):
            raise ValueError("dispatch_h1 does not match integration operation")
        if self.security_event != (self.security_event_code is not None):
            raise ValueError("security event flag and code are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_mode": self.feature_mode,
            "operation": self.operation,
            "call_meta": self.call_meta,
            "dispatch_h1": self.dispatch_h1,
            "controller_request": self.controller_request,
            "parsed_meta_decision": (
                None
                if self.parsed_meta_decision is None
                else self.parsed_meta_decision.to_dict()
            ),
            "reason": self.reason,
            "budget_before": self.budget_before.to_dict(),
            "budget_after_authorization": self.budget_after_authorization.to_dict(),
            "measurement_policy": self.measurement_policy,
            "meta_parse_valid": self.meta_parse_valid,
            "meta_parse_error_code": self.meta_parse_error_code,
            "security_event": self.security_event,
            "security_event_code": self.security_event_code,
        }


def _closed_decision(
    supervisor: ClosedLoopDecision | AuditedClosedLoopDecision,
) -> tuple[ClosedLoopDecision, bool]:
    if isinstance(supervisor, AuditedClosedLoopDecision):
        return supervisor.decision, bool(supervisor.audit.parse_valid)
    if isinstance(supervisor, ClosedLoopDecision):
        return supervisor, True
    raise TypeError(
        "supervisor must be ClosedLoopDecision or AuditedClosedLoopDecision"
    )


def _directive(
    *,
    feature_mode: str,
    operation: str,
    call_meta: bool,
    controller_request: str | None,
    parsed_meta_decision: MetaControllerDecision | None,
    reason: str,
    budget: LoopBudget,
    budget_after: LoopBudget | None = None,
    measurement_policy: str,
    meta_parse_valid: bool | None = None,
    meta_parse_error_code: str | None = None,
    security_event_code: str | None = None,
) -> IntegrationDirective:
    return IntegrationDirective(
        feature_mode=feature_mode,
        operation=operation,
        call_meta=call_meta,
        dispatch_h1=operation.startswith("dispatch_"),
        controller_request=controller_request,
        parsed_meta_decision=parsed_meta_decision,
        reason=reason,
        budget_before=budget,
        budget_after_authorization=budget if budget_after is None else budget_after,
        measurement_policy=measurement_policy,
        meta_parse_valid=meta_parse_valid,
        meta_parse_error_code=meta_parse_error_code,
        security_event=security_event_code is not None,
        security_event_code=security_event_code,
    )


def _stop(
    *,
    feature_mode: str,
    reason: str,
    budget: LoopBudget,
    measurement_policy: str,
    parsed: MetaControllerDecision | None = None,
    call_meta: bool = False,
) -> IntegrationDirective:
    return _directive(
        feature_mode=feature_mode,
        operation="stop",
        call_meta=call_meta,
        controller_request="stop",
        parsed_meta_decision=parsed,
        reason=reason,
        budget=budget,
        measurement_policy=measurement_policy,
        meta_parse_valid=(None if not call_meta else True),
    )


def _reobserve(
    *,
    feature_mode: str,
    reason: str,
    budget: LoopBudget,
    measurement_policy: str,
    parsed: MetaControllerDecision | None = None,
    call_meta: bool = False,
) -> IntegrationDirective:
    if budget.measurement_steps == 0:
        return _stop(
            feature_mode=feature_mode,
            reason="measurement_budget_exhausted",
            budget=budget,
            measurement_policy=measurement_policy,
            parsed=parsed,
            call_meta=call_meta,
        )
    return _directive(
        feature_mode=feature_mode,
        operation="reobserve",
        call_meta=call_meta,
        controller_request="reobserve",
        parsed_meta_decision=parsed,
        reason=reason,
        budget=budget,
        budget_after=budget.consume_measurement(),
        measurement_policy=measurement_policy,
        meta_parse_valid=(None if not call_meta else True),
    )


def integrate_supervisor_step(
    supervisor: ClosedLoopDecision | AuditedClosedLoopDecision,
    *,
    feature_mode: str,
    budget: LoopBudget,
    measurement_state: str,
    meta_payload: (
        MetaControllerDecision | str | bytes | Mapping[str, Any] | None
    ) = None,
    awaiting_reobservation: bool = False,
) -> IntegrationDirective:
    """Resolve one dispatch authorization with supervisor-first precedence.

    Returned budgets are consumed at authorization time: one measurement token
    for ``reobserve`` or one control token for an H1 dispatch.  Callers must not
    dispatch twice from a single directive.
    """

    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode {feature_mode!r}")
    if measurement_state not in MEASUREMENT_STATES:
        raise ValueError(f"unknown measurement state {measurement_state!r}")
    if not isinstance(awaiting_reobservation, bool):
        raise TypeError("awaiting_reobservation must be a bool")
    closed, supervisor_parse_valid = _closed_decision(supervisor)
    if closed.executed_steps != budget.executed_control_steps:
        raise ValueError(
            "supervisor executed_steps and candidate loop budget disagree"
        )
    measurement_policy = closed.frozen_measurement_policy

    # Existing supervisor guards always have precedence, and model output is
    # not parsed at all on these branches.
    if not supervisor_parse_valid:
        return _stop(
            feature_mode=feature_mode,
            reason="invalid_supervisor_output_fail_safe_stop",
            budget=budget,
            measurement_policy=measurement_policy,
        )
    if closed.should_stop or not closed.execute_frozen_cem:
        return _stop(
            feature_mode=feature_mode,
            reason=closed.guard_reason,
            budget=budget,
            measurement_policy=measurement_policy,
        )
    if closed.effective_remaining_step_budget == 0 or budget.control_steps == 0:
        return _stop(
            feature_mode=feature_mode,
            reason="control_budget_exhausted",
            budget=budget,
            measurement_policy=measurement_policy,
        )
    supervisor_requires_recovery = (
        closed.supervisor.diagnosis != "nominal"
        or closed.supervisor.measurement_policy != "standard"
        or closed.controller_operation
        in {"reacquire_then_frozen_cem", "switch_measurement_then_frozen_cem"}
    )
    if supervisor_requires_recovery:
        return _reobserve(
            feature_mode=feature_mode,
            reason="supervisor_measurement_recovery_precedence",
            budget=budget,
            measurement_policy=measurement_policy,
        )
    if awaiting_reobservation:
        return _reobserve(
            feature_mode=feature_mode,
            reason="post_action_reobservation_required",
            budget=budget,
            measurement_policy=measurement_policy,
        )
    if measurement_state != "valid":
        return _reobserve(
            feature_mode=feature_mode,
            reason="measurement_invalid_or_requires_recovery",
            budget=budget,
            measurement_policy=measurement_policy,
        )

    if feature_mode == "off":
        return _directive(
            feature_mode=feature_mode,
            operation="dispatch_default_h1",
            call_meta=False,
            controller_request="run_default_h1",
            parsed_meta_decision=None,
            reason="feature_mode_off_unchanged_default",
            budget=budget,
            budget_after=budget.consume_control(),
            measurement_policy=measurement_policy,
        )

    parsed: MetaControllerDecision | None = None
    parse_error: str | None = None
    try:
        if meta_payload is None:
            raise MetaContractError("missing_meta_output", "meta output is missing")
        parsed = parse_meta_output(meta_payload)
    except MetaContractError as exc:
        parse_error = exc.code

    if feature_mode == "shadow":
        security_code = (
            "direct_continuous_action_injection"
            if parse_error == "direct_continuous_action_injection"
            else None
        )
        return _directive(
            feature_mode=feature_mode,
            operation="dispatch_default_h1",
            call_meta=True,
            controller_request="run_default_h1",
            parsed_meta_decision=parsed,
            reason=(
                "shadow_audit_only"
                if parse_error is None
                else "shadow_invalid_meta_audited_default_unchanged"
            ),
            budget=budget,
            budget_after=budget.consume_control(),
            measurement_policy=measurement_policy,
            meta_parse_valid=parse_error is None,
            meta_parse_error_code=parse_error,
            security_event_code=security_code,
        )

    if parsed is None:
        security_code = (
            "direct_continuous_action_injection"
            if parse_error == "direct_continuous_action_injection"
            else None
        )
        return _directive(
            feature_mode=feature_mode,
            operation="dispatch_default_h1",
            call_meta=True,
            controller_request="run_default_h1",
            parsed_meta_decision=None,
            reason=(
                "security_rejection_then_default_h1"
                if security_code is not None
                else "invalid_meta_output_then_default_h1"
            ),
            budget=budget,
            budget_after=budget.consume_control(),
            measurement_policy=measurement_policy,
            meta_parse_valid=False,
            meta_parse_error_code=parse_error,
            security_event_code=security_code,
        )

    # Stop has precedence over a simultaneous revalidation request.
    if parsed.decision == "stop":
        return _stop(
            feature_mode=feature_mode,
            reason="meta_conservative_stop",
            budget=budget,
            measurement_policy=measurement_policy,
            parsed=parsed,
            call_meta=True,
        )
    if parsed.decision == "reobserve" or parsed.observation_request == "revalidate":
        return _reobserve(
            feature_mode=feature_mode,
            reason="meta_reobservation_request",
            budget=budget,
            measurement_policy=measurement_policy,
            parsed=parsed,
            call_meta=True,
        )
    return _directive(
        feature_mode=feature_mode,
        operation="dispatch_meta_h1",
        call_meta=True,
        controller_request=parsed.decision,
        parsed_meta_decision=parsed,
        reason="valid_meta_h1_request",
        budget=budget,
        budget_after=budget.consume_control(),
        measurement_policy=measurement_policy,
        meta_parse_valid=True,
    )


__all__ = [
    "INTEGRATION_OPERATIONS",
    "IntegrationDirective",
    "LoopBudget",
    "MEASUREMENT_STATES",
    "integrate_supervisor_step",
]
