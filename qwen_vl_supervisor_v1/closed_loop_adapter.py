"""Safety boundary between Qwen-VL supervisor output and frozen control code.

Qwen-VL is allowed to choose only a diagnosis, a measurement/recovery policy,
and a high-level action.  This module deliberately has no representation for a
continuous actuator command.  A caller-owned callback is responsible for any
measurement switching and for invoking the frozen H1 one-step CEM controller.

The adapter is intentionally independent of the frozen controller modules.  It
therefore cannot mutate controller state, change controller hyperparameters, or
silently replace the frozen continuation decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, TypeVar

from .contracts import (
    CANONICAL_POLICY_TO_SOURCE,
    DIAGNOSES as CONTRACT_DIAGNOSES,
    MEASUREMENT_POLICIES as CONTRACT_MEASUREMENT_POLICIES,
    SUPERVISOR_ACTIONS as CONTRACT_SUPERVISOR_ACTIONS,
)


FROZEN_MAX_HORIZON = 8

DIAGNOSES = frozenset(CONTRACT_DIAGNOSES)
MEASUREMENT_POLICIES = frozenset(CONTRACT_MEASUREMENT_POLICIES)
SUPERVISOR_ACTIONS = frozenset(CONTRACT_SUPERVISOR_ACTIONS)

_REQUIRED_OUTPUT_KEYS = frozenset(
    {"diagnosis", "measurement_policy", "supervisor_action"}
)

# This is the only terminology translation at the supervisor/controller
# boundary.  MappingProxyType prevents accidental process-local modification.
SUPERVISOR_TO_FROZEN_POLICY: Mapping[str, str] = MappingProxyType(
    dict(CANONICAL_POLICY_TO_SOURCE)
)
FROZEN_TO_SUPERVISOR_POLICY: Mapping[str, str] = MappingProxyType(
    {value: key for key, value in SUPERVISOR_TO_FROZEN_POLICY.items()}
)

_EXPECTED_POLICY_BY_DIAGNOSIS: Mapping[str, str] = MappingProxyType(
    {
        "nominal": "standard",
        "sensor_saturation": "lower_exposure_reacquire",
        "secondary_reflection": "primary_spot",
    }
)

# ``stop`` is valid for every otherwise consistent diagnosis/policy pair: the
# learned supervisor may always stop conservatively.  ``continue`` is limited
# to the nominal/standard route and remains subject to the frozen continuation
# gate below.  Static SFT examples use only execute/reacquire/switch_measurement.
_ALLOWED_ACTIONS_BY_DIAGNOSIS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "nominal": frozenset({"execute", "continue", "stop"}),
        "sensor_saturation": frozenset({"reacquire", "stop"}),
        "secondary_reflection": frozenset({"switch_measurement", "stop"}),
    }
)

_OPERATION_BY_ACTION: Mapping[str, str] = MappingProxyType(
    {
        "execute": "frozen_cem_step",
        "continue": "frozen_cem_step",
        "reacquire": "reacquire_then_frozen_cem",
        "switch_measurement": "switch_measurement_then_frozen_cem",
        "stop": "stop",
    }
)


class SupervisorDecisionError(ValueError):
    """Raised when model output violates the strict supervisor contract."""


@dataclass(frozen=True)
class SupervisorDecision:
    """Strict, actuator-free Qwen-VL output."""

    diagnosis: str
    measurement_policy: str
    supervisor_action: str

    def to_dict(self) -> dict[str, str]:
        """Return the canonical JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True)
class ClosedLoopDecision:
    """High-level directive passed to the callback that owns frozen control.

    ``controller_operation`` describes orchestration only; it is never a
    numerical control action.  If ``should_stop`` is true, the callback must not
    run CEM or acquire another measurement.
    """

    supervisor: SupervisorDecision
    frozen_measurement_policy: str
    controller_operation: str
    should_stop: bool
    execute_frozen_cem: bool
    guard_reason: str
    executed_steps: int
    effective_remaining_step_budget: int
    frozen_continuation_allows_next: bool
    budget_was_clamped: bool


@dataclass(frozen=True)
class SupervisorAdaptationAudit:
    """Audit record for strict parsing and fail-safe adaptation.

    The rejected payload is represented by a SHA-256 digest rather than copied
    into controller logs.  ``parse_valid`` is the authoritative distinction
    between a model-requested stop and a stop synthesized by the safety
    boundary.
    """

    parse_valid: bool
    fail_safe_applied: bool
    error_code: str | None
    error_message: str | None
    payload_sha256: str
    payload_kind: str


@dataclass(frozen=True)
class AuditedClosedLoopDecision:
    """A guarded directive paired with its strict-parse audit record."""

    decision: ClosedLoopDecision
    audit: SupervisorAdaptationAudit


CallbackResult = TypeVar("CallbackResult")


class FrozenControlCallback(Protocol[CallbackResult]):
    """Callback boundary whose implementation owns measurement and CEM."""

    def __call__(self, decision: ClosedLoopDecision) -> CallbackResult:
        """Consume one guarded, actuator-free high-level decision."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise SupervisorDecisionError(f"duplicate JSON key: {key!r}")
        parsed[key] = value
    return parsed


def _load_payload(payload: str | bytes | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SupervisorDecisionError("model output is not UTF-8") from exc

    if isinstance(payload, str):
        try:
            decoded = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        except SupervisorDecisionError:
            raise
        except (json.JSONDecodeError, TypeError) as exc:
            raise SupervisorDecisionError("model output is not one valid JSON value") from exc
    elif isinstance(payload, Mapping):
        decoded = dict(payload)
    else:
        raise SupervisorDecisionError(
            "model output must be a JSON object, JSON string, or UTF-8 JSON bytes"
        )

    if not isinstance(decoded, dict):
        raise SupervisorDecisionError("model output must be exactly one JSON object")
    return decoded


def _payload_audit_bytes(
    payload: Any,
) -> tuple[bytes, str]:
    """Return deterministic audit bytes without interpreting model output."""

    if isinstance(payload, bytes):
        return payload, "bytes"
    if isinstance(payload, str):
        return payload.encode("utf-8"), "str"
    if isinstance(payload, SupervisorDecision):
        payload = payload.to_dict()
        payload_kind = "SupervisorDecision"
    elif isinstance(payload, Mapping):
        payload_kind = "mapping"
    else:
        payload_kind = f"{type(payload).__module__}.{type(payload).__qualname__}"
        encoded = json.dumps(
            {"invalid_payload_type": payload_kind},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return encoded, payload_kind

    try:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        # A non-JSON mapping is itself invalid model output.  Keep this helper
        # total and deterministic without recording object reprs (which may
        # contain process-specific memory addresses).
        key_types = sorted(
            f"{type(key).__module__}.{type(key).__qualname__}"
            for key in payload.keys()
        )
        value_types = sorted(
            f"{type(value).__module__}.{type(value).__qualname__}"
            for value in payload.values()
        )
        encoded = json.dumps(
            {
                "non_json_mapping": True,
                "key_types": key_types,
                "value_types": value_types,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    return encoded, payload_kind


def _parse_error_code(exc: SupervisorDecisionError) -> str:
    """Map strict-parser messages to stable, low-cardinality audit codes."""

    message = str(exc)
    if "not UTF-8" in message:
        return "invalid_utf8"
    if "duplicate JSON key" in message:
        return "duplicate_json_key"
    if "not one valid JSON value" in message:
        return "invalid_json"
    if "exactly one JSON object" in message:
        return "invalid_json_top_level"
    if "must be a JSON object" in message:
        return "invalid_payload_type"
    if "decision keys must be exactly" in message:
        return "invalid_decision_keys"
    if "must be a string enum" in message:
        return "invalid_field_type"
    if message.startswith("invalid "):
        return "invalid_enum"
    if "inconsistent diagnosis/" in message:
        return "inconsistent_decision"
    return "strict_parse_rejected"


def parse_supervisor_decision(
    payload: str | bytes | Mapping[str, Any] | SupervisorDecision,
) -> SupervisorDecision:
    """Parse and strictly validate the three-field Qwen-VL decision.

    Extra fields are rejected, so actuator coordinates, action vectors,
    confidence values, and rationales cannot cross this interface.  The
    diagnosis, policy, and action must also form a semantically consistent
    combination.
    """

    if isinstance(payload, SupervisorDecision):
        decoded: Mapping[str, Any] = payload.to_dict()
    else:
        decoded = _load_payload(payload)

    actual_keys = frozenset(decoded)
    if actual_keys != _REQUIRED_OUTPUT_KEYS:
        missing = sorted(_REQUIRED_OUTPUT_KEYS - actual_keys)
        extra = sorted(actual_keys - _REQUIRED_OUTPUT_KEYS)
        raise SupervisorDecisionError(
            f"decision keys must be exactly {sorted(_REQUIRED_OUTPUT_KEYS)!r}; "
            f"missing={missing!r}, extra={extra!r}"
        )

    diagnosis = decoded["diagnosis"]
    measurement_policy = decoded["measurement_policy"]
    supervisor_action = decoded["supervisor_action"]
    for field_name, value in (
        ("diagnosis", diagnosis),
        ("measurement_policy", measurement_policy),
        ("supervisor_action", supervisor_action),
    ):
        if not isinstance(value, str):
            raise SupervisorDecisionError(f"{field_name} must be a string enum")

    if diagnosis not in DIAGNOSES:
        raise SupervisorDecisionError(
            f"invalid diagnosis {diagnosis!r}; expected one of {sorted(DIAGNOSES)!r}"
        )
    if measurement_policy not in MEASUREMENT_POLICIES:
        raise SupervisorDecisionError(
            "invalid measurement_policy "
            f"{measurement_policy!r}; expected one of {sorted(MEASUREMENT_POLICIES)!r}"
        )
    if supervisor_action not in SUPERVISOR_ACTIONS:
        raise SupervisorDecisionError(
            "invalid supervisor_action "
            f"{supervisor_action!r}; expected one of {sorted(SUPERVISOR_ACTIONS)!r}"
        )

    expected_policy = _EXPECTED_POLICY_BY_DIAGNOSIS[diagnosis]
    if measurement_policy != expected_policy:
        raise SupervisorDecisionError(
            "inconsistent diagnosis/policy combination: "
            f"{diagnosis!r} requires {expected_policy!r}"
        )

    allowed_actions = _ALLOWED_ACTIONS_BY_DIAGNOSIS[diagnosis]
    if supervisor_action not in allowed_actions:
        raise SupervisorDecisionError(
            "inconsistent diagnosis/action combination: "
            f"{diagnosis!r} permits only {sorted(allowed_actions)!r}"
        )

    return SupervisorDecision(
        diagnosis=diagnosis,
        measurement_policy=measurement_policy,
        supervisor_action=supervisor_action,
    )


def to_frozen_policy(supervisor_policy: str) -> str:
    """Translate an external supervisor policy to the frozen repository name."""

    try:
        return SUPERVISOR_TO_FROZEN_POLICY[supervisor_policy]
    except KeyError as exc:
        raise SupervisorDecisionError(
            f"unknown supervisor measurement policy: {supervisor_policy!r}"
        ) from exc


def from_frozen_policy(frozen_policy: str) -> str:
    """Reverse the documented policy translation exactly."""

    try:
        return FROZEN_TO_SUPERVISOR_POLICY[frozen_policy]
    except KeyError as exc:
        raise SupervisorDecisionError(
            f"unknown frozen measurement policy: {frozen_policy!r}"
        ) from exc


def _require_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def adapt_supervisor_decision(
    payload: str | bytes | Mapping[str, Any] | SupervisorDecision,
    *,
    executed_steps: int,
    remaining_step_budget: int,
    frozen_continuation_allows_next: bool,
) -> ClosedLoopDecision:
    """Apply frozen continuation and horizon guards to a supervisor decision.

    The effective budget is capped by ``8 - executed_steps``.  A caller cannot
    increase that limit by reporting a larger remaining budget.  Every
    non-stop model action is converted to a conservative stop when the frozen
    continuation policy denies another step or when the effective budget is
    exhausted.  A model-requested stop is always honored.
    """

    supervisor = parse_supervisor_decision(payload)
    executed_steps = _require_nonnegative_int("executed_steps", executed_steps)
    remaining_step_budget = _require_nonnegative_int(
        "remaining_step_budget", remaining_step_budget
    )
    if not isinstance(frozen_continuation_allows_next, bool):
        raise TypeError("frozen_continuation_allows_next must be a bool")

    frozen_horizon_remaining = max(0, FROZEN_MAX_HORIZON - executed_steps)
    effective_remaining = min(remaining_step_budget, frozen_horizon_remaining)
    budget_was_clamped = remaining_step_budget > frozen_horizon_remaining

    if supervisor.supervisor_action == "stop":
        should_stop = True
        guard_reason = "supervisor_conservative_stop"
    elif effective_remaining == 0:
        should_stop = True
        guard_reason = "frozen_horizon_or_budget_exhausted"
    elif not frozen_continuation_allows_next:
        should_stop = True
        guard_reason = "frozen_continuation_denied"
    else:
        should_stop = False
        guard_reason = "allowed_by_frozen_guards"

    controller_operation = (
        "stop"
        if should_stop
        else _OPERATION_BY_ACTION[supervisor.supervisor_action]
    )
    return ClosedLoopDecision(
        supervisor=supervisor,
        frozen_measurement_policy=to_frozen_policy(
            supervisor.measurement_policy
        ),
        controller_operation=controller_operation,
        should_stop=should_stop,
        execute_frozen_cem=not should_stop,
        guard_reason=guard_reason,
        executed_steps=executed_steps,
        effective_remaining_step_budget=effective_remaining,
        frozen_continuation_allows_next=frozen_continuation_allows_next,
        budget_was_clamped=budget_was_clamped,
    )


def adapt_supervisor_decision_or_safe_stop(
    payload: str | bytes | Mapping[str, Any] | SupervisorDecision,
    *,
    executed_steps: int,
    remaining_step_budget: int,
    frozen_continuation_allows_next: bool,
) -> AuditedClosedLoopDecision:
    """Strictly adapt valid output or return an audited conservative stop.

    This is the closed-loop-facing API for untrusted model text.  A strict
    parser rejection never reaches controller dispatch: it becomes a
    deterministic ``stop`` directive with ``execute_frozen_cem=False``.  The
    original output remains distinguishable from the synthesized directive via
    ``audit.parse_valid`` and its content digest.  Execution-context errors
    (invalid budgets or continuation-gate types) are programmer errors and are
    intentionally not converted into model-output failures.

    Callers that need rejection semantics should continue to use
    :func:`parse_supervisor_decision` or :func:`adapt_supervisor_decision`.
    """

    executed_steps = _require_nonnegative_int("executed_steps", executed_steps)
    remaining_step_budget = _require_nonnegative_int(
        "remaining_step_budget", remaining_step_budget
    )
    if not isinstance(frozen_continuation_allows_next, bool):
        raise TypeError("frozen_continuation_allows_next must be a bool")

    audit_bytes, payload_kind = _payload_audit_bytes(payload)
    payload_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    try:
        decision = adapt_supervisor_decision(
            payload,
            executed_steps=executed_steps,
            remaining_step_budget=remaining_step_budget,
            frozen_continuation_allows_next=frozen_continuation_allows_next,
        )
    except SupervisorDecisionError as exc:
        # The fallback enum tuple is valid under the same strict contract, but
        # is not attributed to the model: parse_valid=False is authoritative.
        fail_safe_supervisor = SupervisorDecision(
            diagnosis="nominal",
            measurement_policy="standard",
            supervisor_action="stop",
        )
        fallback = adapt_supervisor_decision(
            fail_safe_supervisor,
            executed_steps=executed_steps,
            remaining_step_budget=remaining_step_budget,
            frozen_continuation_allows_next=frozen_continuation_allows_next,
        )
        decision = ClosedLoopDecision(
            supervisor=fallback.supervisor,
            frozen_measurement_policy=fallback.frozen_measurement_policy,
            controller_operation="stop",
            should_stop=True,
            execute_frozen_cem=False,
            guard_reason="invalid_supervisor_output_fail_safe_stop",
            executed_steps=fallback.executed_steps,
            effective_remaining_step_budget=fallback.effective_remaining_step_budget,
            frozen_continuation_allows_next=(
                fallback.frozen_continuation_allows_next
            ),
            budget_was_clamped=fallback.budget_was_clamped,
        )
        return AuditedClosedLoopDecision(
            decision=decision,
            audit=SupervisorAdaptationAudit(
                parse_valid=False,
                fail_safe_applied=True,
                error_code=_parse_error_code(exc),
                error_message=str(exc),
                payload_sha256=payload_sha256,
                payload_kind=payload_kind,
            ),
        )

    return AuditedClosedLoopDecision(
        decision=decision,
        audit=SupervisorAdaptationAudit(
            parse_valid=True,
            fail_safe_applied=False,
            error_code=None,
            error_message=None,
            payload_sha256=payload_sha256,
            payload_kind=payload_kind,
        ),
    )


def dispatch_to_frozen_callback(
    decision: ClosedLoopDecision,
    callback: FrozenControlCallback[CallbackResult] | Callable[[ClosedLoopDecision], CallbackResult],
) -> CallbackResult:
    """Hand a guarded high-level directive to caller-owned frozen control.

    The callback must implement measurement acquisition/switching and the
    frozen CEM call.  Keeping that work outside this module makes ownership
    explicit and prevents the supervisor adapter from creating actuator values.
    """

    if not isinstance(decision, ClosedLoopDecision):
        raise TypeError("decision must be a ClosedLoopDecision")
    if not callable(callback):
        raise TypeError("callback must be callable")
    return callback(decision)


def adapt_and_dispatch(
    payload: str | bytes | Mapping[str, Any] | SupervisorDecision,
    callback: FrozenControlCallback[CallbackResult] | Callable[[ClosedLoopDecision], CallbackResult],
    *,
    executed_steps: int,
    remaining_step_budget: int,
    frozen_continuation_allows_next: bool,
) -> CallbackResult:
    """Convenience wrapper for strict parsing, safety gating, and dispatch."""

    decision = adapt_supervisor_decision(
        payload,
        executed_steps=executed_steps,
        remaining_step_budget=remaining_step_budget,
        frozen_continuation_allows_next=frozen_continuation_allows_next,
    )
    return dispatch_to_frozen_callback(decision, callback)


__all__ = [
    "AuditedClosedLoopDecision",
    "ClosedLoopDecision",
    "DIAGNOSES",
    "FROZEN_MAX_HORIZON",
    "FROZEN_TO_SUPERVISOR_POLICY",
    "FrozenControlCallback",
    "MEASUREMENT_POLICIES",
    "SUPERVISOR_ACTIONS",
    "SUPERVISOR_TO_FROZEN_POLICY",
    "SupervisorDecision",
    "SupervisorAdaptationAudit",
    "SupervisorDecisionError",
    "adapt_and_dispatch",
    "adapt_supervisor_decision",
    "adapt_supervisor_decision_or_safe_stop",
    "dispatch_to_frozen_callback",
    "from_frozen_policy",
    "parse_supervisor_decision",
    "to_frozen_policy",
]
