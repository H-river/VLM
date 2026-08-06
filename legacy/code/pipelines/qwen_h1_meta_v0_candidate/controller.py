"""Additive off/shadow/guarded H1 meta-controller runtime.

No Qwen output can cross this module as a continuous action.  The existing
``CEMMPC`` remains the sole continuous planner.  ``off`` constructs it without
any wrapper or RNG mutation; candidate guidance is confined to an independent
H1 planner and a reject-only arbiter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from continuous_control_v12.contracts import (
    Bounds,
    action_dict,
    action_vector,
    apply_action,
    metrics_dict,
    metrics_vector,
    position_vector,
    tolerance_vector,
    validate_action,
    validate_positions,
)
from continuous_control_v12.mpc import CEMMPC, Predictor

from .compiler import (
    ARBITER_CONFIG,
    CompiledH1Guidance,
    assert_h1_only_config,
    compile_guidance,
    validate_preregistered_controller_config,
)
from .contracts import MetaContractError, MetaControllerDecision, parse_meta_output


FEATURE_MODES = frozenset({"off", "shadow", "guarded"})


class ControllerInvariantError(RuntimeError):
    """An invariant of the unchanged default path was violated."""


@dataclass(frozen=True)
class PhysicalActionEvaluation:
    """Physical forward prediction used only for canonical arbitration."""

    predicted_next_metrics: tuple[float, float, float, float, float]
    uncertainty: tuple[float, float, float, float, float]
    clipping_probability: float
    boundary_probability: float
    actuator_limit_probability: float
    member_next_metrics: tuple[tuple[float, float, float, float, float], ...] | None

    def next_vector(self) -> np.ndarray:
        return np.asarray(self.predicted_next_metrics, dtype=np.float64)

    def uncertainty_vector(self) -> np.ndarray:
        return np.asarray(self.uncertainty, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_next_metrics": metrics_dict(self.predicted_next_metrics),
            "uncertainty": list(self.uncertainty),
            "clipping_probability": self.clipping_probability,
            "boundary_probability": self.boundary_probability,
            "actuator_limit_probability": self.actuator_limit_probability,
            "member_next_metrics": (
                None
                if self.member_next_metrics is None
                else [metrics_dict(row) for row in self.member_next_metrics]
            ),
        }


class ActionEvaluator(Protocol):
    def __call__(
        self,
        positions_mm: np.ndarray,
        current_metrics: np.ndarray,
        action_mm: np.ndarray,
    ) -> PhysicalActionEvaluation:
        """Recompute one physical prediction without an objective view."""


@dataclass(frozen=True)
class MemberCatastropheEvaluation:
    accepted: bool
    available: bool
    maximum_error_increase_tolerances: float | None
    threshold_tolerances: float
    default_member_distances: tuple[float, ...]
    guided_member_distances: tuple[float, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "available": self.available,
            "maximum_error_increase_tolerances": self.maximum_error_increase_tolerances,
            "threshold_tolerances": self.threshold_tolerances,
            "default_member_distances": list(self.default_member_distances),
            "guided_member_distances": list(self.guided_member_distances),
            "reason": self.reason,
        }


MemberCatastrophicEvaluator = Callable[
    [PhysicalActionEvaluation, PhysicalActionEvaluation, np.ndarray, np.ndarray, float],
    MemberCatastropheEvaluation,
]


@dataclass(frozen=True)
class SafetyGateDecision:
    accepted_guided: bool
    rejection_reasons: tuple[str, ...]
    default_canonical_score: float
    guided_canonical_score: float
    guided_mean_normalized_uncertainty: float
    member_catastrophe: MemberCatastropheEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_guided": self.accepted_guided,
            "rejection_reasons": list(self.rejection_reasons),
            "default_canonical_score": self.default_canonical_score,
            "guided_canonical_score": self.guided_canonical_score,
            "guided_mean_normalized_uncertainty": self.guided_mean_normalized_uncertainty,
            "member_catastrophe": self.member_catastrophe.to_dict(),
        }


@dataclass(frozen=True)
class MetaH1ControllerResult:
    mode: str
    operation: str
    dispatch: bool
    selected_source: str | None
    selected_action: Mapping[str, float] | None
    default_plan: Mapping[str, Any] | None
    guided_plan: Mapping[str, Any] | None
    guidance: CompiledH1Guidance | None
    gate: SafetyGateDecision | None
    fallback_reason: str | None
    parse_error_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "operation": self.operation,
            "dispatch": self.dispatch,
            "selected_source": self.selected_source,
            "selected_action": (
                None if self.selected_action is None else dict(self.selected_action)
            ),
            "default_plan": (
                None if self.default_plan is None else dict(self.default_plan)
            ),
            "guided_plan": (
                None if self.guided_plan is None else dict(self.guided_plan)
            ),
            "guidance": None if self.guidance is None else self.guidance.to_dict(),
            "gate": None if self.gate is None else self.gate.to_dict(),
            "fallback_reason": self.fallback_reason,
            "parse_error_code": self.parse_error_code,
        }


def _finite_vector(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _auxiliary_scalar(
    auxiliary: Mapping[str, Any], *names: str, default: float = 0.0
) -> float:
    for name in names:
        if name not in auxiliary:
            continue
        values = np.asarray(auxiliary[name], dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values[0]):
            raise ValueError(f"auxiliary {name!r} is empty or non-finite")
        return float(values[0])
    return float(default)


def make_physical_evaluation(
    predicted_next_metrics: Mapping[str, Any] | Sequence[float],
    uncertainty: Sequence[float],
    auxiliary: Mapping[str, Any],
    *,
    member_next_metrics: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> PhysicalActionEvaluation:
    """Validate and normalize a physical forward result."""

    next_values = _finite_vector(
        "predicted_next_metrics", metrics_vector(predicted_next_metrics), (5,)
    )
    uncertainty_values = _finite_vector("uncertainty", uncertainty, (5,))
    if np.any(uncertainty_values < 0):
        raise ValueError("uncertainty must be nonnegative")
    member_values: tuple[tuple[float, float, float, float, float], ...] | None
    if member_next_metrics is None:
        member_values = None
    else:
        members = np.asarray(member_next_metrics, dtype=np.float64)
        if members.ndim != 2 or members.shape[1] != 5 or members.shape[0] < 1:
            raise ValueError("member_next_metrics must have shape (members, 5)")
        if not np.isfinite(members).all():
            raise ValueError("member_next_metrics contains non-finite values")
        member_values = tuple(
            tuple(float(value) for value in row)  # type: ignore[misc]
            for row in members
        )
    probabilities = {
        "clipping": _auxiliary_scalar(
            auxiliary, "clipping_probability", "clipping_fraction"
        ),
        "boundary": _auxiliary_scalar(
            auxiliary, "boundary_probability", "camera_boundary_probability"
        ),
        "actuator_limit": _auxiliary_scalar(
            auxiliary, "actuator_limit_probability"
        ),
    }
    for name, value in probabilities.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} probability must be in [0, 1]")
    return PhysicalActionEvaluation(
        predicted_next_metrics=tuple(float(value) for value in next_values),  # type: ignore[arg-type]
        uncertainty=tuple(float(value) for value in uncertainty_values),  # type: ignore[arg-type]
        clipping_probability=probabilities["clipping"],
        boundary_probability=probabilities["boundary"],
        actuator_limit_probability=probabilities["actuator_limit"],
        member_next_metrics=member_values,
    )


def predictor_action_evaluator(predictor: Predictor) -> ActionEvaluator:
    """Adapt a standard CEM predictor; member safety remains unavailable."""

    def evaluate(
        positions_mm: np.ndarray,
        current_metrics: np.ndarray,
        action_mm: np.ndarray,
    ) -> PhysicalActionEvaluation:
        predicted, uncertainty, auxiliary = predictor(
            positions_mm.copy(), current_metrics.copy(), action_mm.copy()
        )
        return make_physical_evaluation(predicted, uncertainty, auxiliary)

    return evaluate


def forward_ensemble_action_evaluator(
    model: Any,
    setup_context: Mapping[str, Any],
) -> ActionEvaluator:
    """Expose physical and per-member predictions from ``ForwardEnsemble``.

    The setup context remains private to the numeric model and is never added
    to the Qwen-visible contract.
    """

    if bool(getattr(model, "image_conditioning", False)):
        raise ValueError("candidate H1 evaluator requires the unchanged numeric model")

    def evaluate(
        positions_mm: np.ndarray,
        current_metrics: np.ndarray,
        action_mm: np.ndarray,
    ) -> PhysicalActionEvaluation:
        result = model.predict(
            setup_context,
            positions_mm,
            current_metrics,
            action_mm[None, :],
        )
        residuals = np.asarray(result["member_metric_residuals"], dtype=np.float64)
        if residuals.ndim != 3 or residuals.shape[1:] != (1, 5):
            raise ValueError("forward ensemble returned unexpected member residual shape")
        members = current_metrics[None, :] + residuals[:, 0, :] * tolerance_vector(
            current_metrics
        )[None, :]
        auxiliary = result["auxiliary_predictions"]
        return make_physical_evaluation(
            np.asarray(result["predicted_next_metrics"])[0],
            np.asarray(result["uncertainty"])[0],
            auxiliary,
            member_next_metrics=members,
        )

    return evaluate


def canonical_one_step_score(
    *,
    action_mm: Mapping[str, Any] | Sequence[float],
    evaluation: PhysicalActionEvaluation,
    target_metrics: Mapping[str, Any] | Sequence[float],
    tolerance_reference: Mapping[str, Any] | Sequence[float],
    default_bounds: Bounds,
    default_config: Mapping[str, Any],
    limit_projected: bool,
) -> float:
    """Exact H1 form of the existing canonical default CEM objective."""

    assert_h1_only_config(default_config)
    default_bounds.validate()
    action = action_vector(action_mm)
    validate_action(action, default_bounds)
    target = metrics_vector(target_metrics)
    reference = metrics_vector(tolerance_reference)
    normalized = np.abs(evaluation.next_vector() - target) / tolerance_vector(reference)
    terminal_max = float(normalized.max())
    terminal_mean = float(normalized.mean())
    movement = float(np.abs(action / default_bounds.action_high).sum())
    uncertainty_cost = float(evaluation.uncertainty_vector().mean())
    boundary_cost = (
        float(evaluation.clipping_probability) + float(evaluation.boundary_probability)
    )
    score = (
        terminal_max
        + float(default_config["mean_error_weight"]) * terminal_mean
        + 0.10 * terminal_max
        + float(default_config["movement_weight"]) * movement
        + float(default_config["boundary_penalty"]) * boundary_cost
        + float(default_config["uncertainty_weight"]) * uncertainty_cost
        + float(default_config["limit_penalty"]) * float(bool(limit_projected))
    )
    if not np.isfinite(score):
        raise ValueError("canonical score is non-finite")
    return float(score)


def _proposal_was_projected(plan: Mapping[str, Any]) -> bool:
    requested = action_vector(plan["selected_requested_action"])
    effective = action_vector(plan["selected_effective_action"])
    return not np.allclose(requested, effective, atol=1e-12, rtol=0.0)


def canonical_proposal_score(
    plan: Mapping[str, Any],
    evaluation: PhysicalActionEvaluation,
    *,
    target_metrics: Mapping[str, Any] | Sequence[float],
    tolerance_reference: Mapping[str, Any] | Sequence[float],
    default_bounds: Bounds,
    default_config: Mapping[str, Any],
) -> float:
    return canonical_one_step_score(
        action_mm=plan["selected_effective_action"],
        evaluation=evaluation,
        target_metrics=target_metrics,
        tolerance_reference=tolerance_reference,
        default_bounds=default_bounds,
        default_config=default_config,
        limit_projected=_proposal_was_projected(plan),
    )


def evaluate_member_catastrophic(
    default_evaluation: PhysicalActionEvaluation,
    guided_evaluation: PhysicalActionEvaluation,
    target_metrics: np.ndarray,
    tolerance_reference: np.ndarray,
    threshold_tolerances: float,
) -> MemberCatastropheEvaluation:
    """Reject if any matched ensemble member becomes catastrophically worse."""

    threshold = float(threshold_tolerances)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("catastrophic member threshold must be finite and nonnegative")
    default_members = default_evaluation.member_next_metrics
    guided_members = guided_evaluation.member_next_metrics
    if default_members is None or guided_members is None:
        return MemberCatastropheEvaluation(
            accepted=False,
            available=False,
            maximum_error_increase_tolerances=None,
            threshold_tolerances=threshold,
            default_member_distances=(),
            guided_member_distances=(),
            reason="member_predictions_unavailable",
        )
    default_array = np.asarray(default_members, dtype=np.float64)
    guided_array = np.asarray(guided_members, dtype=np.float64)
    if default_array.shape != guided_array.shape:
        return MemberCatastropheEvaluation(
            accepted=False,
            available=False,
            maximum_error_increase_tolerances=None,
            threshold_tolerances=threshold,
            default_member_distances=(),
            guided_member_distances=(),
            reason="member_prediction_shape_mismatch",
        )
    tolerances = tolerance_vector(tolerance_reference)
    default_distances = np.max(
        np.abs(default_array - target_metrics[None, :]) / tolerances[None, :],
        axis=1,
    )
    guided_distances = np.max(
        np.abs(guided_array - target_metrics[None, :]) / tolerances[None, :],
        axis=1,
    )
    maximum_increase = float(np.max(guided_distances - default_distances))
    accepted = maximum_increase <= threshold + 1e-12
    return MemberCatastropheEvaluation(
        accepted=accepted,
        available=True,
        maximum_error_increase_tolerances=maximum_increase,
        threshold_tolerances=threshold,
        default_member_distances=tuple(float(value) for value in default_distances),
        guided_member_distances=tuple(float(value) for value in guided_distances),
        reason="within_threshold" if accepted else "catastrophic_member_increase",
    )


class _FirstNormalMeanShiftGenerator:
    """Proxy one independent Generator and shift only its first normal draw."""

    def __init__(self, generator: np.random.Generator, shift: Sequence[float]) -> None:
        self._generator = generator
        self._shift = _finite_vector("initial proposal mean", shift, (4,))
        self.normal_calls = 0

    def normal(self, loc=0.0, scale=1.0, size=None):
        self.normal_calls += 1
        if self.normal_calls != 1:
            return self._generator.normal(loc, scale, size=size)
        location = np.asarray(loc, dtype=np.float64)
        if location.ndim < 1 or location.shape[-1] != 4:
            raise ControllerInvariantError(
                "first guided CEM normal draw does not end in four actuator axes"
            )
        shift_shape = (1,) * (location.ndim - 1) + (4,)
        shifted_location = location + self._shift.reshape(shift_shape)
        return self._generator.normal(shifted_location, scale, size=size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator, name)


def _guided_objective_predictor(
    predictor: Predictor,
    guidance: CompiledH1Guidance,
    target_metrics: Mapping[str, Any] | Sequence[float],
) -> Predictor:
    target = metrics_vector(target_metrics)

    def predict(
        positions: np.ndarray,
        metrics: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        physical_next, uncertainty, auxiliary = predictor(positions, metrics, action)
        return (
            guidance.objective_view(physical_next, target),
            uncertainty,
            auxiliary,
        )

    return predict


def plan_default_h1(
    *,
    default_bounds: Bounds,
    predictor: Predictor,
    default_config: Mapping[str, Any],
    seed: int,
    positions_mm: Mapping[str, Any] | Sequence[float],
    current_metrics: Mapping[str, Any] | Sequence[float],
    target_metrics: Mapping[str, Any] | Sequence[float],
    tolerance_reference: Mapping[str, Any] | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Invoke the existing default H1 planner without candidate wrappers."""

    assert_h1_only_config(default_config)
    planner = CEMMPC(
        bounds=default_bounds,
        predictor=predictor,
        config=default_config,
        seed=int(seed),
    )
    return planner.plan(
        positions_mm=positions_mm,
        current_metrics=current_metrics,
        target_metrics=target_metrics,
        allowed_dofs=("lens_x", "lens_y", "camera_x", "camera_y"),
        tolerance_reference=tolerance_reference,
    )


def plan_guided_h1(
    *,
    guidance: CompiledH1Guidance,
    default_bounds: Bounds,
    predictor: Predictor,
    default_config: Mapping[str, Any],
    seed: int,
    positions_mm: Mapping[str, Any] | Sequence[float],
    current_metrics: Mapping[str, Any] | Sequence[float],
    target_metrics: Mapping[str, Any] | Sequence[float],
    tolerance_reference: Mapping[str, Any] | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Plan one guided H1 proposal without invoking Qwen or dispatching."""

    assert_h1_only_config(default_config)
    if not guidance.is_guided:
        raise ValueError("plan_guided_h1 requires effective run_guided_h1 guidance")
    guided_bounds = guidance.bounds_for(default_bounds)
    guided_config = guidance.planner_config(default_config)
    planner = CEMMPC(
        bounds=guided_bounds,
        predictor=_guided_objective_predictor(
            predictor, guidance, target_metrics
        ),
        config=guided_config,
        seed=int(seed),
    )
    planner.rng = _FirstNormalMeanShiftGenerator(  # type: ignore[assignment]
        planner.rng, guidance.initial_mean_mm
    )
    result = planner.plan(
        positions_mm=positions_mm,
        current_metrics=current_metrics,
        target_metrics=target_metrics,
        allowed_dofs=guidance.allowed_dofs,
        tolerance_reference=tolerance_reference,
    )
    output = dict(result)
    history = result.get("iteration_history", [])
    output["guided_search_score"] = (
        None if not history else min(float(row["best_score"]) for row in history)
    )
    output["guided_search_objective_profile"] = guidance.objective_profile
    output["compiled_guidance"] = guidance.to_dict()
    return output


def reject_only_safety_gate(
    *,
    positions_mm: Mapping[str, Any] | Sequence[float],
    target_metrics: Mapping[str, Any] | Sequence[float],
    tolerance_reference: Mapping[str, Any] | Sequence[float],
    default_plan: Mapping[str, Any],
    guided_plan: Mapping[str, Any],
    default_evaluation: PhysicalActionEvaluation,
    guided_evaluation: PhysicalActionEvaluation,
    guidance: CompiledH1Guidance,
    default_bounds: Bounds,
    default_config: Mapping[str, Any],
    measurement_valid: bool,
    member_evaluator: MemberCatastrophicEvaluator = evaluate_member_catastrophic,
) -> SafetyGateDecision:
    """Compare proposals; it may reject guided but never alter default action."""

    assert_h1_only_config(default_config)
    if not isinstance(measurement_valid, bool):
        raise TypeError("measurement_valid must be a bool")
    positions = position_vector(positions_mm)
    validate_positions(positions, default_bounds)
    default_action = action_vector(default_plan["selected_effective_action"])
    try:
        validate_action(default_action, default_bounds)
        apply_action(positions, default_action, default_bounds)
    except (TypeError, ValueError) as exc:
        raise ControllerInvariantError(
            "unchanged default H1 produced an illegal action"
        ) from exc

    reasons: list[str] = []
    if not guidance.is_guided:
        reasons.append("not_effective_guided_decision")
    if not measurement_valid:
        reasons.append("measurement_invalid")
    guided_action = action_vector(guided_plan["selected_effective_action"])
    try:
        compiled_bounds = guidance.bounds_for(default_bounds)
        validate_action(guided_action, compiled_bounds)
        validate_action(guided_action, default_bounds)
        apply_action(positions, guided_action, default_bounds)
    except (TypeError, ValueError):
        reasons.append("guided_action_failed_existing_bounds")

    default_score = canonical_proposal_score(
        default_plan,
        default_evaluation,
        target_metrics=target_metrics,
        tolerance_reference=tolerance_reference,
        default_bounds=default_bounds,
        default_config=default_config,
    )
    guided_score = canonical_proposal_score(
        guided_plan,
        guided_evaluation,
        target_metrics=target_metrics,
        tolerance_reference=tolerance_reference,
        default_bounds=default_bounds,
        default_config=default_config,
    )
    margin = float(ARBITER_CONFIG["canonical_default_objective_margin"])
    if bool(ARBITER_CONFIG["require_guided_not_worse_than_default"]) and (
        guided_score > default_score + margin
    ):
        reasons.append("guided_worse_than_default_canonical_objective")

    mean_uncertainty = float(guided_evaluation.uncertainty_vector().mean())
    if mean_uncertainty > float(
        ARBITER_CONFIG["maximum_mean_normalized_uncertainty"]
    ):
        reasons.append("guided_uncertainty_above_threshold")

    member_result = member_evaluator(
        default_evaluation,
        guided_evaluation,
        metrics_vector(target_metrics),
        metrics_vector(tolerance_reference),
        float(ARBITER_CONFIG["catastrophic_member_max_error_increase_tolerances"]),
    )
    if not member_result.accepted:
        reasons.append(member_result.reason)

    return SafetyGateDecision(
        accepted_guided=not reasons,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        default_canonical_score=default_score,
        guided_canonical_score=guided_score,
        guided_mean_normalized_uncertainty=mean_uncertainty,
        member_catastrophe=member_result,
    )


class MetaH1Controller:
    """Candidate feature-mode wrapper around independent default/guided H1."""

    def __init__(
        self,
        *,
        default_bounds: Bounds,
        predictor: Predictor,
        default_config: Mapping[str, Any],
        seed: int,
        mode: str = "off",
        planner_profile: str = "default_h1",
        action_evaluator: ActionEvaluator | None = None,
        member_evaluator: MemberCatastrophicEvaluator = evaluate_member_catastrophic,
    ) -> None:
        if mode not in FEATURE_MODES:
            raise ValueError(f"unknown feature mode {mode!r}")
        default_bounds.validate()
        self.default_bounds = default_bounds
        self.predictor = predictor
        self.planner_profile = str(planner_profile)
        self.default_config = validate_preregistered_controller_config(
            default_config,
            profile=self.planner_profile,
        )
        self.seed = int(seed)
        self.mode = mode
        self.action_evaluator = (
            predictor_action_evaluator(predictor)
            if action_evaluator is None
            else action_evaluator
        )
        self.member_evaluator = member_evaluator

    @staticmethod
    def _selected(
        *,
        mode: str,
        source: str,
        default_plan: Mapping[str, Any],
        guided_plan: Mapping[str, Any] | None = None,
        guidance: CompiledH1Guidance | None = None,
        gate: SafetyGateDecision | None = None,
        fallback_reason: str | None = None,
        parse_error_code: str | None = None,
    ) -> MetaH1ControllerResult:
        plan = default_plan if source == "default" else guided_plan
        if plan is None:
            raise ControllerInvariantError("selected planner result is missing")
        return MetaH1ControllerResult(
            mode=mode,
            operation="dispatch_h1",
            dispatch=True,
            selected_source=source,
            selected_action=dict(plan["selected_effective_action"]),
            default_plan=default_plan,
            guided_plan=guided_plan,
            guidance=guidance,
            gate=gate,
            fallback_reason=fallback_reason,
            parse_error_code=parse_error_code,
        )

    def run(
        self,
        *,
        positions_mm: Mapping[str, Any] | Sequence[float],
        current_metrics: Mapping[str, Any] | Sequence[float],
        target_metrics: Mapping[str, Any] | Sequence[float],
        guidance_payload: (
            MetaControllerDecision | str | bytes | Mapping[str, Any] | None
        ) = None,
        tolerance_reference: Mapping[str, Any] | Sequence[float] | None = None,
        measurement_valid: bool = True,
    ) -> MetaH1ControllerResult:
        if not isinstance(measurement_valid, bool):
            raise TypeError("measurement_valid must be a bool")
        if not measurement_valid:
            return MetaH1ControllerResult(
                mode=self.mode,
                operation="reobserve",
                dispatch=False,
                selected_source=None,
                selected_action=None,
                default_plan=None,
                guided_plan=None,
                guidance=None,
                gate=None,
                fallback_reason="measurement_invalid",
                parse_error_code=None,
            )

        reference = current_metrics if tolerance_reference is None else tolerance_reference
        default_plan = plan_default_h1(
            default_bounds=self.default_bounds,
            predictor=self.predictor,
            default_config=self.default_config,
            seed=self.seed,
            positions_mm=positions_mm,
            current_metrics=current_metrics,
            target_metrics=target_metrics,
            tolerance_reference=reference,
        )
        if self.mode == "off":
            return self._selected(
                mode=self.mode, source="default", default_plan=default_plan
            )

        guidance: CompiledH1Guidance | None = None
        guided_plan: Mapping[str, Any] | None = None
        parse_error_code: str | None = None
        fallback_reason: str | None = None
        try:
            if guidance_payload is None:
                raise MetaContractError("missing_meta_output", "meta output is missing")
            parsed = parse_meta_output(guidance_payload)
            guidance = compile_guidance(
                parsed,
                default_bounds=self.default_bounds,
                default_config=self.default_config,
            )
        except MetaContractError as exc:
            parse_error_code = exc.code
            fallback_reason = (
                "direct_continuous_action_injection"
                if exc.code == "direct_continuous_action_injection"
                else "invalid_meta_output"
            )

        if self.mode == "shadow":
            if guidance is not None and guidance.is_guided:
                try:
                    guided_plan = plan_guided_h1(
                        guidance=guidance,
                        default_bounds=self.default_bounds,
                        predictor=self.predictor,
                        default_config=self.default_config,
                        seed=self.seed,
                        positions_mm=positions_mm,
                        current_metrics=current_metrics,
                        target_metrics=target_metrics,
                        tolerance_reference=reference,
                    )
                except Exception:  # audit-only failure must not alter default dispatch
                    fallback_reason = "shadow_guided_planner_failed"
            return self._selected(
                mode=self.mode,
                source="default",
                default_plan=default_plan,
                guided_plan=guided_plan,
                guidance=guidance,
                fallback_reason=fallback_reason,
                parse_error_code=parse_error_code,
            )

        if guidance is None:
            return self._selected(
                mode=self.mode,
                source="default",
                default_plan=default_plan,
                fallback_reason=fallback_reason,
                parse_error_code=parse_error_code,
            )
        if guidance.effective_decision == "stop":
            return MetaH1ControllerResult(
                mode=self.mode,
                operation="stop",
                dispatch=False,
                selected_source=None,
                selected_action=None,
                default_plan=default_plan,
                guided_plan=None,
                guidance=guidance,
                gate=None,
                fallback_reason=None,
                parse_error_code=None,
            )
        if guidance.observation_request == "revalidate" or (
            guidance.effective_decision == "reobserve"
        ):
            return MetaH1ControllerResult(
                mode=self.mode,
                operation="reobserve",
                dispatch=False,
                selected_source=None,
                selected_action=None,
                default_plan=default_plan,
                guided_plan=None,
                guidance=guidance,
                gate=None,
                fallback_reason=None,
                parse_error_code=None,
            )
        if not guidance.is_guided:
            return self._selected(
                mode=self.mode,
                source="default",
                default_plan=default_plan,
                guidance=guidance,
                fallback_reason=guidance.fallback_reason,
            )

        try:
            guided_plan = plan_guided_h1(
                guidance=guidance,
                default_bounds=self.default_bounds,
                predictor=self.predictor,
                default_config=self.default_config,
                seed=self.seed,
                positions_mm=positions_mm,
                current_metrics=current_metrics,
                target_metrics=target_metrics,
                tolerance_reference=reference,
            )
        except Exception:
            return self._selected(
                mode=self.mode,
                source="default",
                default_plan=default_plan,
                guidance=guidance,
                fallback_reason="guided_planner_failed",
            )

        positions = position_vector(positions_mm)
        current = metrics_vector(current_metrics)
        try:
            default_evaluation = self.action_evaluator(
                positions.copy(),
                current.copy(),
                action_vector(default_plan["selected_effective_action"]),
            )
            guided_evaluation = self.action_evaluator(
                positions.copy(),
                current.copy(),
                action_vector(guided_plan["selected_effective_action"]),
            )
            gate = reject_only_safety_gate(
                positions_mm=positions,
                target_metrics=target_metrics,
                tolerance_reference=reference,
                default_plan=default_plan,
                guided_plan=guided_plan,
                default_evaluation=default_evaluation,
                guided_evaluation=guided_evaluation,
                guidance=guidance,
                default_bounds=self.default_bounds,
                default_config=self.default_config,
                measurement_valid=True,
                member_evaluator=self.member_evaluator,
            )
        except ControllerInvariantError:
            raise
        except Exception:
            return self._selected(
                mode=self.mode,
                source="default",
                default_plan=default_plan,
                guided_plan=guided_plan,
                guidance=guidance,
                fallback_reason="physical_arbitration_unavailable",
            )
        source = "guided" if gate.accepted_guided else "default"
        return self._selected(
            mode=self.mode,
            source=source,
            default_plan=default_plan,
            guided_plan=guided_plan,
            guidance=guidance,
            gate=gate,
            fallback_reason=(None if gate.accepted_guided else "candidate_gate_rejected"),
        )


__all__ = [
    "ActionEvaluator",
    "ControllerInvariantError",
    "FEATURE_MODES",
    "MemberCatastropheEvaluation",
    "MemberCatastrophicEvaluator",
    "MetaH1Controller",
    "MetaH1ControllerResult",
    "PhysicalActionEvaluation",
    "SafetyGateDecision",
    "canonical_one_step_score",
    "canonical_proposal_score",
    "evaluate_member_catastrophic",
    "forward_ensemble_action_evaluator",
    "make_physical_evaluation",
    "plan_default_h1",
    "plan_guided_h1",
    "predictor_action_evaluator",
    "reject_only_safety_gate",
]
