"""Deterministic compiler from discrete Qwen output to bounded H1 guidance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from continuous_control_v12.contracts import Bounds, metrics_vector
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS

from .contracts import MetaControllerDecision, load_protocol_json, parse_meta_output


class H1OnlyViolation(ValueError):
    """Raised before dispatch whenever a planner is not exactly horizon one."""


_RAW_MAPPING = load_protocol_json("compiler_mapping.json")
if tuple(_RAW_MAPPING["canonical_metric_order"]) != tuple(STATE_FIELDS):
    raise RuntimeError("frozen compiler metric order differs from repository contract")
if tuple(_RAW_MAPPING["canonical_action_order"]) != tuple(ACTION_FIELDS):
    raise RuntimeError("frozen compiler action order differs from repository contract")

OBJECTIVE_PROFILES = MappingProxyType(
    {
        name: tuple(float(value) for value in values)
        for name, values in _RAW_MAPPING["objective_profiles"].items()
    }
)
MASK_PROFILES = MappingProxyType(
    {
        name: tuple(str(value) for value in values)
        for name, values in _RAW_MAPPING["mask_profiles"].items()
    }
)
STEP_SCALES = MappingProxyType(
    {name: float(value) for name, value in _RAW_MAPPING["step_scales"].items()}
)
RISK_MODES = MappingProxyType(
    {
        name: MappingProxyType(
            {key: float(value) for key, value in values.items()}
        )
        for name, values in _RAW_MAPPING["risk_modes"].items()
    }
)
DIRECTION_MAPPING = MappingProxyType(
    {
        name: float(value)
        for name, value in _RAW_MAPPING["direction_mapping"].items()
        if name in {"increase", "decrease", "hold", "unknown"}
    }
)
INITIAL_MEAN_FRACTION = float(
    _RAW_MAPPING["direction_mapping"][
        "initial_mean_fraction_of_compiled_positive_bound"
    ]
)
ARBITER_CONFIG = MappingProxyType(
    {key: value for key, value in _RAW_MAPPING["arbiter"].items()}
)
_DEFAULT_SOURCE = _RAW_MAPPING["default_h1_source"]


def default_h1_config() -> dict[str, Any]:
    """Return a new exact locked-primary H1 planner configuration."""

    return {
        "horizon": 1,
        "population": int(_DEFAULT_SOURCE["population"]),
        "elites": int(_DEFAULT_SOURCE["elites"]),
        "cem_iterations": int(_DEFAULT_SOURCE["cem_iterations"]),
        "mean_error_weight": float(_DEFAULT_SOURCE["mean_error_weight"]),
        "movement_weight": float(_DEFAULT_SOURCE["movement_weight"]),
        "limit_penalty": float(_DEFAULT_SOURCE["limit_penalty"]),
        "boundary_penalty": float(_DEFAULT_SOURCE["boundary_penalty"]),
        "uncertainty_weight": float(_DEFAULT_SOURCE["uncertainty_weight"]),
    }


def dual_budget_h1_config() -> dict[str, Any]:
    """Return the sole preregistered compute-fair dual-proposal baseline."""

    config = default_h1_config()
    config["population"] = 48
    config["cem_iterations"] = 3
    return config


def assert_h1_only_config(config: Mapping[str, Any]) -> None:
    """Reject H3 and every other horizon before constructing a planner."""

    horizon = config.get("horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise H1OnlyViolation("candidate planner horizon must be the integer 1")
    if int(horizon) != 1:
        raise H1OnlyViolation(
            f"candidate permits H1 only; received horizon={horizon!r}"
        )


def validate_preregistered_controller_config(
    config: Mapping[str, Any], *, profile: str
) -> dict[str, Any]:
    """Fail closed unless a controller uses an exact preregistered H1 profile.

    ``assert_h1_only_config`` is intentionally a narrow horizon guard used by
    low-level planner helpers.  The public meta-controller has a stronger
    contract: all default-objective and CEM-budget fields must match either the
    locked primary H1 profile or the explicitly named compute-fair dual-budget
    baseline.
    """

    assert_h1_only_config(config)
    profiles = {
        "default_h1": default_h1_config(),
        "dual_budget_default_h1": dual_budget_h1_config(),
    }
    if profile not in profiles:
        raise H1OnlyViolation(
            "controller planner profile must be default_h1 or "
            "dual_budget_default_h1"
        )
    supplied = dict(config)
    expected = profiles[profile]
    if set(supplied) != set(expected):
        raise H1OnlyViolation(
            f"{profile} planner fields differ from the preregistered registry"
        )
    if supplied != expected:
        raise H1OnlyViolation(
            f"{profile} planner values differ from the preregistered registry"
        )
    return dict(expected)


@dataclass(frozen=True)
class CompiledH1Guidance:
    """Complete deterministic, continuous-value-free compilation result."""

    requested_decision: str
    effective_decision: str
    observation_request: str
    objective_profile: str
    objective_multipliers: tuple[float, float, float, float, float]
    mask_profile: str
    allowed_dofs: tuple[str, ...]
    directional_prior: tuple[str, str, str, str]
    initial_mean_mm: tuple[float, float, float, float]
    step_scale: str
    action_bound_scale: float
    risk_mode: str
    uncertainty_weight_multiplier: float
    confidence: str
    reason_codes: tuple[str, ...]
    fallback_reason: str | None

    @property
    def is_guided(self) -> bool:
        return self.effective_decision == "run_guided_h1"

    def bounds_for(self, default_bounds: Bounds) -> Bounds:
        default_bounds.validate()
        scale = float(self.action_bound_scale)
        if not (0.0 < scale <= 1.0):
            raise ValueError("compiled action-bound scale must be in (0, 1]")
        compiled = Bounds(
            action_low=np.asarray(default_bounds.action_low, dtype=np.float64) * scale,
            action_high=np.asarray(default_bounds.action_high, dtype=np.float64) * scale,
            position_low=np.asarray(default_bounds.position_low, dtype=np.float64).copy(),
            position_high=np.asarray(default_bounds.position_high, dtype=np.float64).copy(),
        )
        compiled.validate()
        if np.any(compiled.action_low < default_bounds.action_low - 1e-15) or np.any(
            compiled.action_high > default_bounds.action_high + 1e-15
        ):
            raise ValueError("compiler expanded a canonical action bound")
        return compiled

    def planner_config(self, default_config: Mapping[str, Any]) -> dict[str, Any]:
        assert_h1_only_config(default_config)
        compiled = dict(default_config)
        base_uncertainty = float(default_config["uncertainty_weight"])
        compiled["uncertainty_weight"] = (
            base_uncertainty * float(self.uncertainty_weight_multiplier)
        )
        if compiled["uncertainty_weight"] + 1e-15 < base_uncertainty:
            raise ValueError("compiler reduced uncertainty weight")
        return compiled

    def objective_view(
        self,
        predicted_next: Mapping[str, Any] | Sequence[float],
        target: Mapping[str, Any] | Sequence[float],
    ) -> np.ndarray:
        predicted = metrics_vector(predicted_next)
        if self.objective_profile == "balanced":
            # Avoid arithmetic so the balanced view is bitwise identical.
            return predicted.copy()
        target_values = metrics_vector(target)
        multipliers = np.asarray(self.objective_multipliers, dtype=np.float64)
        return target_values + multipliers * (predicted - target_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_decision": self.requested_decision,
            "effective_decision": self.effective_decision,
            "observation_request": self.observation_request,
            "objective_profile": self.objective_profile,
            "objective_multipliers": list(self.objective_multipliers),
            "mask_profile": self.mask_profile,
            "allowed_dofs": list(self.allowed_dofs),
            "directional_prior": dict(zip(ACTION_FIELDS, self.directional_prior, strict=True)),
            "initial_mean_mm": dict(zip(ACTION_FIELDS, self.initial_mean_mm, strict=True)),
            "step_scale": self.step_scale,
            "action_bound_scale": self.action_bound_scale,
            "risk_mode": self.risk_mode,
            "uncertainty_weight_multiplier": self.uncertainty_weight_multiplier,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "fallback_reason": self.fallback_reason,
        }


def compile_guidance(
    decision: MetaControllerDecision | str | bytes | Mapping[str, Any],
    *,
    default_bounds: Bounds,
    default_config: Mapping[str, Any] | None = None,
) -> CompiledH1Guidance:
    """Compile one strict discrete decision into an immutable H1 guide."""

    parsed = parse_meta_output(decision)
    config = default_h1_config() if default_config is None else default_config
    assert_h1_only_config(config)
    default_bounds.validate()

    requested = parsed.decision
    effective = requested
    fallback_reason: str | None = None
    if requested == "run_guided_h1" and parsed.confidence == "low":
        effective = "run_default_h1"
        fallback_reason = "low_confidence_guidance"

    if effective != "run_guided_h1":
        objective_profile = "balanced"
        mask_profile = "default"
        directions = ("unknown", "unknown", "unknown", "unknown")
        step_scale = "default"
        risk_mode = "standard"
    else:
        objective_profile = parsed.objective_profile
        mask_profile = parsed.mask_profile
        directions = parsed.directional_prior.as_tuple()
        step_scale = parsed.step_scale
        risk_mode = parsed.risk_mode

    objective = tuple(OBJECTIVE_PROFILES[objective_profile])
    if len(objective) != 5:
        raise RuntimeError("frozen objective profile must contain five multipliers")
    scale = STEP_SCALES[step_scale] * float(
        RISK_MODES[risk_mode]["trust_region_multiplier"]
    )
    if not (0.0 < scale <= 1.0):
        raise RuntimeError("frozen compiler mapping expands the action trust region")
    compiled_high = np.asarray(default_bounds.action_high, dtype=np.float64) * scale
    direction_values = np.asarray(
        [DIRECTION_MAPPING[direction] for direction in directions], dtype=np.float64
    )
    initial_mean = direction_values * INITIAL_MEAN_FRACTION * compiled_high

    return CompiledH1Guidance(
        requested_decision=requested,
        effective_decision=effective,
        observation_request=parsed.observation_request,
        objective_profile=objective_profile,
        objective_multipliers=objective,  # type: ignore[arg-type]
        mask_profile=mask_profile,
        allowed_dofs=MASK_PROFILES[mask_profile],
        directional_prior=directions,  # type: ignore[arg-type]
        initial_mean_mm=tuple(float(value) for value in initial_mean),  # type: ignore[arg-type]
        step_scale=step_scale,
        action_bound_scale=float(scale),
        risk_mode=risk_mode,
        uncertainty_weight_multiplier=float(
            RISK_MODES[risk_mode]["uncertainty_weight_multiplier"]
        ),
        confidence=parsed.confidence,
        reason_codes=parsed.reason_codes,
        fallback_reason=fallback_reason,
    )


__all__ = [
    "ARBITER_CONFIG",
    "CompiledH1Guidance",
    "DIRECTION_MAPPING",
    "H1OnlyViolation",
    "MASK_PROFILES",
    "OBJECTIVE_PROFILES",
    "RISK_MODES",
    "STEP_SCALES",
    "assert_h1_only_config",
    "compile_guidance",
    "default_h1_config",
    "dual_budget_h1_config",
    "validate_preregistered_controller_config",
]
