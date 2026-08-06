"""Strict candidate-only contracts for the Qwen H1 meta-controller.

The JSON files in :mod:`qwen_h1_meta_v0_candidate.protocol` are frozen.  This
module reads and verifies them, then adds the semantic checks that JSON Schema
cannot express (duplicate keys, finite numbers, actuator identity, and the
absence of continuous-action or hidden-field injection).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from continuous_control_v12.contracts import ACTION_TO_POSITION
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS


PROTOCOL_DIR = Path(__file__).resolve().parent / "protocol"
INPUT_SCHEMA_NAME = "input_schema.json"
OUTPUT_SCHEMA_NAME = "output_schema.json"

_EXPECTED_ACTION_BOUNDS = MappingProxyType(
    {
        "lens_x_delta_mm": (-0.05, 0.05),
        "lens_y_delta_mm": (-0.05, 0.05),
        "camera_x_delta_mm": (-0.02, 0.02),
        "camera_y_delta_mm": (-0.02, 0.02),
    }
)
_FORBIDDEN_VISIBLE_KEYS = frozenset(
    {
        "setup_context",
        "future_state",
        "oracle_best_action",
        "oracle_best_configuration",
        "q_star",
        "q_goal",
        "setup_id",
        "pair_id",
        "split_id",
        "file_path",
        "generator_family",
        "rollout_success",
        "future_episode_information",
        "unexecuted_default_h1_action",
        "goal_positions",
        "goal_positions_mm",
        "target_positions",
        "target_positions_mm",
    }
)
_DIRECT_ACTION_OUTPUT_KEYS = frozenset(
    {
        "action",
        "action_mm",
        "continuous_action",
        "continuous_action_mm",
        "selected_action",
        "selected_action_mm",
        "actuator_command",
        "actuator_commands",
        "command_vector",
        "initial_mean",
        "proposal_mean",
        "proposal_std",
        "mean",
        "mu",
        "sigma",
        "std",
        "variance",
        "covariance",
        "cov",
        "action_bounds",
        "bounds",
        "action_low",
        "action_high",
    }
)


class MetaContractError(ValueError):
    """Stable strict-contract rejection with a low-cardinality error code."""

    def __init__(self, code: str, message: str, *, path: str = "root") -> None:
        super().__init__(message)
        self.code = str(code)
        self.path = str(path)


@dataclass(frozen=True)
class DirectionalPrior:
    lens_x_delta_mm: str
    lens_y_delta_mm: str
    camera_x_delta_mm: str
    camera_y_delta_mm: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return tuple(getattr(self, field) for field in ACTION_FIELDS)  # type: ignore[return-value]

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in ACTION_FIELDS}


@dataclass(frozen=True)
class MetaControllerDecision:
    schema_version: str
    decision: str
    observation_request: str
    objective_profile: str
    mask_profile: str
    directional_prior: DirectionalPrior
    step_scale: str
    risk_mode: str
    confidence: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "observation_request": self.observation_request,
            "objective_profile": self.objective_profile,
            "mask_profile": self.mask_profile,
            "directional_prior": self.directional_prior.to_dict(),
            "step_scale": self.step_scale,
            "risk_mode": self.risk_mode,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class MetaControllerInput:
    """Deeply immutable validated input payload."""

    payload: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.payload)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(nested) for nested in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw(nested) for nested in value]
    return value


@lru_cache(maxsize=1)
def verify_protocol_freeze() -> Mapping[str, str]:
    """Verify every frozen protocol file against its preregistered digest."""

    manifest_path = PROTOCOL_DIR / "protocol_freeze_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot read protocol freeze manifest") from exc
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise RuntimeError("protocol freeze manifest has no file digest mapping")
    verified: dict[str, str] = {}
    for name, digest in expected.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError("protocol freeze manifest contains an unsafe filename")
        path = PROTOCOL_DIR / name
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(f"cannot read frozen protocol file {name!r}") from exc
        if actual != digest:
            raise RuntimeError(
                f"frozen protocol digest mismatch for {name!r}: "
                f"expected {digest}, got {actual}"
            )
        verified[name] = actual
    return MappingProxyType(verified)


@lru_cache(maxsize=None)
def load_protocol_json(name: str) -> Mapping[str, Any]:
    """Return an immutable verified protocol document by basename."""

    verify_protocol_freeze()
    if Path(name).name != name or name not in verify_protocol_freeze():
        raise ValueError(f"unknown frozen protocol document: {name!r}")
    try:
        decoded = json.loads((PROTOCOL_DIR / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot decode frozen protocol file {name!r}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"frozen protocol file {name!r} is not an object")
    return _freeze(decoded)


def _duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise MetaContractError(
                "duplicate_json_key", f"duplicate JSON key: {key!r}"
            )
        output[key] = value
    return output


def _reject_nonfinite_constant(token: str) -> None:
    raise MetaContractError(
        "non_finite_number", f"non-finite JSON numeric token is forbidden: {token}"
    )


def _decode_payload(payload: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise MetaContractError("invalid_utf8", "payload is not valid UTF-8") from exc
    if isinstance(payload, str):
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_duplicate_rejector,
                parse_constant=_reject_nonfinite_constant,
            )
        except MetaContractError:
            raise
        except (json.JSONDecodeError, TypeError) as exc:
            raise MetaContractError(
                "invalid_json", "payload must be exactly one valid JSON value"
            ) from exc
    elif isinstance(payload, Mapping):
        decoded = copy.deepcopy(dict(payload))
    else:
        raise MetaContractError(
            "invalid_payload_type",
            "payload must be a JSON object, JSON string, or UTF-8 JSON bytes",
        )
    if not isinstance(decoded, dict):
        raise MetaContractError(
            "invalid_json_top_level", "payload must be exactly one JSON object"
        )
    return decoded


def _normalized_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _walk_keys_and_numbers(
    value: Any,
    *,
    path: str = "root",
    reject_hidden: bool,
    reject_direct_action: bool,
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if reject_hidden and normalized in _FORBIDDEN_VISIBLE_KEYS:
                raise MetaContractError(
                    "forbidden_visible_field",
                    f"forbidden model-visible field at {child_path}",
                    path=child_path,
                )
            if reject_direct_action and normalized in _DIRECT_ACTION_OUTPUT_KEYS:
                raise MetaContractError(
                    "direct_continuous_action_injection",
                    f"continuous-action field is forbidden at {child_path}",
                    path=child_path,
                )
            _walk_keys_and_numbers(
                nested,
                path=child_path,
                reject_hidden=reject_hidden,
                reject_direct_action=reject_direct_action,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _walk_keys_and_numbers(
                nested,
                path=f"{path}[{index}]",
                reject_hidden=reject_hidden,
                reject_direct_action=reject_direct_action,
            )
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        raise MetaContractError(
            "non_finite_number",
            f"non-finite numeric value at {path}",
            path=path,
        )


@lru_cache(maxsize=2)
def _validator(schema_name: str) -> Draft202012Validator:
    schema = _thaw(load_protocol_json(schema_name))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_schema(decoded: Mapping[str, Any], schema_name: str) -> None:
    errors = sorted(
        _validator(schema_name).iter_errors(decoded),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if not errors:
        return
    error = errors[0]
    path = "root" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in error.absolute_path
    )
    raise MetaContractError(
        "schema_validation",
        f"{path}: {error.message}",
        path=path,
    )


def _validate_input_semantics(decoded: Mapping[str, Any]) -> None:
    image = decoded["current_beam_image"]
    if image["width_px"] != 1024 or image["height_px"] != 1024:
        raise MetaContractError(
            "invalid_image_role",
            "current meta-controller image must be the 1024x1024 sensor-frame image",
            path="root.current_beam_image",
        )

    semantics = decoded["actuator_semantics"]
    by_action: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(semantics):
        action_id = row["action_id"]
        if action_id in by_action:
            raise MetaContractError(
                "duplicate_actuator_semantics",
                f"duplicate actuator semantics for {action_id!r}",
                path=f"root.actuator_semantics[{index}]",
            )
        by_action[action_id] = row
    if set(by_action) != set(ACTION_FIELDS):
        raise MetaContractError(
            "invalid_actuator_semantics",
            "actuator semantics must cover each canonical action exactly once",
            path="root.actuator_semantics",
        )
    for action_id in ACTION_FIELDS:
        row = by_action[action_id]
        if row["position_id"] != ACTION_TO_POSITION[action_id]:
            raise MetaContractError(
                "invalid_actuator_semantics",
                f"{action_id!r} must map to {ACTION_TO_POSITION[action_id]!r}",
                path="root.actuator_semantics",
            )
        actual_bounds = tuple(float(value) for value in row["legal_per_step_bounds_mm"])
        if actual_bounds != _EXPECTED_ACTION_BOUNDS[action_id]:
            raise MetaContractError(
                "invalid_actuator_semantics",
                f"{action_id!r} has non-canonical per-step bounds",
                path="root.actuator_semantics",
            )

    for index, row in enumerate(decoded["history"]):
        expected_padding = "none" if row["valid"] else "no_history"
        if row["padding_reason"] != expected_padding:
            raise MetaContractError(
                "invalid_history_padding",
                "history valid flag and padding_reason are inconsistent",
                path=f"root.history[{index}]",
            )
        if not row["valid"]:
            padded_numeric_groups = (
                row["executed_action_mm"],
                row["measured_beam_delta"],
                row["predicted_beam_delta"],
                row["prediction_residual"],
                row["ensemble_uncertainty"],
            )
            if any(
                float(value) != 0.0
                for group in padded_numeric_groups
                for value in group.values()
            ):
                raise MetaContractError(
                    "nonzero_history_padding",
                    "no_history padding must contain only zero numeric values",
                    path=f"root.history[{index}]",
                )
        uncertainty = [float(row["ensemble_uncertainty"][field]) for field in STATE_FIELDS]
        if any(value < 0 for value in uncertainty):
            raise MetaContractError(
                "negative_uncertainty",
                "ensemble uncertainty must be nonnegative",
                path=f"root.history[{index}].ensemble_uncertainty",
            )

    forward = decoded["forward_uncertainty"]
    per_metric = np.asarray(
        [float(forward["per_metric"][field]) for field in STATE_FIELDS],
        dtype=np.float64,
    )
    if np.any(per_metric < 0):
        raise MetaContractError(
            "negative_uncertainty",
            "forward uncertainty must be nonnegative",
            path="root.forward_uncertainty.per_metric",
        )
    if not np.isclose(float(forward["mean"]), float(per_metric.mean()), atol=1e-12, rtol=1e-12):
        raise MetaContractError(
            "inconsistent_uncertainty_summary",
            "forward uncertainty mean does not match per_metric",
            path="root.forward_uncertainty.mean",
        )
    if not np.isclose(float(forward["maximum"]), float(per_metric.max()), atol=1e-12, rtol=1e-12):
        raise MetaContractError(
            "inconsistent_uncertainty_summary",
            "forward uncertainty maximum does not match per_metric",
            path="root.forward_uncertainty.maximum",
        )


def parse_meta_input(
    payload: str | bytes | Mapping[str, Any] | MetaControllerInput,
) -> MetaControllerInput:
    """Strictly parse and semantically validate a runtime-visible input."""

    if isinstance(payload, MetaControllerInput):
        decoded = payload.to_dict()
    else:
        decoded = _decode_payload(payload)
    _walk_keys_and_numbers(
        decoded, reject_hidden=True, reject_direct_action=False
    )
    _validate_schema(decoded, INPUT_SCHEMA_NAME)
    _validate_input_semantics(decoded)
    return MetaControllerInput(payload=_freeze(decoded))


def parse_meta_output(
    payload: str | bytes | Mapping[str, Any] | MetaControllerDecision,
) -> MetaControllerDecision:
    """Strictly parse the finite, actuator-value-free Qwen output."""

    if isinstance(payload, MetaControllerDecision):
        decoded = payload.to_dict()
    else:
        decoded = _decode_payload(payload)
    _walk_keys_and_numbers(
        decoded, reject_hidden=True, reject_direct_action=True
    )
    _validate_schema(decoded, OUTPUT_SCHEMA_NAME)
    directions = decoded["directional_prior"]
    return MetaControllerDecision(
        schema_version=decoded["schema_version"],
        decision=decoded["decision"],
        observation_request=decoded["observation_request"],
        objective_profile=decoded["objective_profile"],
        mask_profile=decoded["mask_profile"],
        directional_prior=DirectionalPrior(
            **{field: directions[field] for field in ACTION_FIELDS}
        ),
        step_scale=decoded["step_scale"],
        risk_mode=decoded["risk_mode"],
        confidence=decoded["confidence"],
        reason_codes=tuple(decoded["reason_codes"]),
    )


__all__ = [
    "DirectionalPrior",
    "MetaContractError",
    "MetaControllerDecision",
    "MetaControllerInput",
    "load_protocol_json",
    "parse_meta_input",
    "parse_meta_output",
    "verify_protocol_freeze",
]
