"""Units, fields, bounds, hashes, and leakage guards for v12."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS, fixed_action_grid

SETUP_CONTEXT_FIELDS = (
    "wavelength_nm",
    "beam_waist_mm",
    "power_w",
    "lens_focal_length_mm",
    "lens_aperture_mm",
    "source_to_lens_mm",
    "lens_to_camera_mm",
    "pixel_size_um",
)
POSITION_FIELDS = (
    "lens_x_mm",
    "lens_y_mm",
    "camera_x_mm",
    "camera_y_mm",
)
ACTION_TO_POSITION = dict(zip(ACTION_FIELDS, POSITION_FIELDS, strict=True))
OUTPUT_FIELDS = STATE_FIELDS
LEGACY_ACTIONS = fixed_action_grid()
ZERO_ACTION_INDEX = 40
MM_TO_M = 1e-3
M_TO_MM = 1e3


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_seed(*parts: Any) -> int:
    return int(
        hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16],
        16,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def split_hash(group_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(group_ids)).encode()).hexdigest()


def setup_hash(
    setup_context: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
) -> str:
    return stable_hash(
        {
            "setup_context": {
                field: round(float(setup_context[field]), 12)
                for field in SETUP_CONTEXT_FIELDS
            },
            "simulator_fixed": simulator_fixed,
        }
    )


def context_hash(
    setup_context: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    positions_mm: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> str:
    return stable_hash(
        {
            "setup_hash": setup_hash(setup_context, simulator_fixed),
            "positions_mm": {
                field: round(float(positions_mm[field]), 12)
                for field in POSITION_FIELDS
            },
            "metrics": {
                field: round(float(metrics[field]), 9)
                for field in OUTPUT_FIELDS
            },
        }
    )


@dataclass(frozen=True)
class Bounds:
    action_low: np.ndarray
    action_high: np.ndarray
    position_low: np.ndarray
    position_high: np.ndarray

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Bounds":
        action = config["per_step_action_bounds_mm"]
        position = config["absolute_position_limits_mm"]
        return cls(
            action_low=np.asarray(
                [float(action[field][0]) for field in ACTION_FIELDS],
                dtype=np.float64,
            ),
            action_high=np.asarray(
                [float(action[field][1]) for field in ACTION_FIELDS],
                dtype=np.float64,
            ),
            position_low=np.asarray(
                [float(position[field][0]) for field in POSITION_FIELDS],
                dtype=np.float64,
            ),
            position_high=np.asarray(
                [float(position[field][1]) for field in POSITION_FIELDS],
                dtype=np.float64,
            ),
        )

    def validate(self) -> None:
        if (
            self.action_low.shape != (4,)
            or self.action_high.shape != (4,)
            or self.position_low.shape != (4,)
            or self.position_high.shape != (4,)
        ):
            raise ValueError("v12 bounds must be four-dimensional")
        if np.any(self.action_low >= self.action_high):
            raise ValueError("invalid action bounds")
        if np.any(self.position_low >= self.position_high):
            raise ValueError("invalid position bounds")
        if not np.allclose(self.action_low, -self.action_high):
            raise ValueError("per-step action bounds must be symmetric")


def action_vector(action: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    if isinstance(action, Mapping):
        return np.asarray(
            [float(action[field]) for field in ACTION_FIELDS], dtype=np.float64
        )
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("action must have four values")
    return values


def position_vector(
    positions: Mapping[str, Any] | Sequence[float],
) -> np.ndarray:
    if isinstance(positions, Mapping):
        return np.asarray(
            [float(positions[field]) for field in POSITION_FIELDS],
            dtype=np.float64,
        )
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("positions must have four values")
    return values


def action_dict(action: Sequence[float]) -> dict[str, float]:
    values = action_vector(action)
    return {
        field: float(values[index]) for index, field in enumerate(ACTION_FIELDS)
    }


def position_dict(positions: Sequence[float]) -> dict[str, float]:
    values = position_vector(positions)
    return {
        field: float(values[index])
        for index, field in enumerate(POSITION_FIELDS)
    }


def validate_action(action: Mapping[str, Any] | Sequence[float], bounds: Bounds) -> None:
    values = action_vector(action)
    if not np.isfinite(values).all():
        raise ValueError("action contains non-finite values")
    if np.any(values < bounds.action_low - 1e-12) or np.any(
        values > bounds.action_high + 1e-12
    ):
        raise ValueError("action exceeds per-step bounds")


def validate_positions(
    positions: Mapping[str, Any] | Sequence[float],
    bounds: Bounds,
) -> None:
    values = position_vector(positions)
    if not np.isfinite(values).all():
        raise ValueError("positions contain non-finite values")
    if np.any(values < bounds.position_low - 1e-12) or np.any(
        values > bounds.position_high + 1e-12
    ):
        raise ValueError("positions exceed configured absolute limits")


def project_action(
    positions: Mapping[str, Any] | Sequence[float],
    action: Mapping[str, Any] | Sequence[float],
    bounds: Bounds,
) -> np.ndarray:
    """Project onto both per-step and remaining absolute-position bounds."""

    current = position_vector(positions)
    values = np.clip(action_vector(action), bounds.action_low, bounds.action_high)
    values = np.minimum(values, bounds.position_high - current)
    values = np.maximum(values, bounds.position_low - current)
    return values.astype(np.float64)


def apply_action(
    positions: Mapping[str, Any] | Sequence[float],
    action: Mapping[str, Any] | Sequence[float],
    bounds: Bounds,
    *,
    project: bool = False,
) -> np.ndarray:
    current = position_vector(positions)
    values = project_action(current, action, bounds) if project else action_vector(action)
    validate_action(values, bounds)
    output = current + values
    validate_positions(output, bounds)
    return output


def tolerance_vector(current_metrics: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    if isinstance(current_metrics, Mapping):
        values = np.asarray(
            [float(current_metrics[field]) for field in OUTPUT_FIELDS],
            dtype=np.float64,
        )
    else:
        values = np.asarray(current_metrics, dtype=np.float64)
    if values.shape != (5,):
        raise ValueError("metrics must have five values")
    return np.asarray(
        [1.0, 1.0, 2.0, 2.0, max(0.05 * abs(float(values[4])), 1e-6)],
        dtype=np.float64,
    )


def metrics_vector(metrics: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    if isinstance(metrics, Mapping):
        return np.asarray(
            [float(metrics[field]) for field in OUTPUT_FIELDS], dtype=np.float64
        )
    values = np.asarray(metrics, dtype=np.float64)
    if values.shape != (5,):
        raise ValueError("metrics must have five values")
    return values


def metrics_dict(metrics: Sequence[float]) -> dict[str, float]:
    values = metrics_vector(metrics)
    return {
        field: float(values[index]) for index, field in enumerate(OUTPUT_FIELDS)
    }


def normalized_error(
    metrics: Mapping[str, Any] | Sequence[float],
    target: Mapping[str, Any] | Sequence[float],
    tolerance_reference: Mapping[str, Any] | Sequence[float],
) -> np.ndarray:
    return np.abs(metrics_vector(metrics) - metrics_vector(target)) / tolerance_vector(
        tolerance_reference
    )


def normalized_distance(
    metrics: Mapping[str, Any] | Sequence[float],
    target: Mapping[str, Any] | Sequence[float],
    tolerance_reference: Mapping[str, Any] | Sequence[float],
) -> float:
    return float(np.max(normalized_error(metrics, target, tolerance_reference)))


def assert_no_q_star(value: Any, *, path: str = "root") -> None:
    """Reject oracle actuator targets from every deployed input structure."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower().replace(" ", "")
            if lowered in {
                "q*",
                "q_star",
                "q_goal",
                "q_goal_mm",
                "goal_positions",
                "goal_positions_mm",
                "target_positions",
                "target_positions_mm",
            }:
                raise ValueError(f"oracle actuator target leaked into {path}.{key}")
            assert_no_q_star(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            assert_no_q_star(nested, path=f"{path}[{index}]")


def legacy_action_array() -> np.ndarray:
    return np.asarray(
        [
            [float(action[field]) for field in ACTION_FIELDS]
            for action in LEGACY_ACTIONS
        ],
        dtype=np.float64,
    )
