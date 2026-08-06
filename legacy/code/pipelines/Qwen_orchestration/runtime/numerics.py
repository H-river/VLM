"""Frozen numerical field definitions and strict conversion helpers."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .errors import ContractError


SETUP_FIELDS = (
    "wavelength_nm",
    "beam_waist_mm",
    "power_w",
    "lens_focal_length_mm",
    "lens_aperture_mm",
    "source_to_lens_mm",
    "lens_to_camera_mm",
    "lens_x_offset_mm",
    "lens_y_offset_mm",
    "camera_x_offset_mm",
    "camera_y_offset_mm",
    "pixel_size_um",
)
STATE_FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
)
ACTION_FIELDS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
DIRECTION_FIELDS = (
    "centroid_x",
    "centroid_y",
    "width_x",
    "width_y",
    "peak_intensity",
)
CLASSES = ("decrease", "no_change", "increase")

MATCHING_TOLERANCE = {
    "centroid_vector_px": 0.5,
    "width_each_px": 1.0,
    "peak_relative": 0.02,
}


def strict_numeric_mapping(
    value: Any, fields: Sequence[str], name: str
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    expected, actual = set(fields), set(value)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ContractError(f"{name} fields differ: missing={missing}, extra={extra}")
    result: dict[str, float] = {}
    for field in fields:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractError(f"{name}.{field} must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ContractError(f"{name}.{field} must be finite")
        result[field] = number
    return result


def model_feature_vector(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    action: Mapping[str, Any],
) -> np.ndarray:
    setup_out = strict_numeric_mapping(setup, SETUP_FIELDS, "setup")
    current_out = strict_numeric_mapping(current, STATE_FIELDS, "current_beam_state")
    action_out = strict_numeric_mapping(action, ACTION_FIELDS, "action")
    merged = {**setup_out, **current_out, **action_out}
    values = [
        math.log1p(max(merged[field], 0.0))
        if field == "peak_intensity"
        else merged[field]
        for field in (*SETUP_FIELDS, *STATE_FIELDS, *ACTION_FIELDS)
    ]
    return np.asarray(values, dtype=np.float32)


def inverse_feature_vector(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    desired: Mapping[str, Any],
) -> np.ndarray:
    setup_out = strict_numeric_mapping(setup, SETUP_FIELDS, "setup")
    current_out = strict_numeric_mapping(current, STATE_FIELDS, "current_beam_state")
    desired_out = strict_numeric_mapping(desired, STATE_FIELDS, "desired_beam_state")
    setup_values = [setup_out[key] for key in SETUP_FIELDS]
    current_values = [current_out[key] for key in STATE_FIELDS]
    desired_values = [desired_out[key] for key in STATE_FIELDS]
    difference = [after - before for after, before in zip(desired_values, current_values)]
    peak = max(abs(current_values[-1]), 1e-9)
    derived = [
        difference[-1] / peak,
        math.hypot(difference[0], difference[1]),
        math.hypot(setup_out["lens_x_offset_mm"], setup_out["lens_y_offset_mm"]),
        math.hypot(setup_out["camera_x_offset_mm"], setup_out["camera_y_offset_mm"]),
    ]
    return np.asarray(setup_values + current_values + desired_values + difference + derived,
                      dtype=np.float32)


def fixed_action_grid() -> list[dict[str, float]]:
    lens = (-0.05, 0.0, 0.05)
    camera = (-0.02, 0.0, 0.02)
    return [
        dict(zip(ACTION_FIELDS, map(float, values)))
        for values in itertools.product(lens, lens, camera, camera)
    ]


def state_residual(
    candidate: Mapping[str, Any],
    desired: Mapping[str, Any],
    tolerance: Mapping[str, float] = MATCHING_TOLERANCE,
) -> float:
    cx = float(candidate["centroid_x_px"]) - float(desired["centroid_x_px"])
    cy = float(candidate["centroid_y_px"]) - float(desired["centroid_y_px"])
    sx = abs(float(candidate["sigma_x_px"]) - float(desired["sigma_x_px"]))
    sy = abs(float(candidate["sigma_y_px"]) - float(desired["sigma_y_px"]))
    peak_scale = max(abs(float(desired["peak_intensity"])), 1e-12)
    peak = abs(float(candidate["peak_intensity"]) - float(desired["peak_intensity"])) / peak_scale
    normalized = (
        math.hypot(cx, cy) / tolerance["centroid_vector_px"],
        sx / tolerance["width_each_px"],
        sy / tolerance["width_each_px"],
        peak / tolerance["peak_relative"],
    )
    return math.sqrt(sum(value * value for value in normalized) / len(normalized))


def classify_residuals(
    residuals: np.ndarray, feasibility_cutoff: float, ambiguity_margin: float
) -> str:
    ordered = np.sort(residuals)
    if ordered[0] > feasibility_cutoff:
        return "infeasible_within_limits"
    if len(ordered) > 1 and ordered[1] <= ordered[0] + ambiguity_margin:
        return "ambiguous"
    return "unique"


def movement_mm(action: Mapping[str, Any]) -> float:
    return sum(abs(float(action[field])) for field in ACTION_FIELDS)


def sensor_to_legacy_initial(
    state: Mapping[str, Any], setup: Mapping[str, Any]
) -> dict[str, float]:
    result = strict_numeric_mapping(state, STATE_FIELDS, "current_beam_state")
    setup_out = strict_numeric_mapping(setup, SETUP_FIELDS, "setup")
    pitch_mm = setup_out["pixel_size_um"] / 1000.0
    result["centroid_x_px"] += setup_out["camera_x_offset_mm"] / pitch_mm
    result["centroid_y_px"] += setup_out["camera_y_offset_mm"] / pitch_mm
    return result


def legacy_to_sensor(
    state: Mapping[str, Any],
    setup: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, float]:
    result = strict_numeric_mapping(state, STATE_FIELDS, "beam_state")
    setup_out = strict_numeric_mapping(setup, SETUP_FIELDS, "setup")
    action_out = strict_numeric_mapping(action, ACTION_FIELDS, "action")
    pitch_mm = setup_out["pixel_size_um"] / 1000.0
    result["centroid_x_px"] -= (
        setup_out["camera_x_offset_mm"] + action_out["camera_x_delta_mm"]
    ) / pitch_mm
    result["centroid_y_px"] -= (
        setup_out["camera_y_offset_mm"] + action_out["camera_y_delta_mm"]
    ) / pitch_mm
    return result

