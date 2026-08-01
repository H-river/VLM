"""Registered deterministic tools for calibrated visual evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .visual_state_tool_v10_1 import (
    classify,
    classify_pair,
    extract_features,
    extract_pair_features,
)
from .run_adaptive_visual_tool_v10_15 import (
    adaptive_pair_answer,
    adaptive_state_answer,
)


STATE_TOOL = "measure_calibrated_visual_state_v1"
PAIR_TOOL = "measure_calibrated_visual_pair_v1"
STATE_SOURCE_ROLES = ("image_path",)
PAIR_SOURCE_ROLES = ("first_image_path", "second_image_path")
PAIR_OPTIONAL_REFERENCE_ROLES = ("signed_difference_reference_path",)

TOOL_CATALOG = {
    STATE_TOOL: {
        "operation": "classify beam position and width bands from one calibrated image",
        "required_roles": ["image_path"],
    },
    PAIR_TOOL: {
        "operation": "classify centroid, width, and intensity changes between ordered calibrated images",
        "required_roles": ["first_image_path", "second_image_path"],
    },
    "simulate_registered_forward_response_v1": {
        "operation": "compute an exact after-state from registered simulator state and action",
        "required_roles": ["state_handle", "action"],
    },
    "compare_registered_counterfactual_responses_v1": {
        "operation": "compute exact paired simulator responses from two registered states",
        "required_roles": ["scenario_a_state_handle", "scenario_b_state_handle", "shared_action"],
    },
}


def _path_from_mapping(
    visible_evidence: Mapping[str, Any], source_map: Mapping[str, Any], role: str
) -> str:
    value = visible_evidence.get(role)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing visible path for {role}")
    if source_map.get(role) != value:
        raise ValueError(f"{role} must copy the exact prompt-visible path")
    return value


def normalize_source_map(
    tool_name: str,
    visible_evidence: Mapping[str, Any],
    source_map: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate required roles and remove only exact known optional references."""

    expected = (
        set(STATE_SOURCE_ROLES)
        if tool_name == STATE_TOOL
        else set(PAIR_SOURCE_ROLES)
        if tool_name == PAIR_TOOL
        else set()
    )
    if not expected:
        raise ValueError(f"Unsupported visual tool: {tool_name}")
    missing = expected - set(source_map)
    if missing:
        raise ValueError(f"Source roles are incomplete: {sorted(missing)}")
    extras = set(source_map) - expected
    allowed_extras = set(PAIR_OPTIONAL_REFERENCE_ROLES) if tool_name == PAIR_TOOL else set()
    if not extras <= allowed_extras:
        raise ValueError(f"Source map contains unsupported extras: {sorted(extras)}")
    for role in extras:
        if source_map.get(role) != visible_evidence.get(role):
            raise ValueError(f"Optional {role} must copy the exact prompt-visible path")
    return {role: source_map[role] for role in expected}


def run_mapped_visual_tool(
    tool_name: str,
    visible_evidence: Mapping[str, Any],
    source_map: Mapping[str, Any],
    *,
    image_root: Path,
    state_calibration: Mapping[str, Any],
    pair_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate source roles, execute the registered image tool, and return compact evidence."""

    if tool_name == STATE_TOOL:
        normalized = normalize_source_map(tool_name, visible_evidence, source_map)
        path = image_root / _path_from_mapping(visible_evidence, normalized, "image_path")
        if state_calibration.get("classifier") in {
            "quality_routed_visual_v1",
            "quality_routed_visual_v2",
            "quality_routed_visual_v3",
            "quality_routed_visual_v4",
        }:
            answer, _ = adaptive_state_answer(
                path,
                state_calibration["clean_calibration"],
                state_calibration["robust_calibration"],
                state_calibration.get(
                    "dimnoise_calibration",
                    state_calibration["robust_calibration"],
                ),
                state_calibration.get(
                    "noise_calibration",
                    state_calibration["robust_calibration"],
                ),
            )
            return answer
        return classify(
            extract_features(
                path,
                sensor_crop_px=int(state_calibration.get("sensor_crop_px", 512)),
                overlay_handling=str(
                    state_calibration.get("overlay_handling", "mask_zero")
                ),
                noise_floor_sigma=float(
                    state_calibration.get("noise_floor_sigma", 0.0)
                ),
                relative_floor=float(state_calibration.get("relative_floor", 0.0)),
                denoise_passes=int(state_calibration.get("denoise_passes", 0)),
            ),
            state_calibration,
        )
    if tool_name == PAIR_TOOL:
        normalized = normalize_source_map(tool_name, visible_evidence, source_map)
        first = image_root / _path_from_mapping(
            visible_evidence, normalized, "first_image_path"
        )
        second = image_root / _path_from_mapping(
            visible_evidence, normalized, "second_image_path"
        )
        if pair_calibration.get("classifier") in {
            "quality_routed_visual_v1",
            "quality_routed_visual_v2",
            "quality_routed_visual_v3",
            "quality_routed_visual_v4",
        }:
            answer, _ = adaptive_pair_answer(
                first,
                second,
                pair_calibration["clean_calibration"],
                pair_calibration["robust_calibration"],
                pair_calibration.get("noise_width_calibration"),
                pair_calibration.get("dimnoise_width_calibration"),
            )
            return {"observed_direction_set": answer}
        return {
            "observed_direction_set": classify_pair(
                extract_pair_features(
                    first,
                    second,
                    sensor_crop_px=int(pair_calibration.get("sensor_crop_px", 512)),
                    peak_feature=str(
                        pair_calibration.get("peak_feature", "energy_squared_ratio")
                    ),
                    overlay_handling=str(
                        pair_calibration.get("overlay_handling", "mask_zero")
                    ),
                    sigma_x_power=float(pair_calibration.get("sigma_x_power", 1.0)),
                    sigma_y_power=float(pair_calibration.get("sigma_y_power", 1.0)),
                    noise_floor_sigma=float(
                        pair_calibration.get("noise_floor_sigma", 0.0)
                    ),
                    relative_floor=float(pair_calibration.get("relative_floor", 0.0)),
                    width_estimator=str(
                        pair_calibration.get("width_estimator", "moments_2d")
                    ),
                    include_differential_features=bool(
                        pair_calibration.get("include_differential_features", False)
                    ),
                    differential_floor_sigma=float(
                        pair_calibration.get("differential_floor_sigma", 3.0)
                    ),
                    denoise_passes=int(pair_calibration.get("denoise_passes", 0)),
                ),
                pair_calibration,
            )
        }
    raise ValueError(f"Unsupported visual tool: {tool_name}")
