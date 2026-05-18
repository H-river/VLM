"""Build supervised reasoning records for profile2setup VLM SFT."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from profile2setup.schema import VARIABLE_ORDER, compute_delta_setup

from .image_rendering import load_intensity
from .schema import (
    CANONICAL_VARIABLE_ORDER,
    PROFILE_CHANGE_APPROXIMATELY_UNCHANGED,
    PROFILE_CHANGE_DECREASES,
    PROFILE_CHANGE_INCREASES,
    PROFILE_CHANGE_MOVES_DOWN,
    PROFILE_CHANGE_MOVES_LEFT,
    PROFILE_CHANGE_MOVES_RIGHT,
    PROFILE_CHANGE_MOVES_UP,
    PROFILE_CHANGE_UNKNOWN,
    default_reasoning_command,
)
from .validator import validate_reasoning_command

DEFAULT_PROFILE_THRESHOLDS = {
    "centroid_px": 0.5,
    "sigma_px": 0.3,
    "relative_intensity": 0.05,
    "absolute_intensity": 1e-12,
}

_DISTANCE_VARIABLES = {"source_to_lens", "lens_to_camera", "focal_length"}
_OFFSET_VARIABLES = {"lens_x", "lens_y", "camera_x", "camera_y"}
_DEFAULT_SETUP_THRESHOLDS = {
    "distance": {"unchanged": 1e-6, "small": 0.02, "medium": 0.08},
    "offset": {"unchanged": 1e-7, "small": 5e-4, "medium": 0.0015},
}


def _safe_real_array(array: Any) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"intensity_array must be 2D, got shape={arr.shape}")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"intensity_array must be numeric, got dtype={arr.dtype}")
    if np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"intensity_array must be real-valued, got dtype={arr.dtype}")
    arr = np.asarray(arr, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    return np.nan_to_num(
        arr,
        nan=0.0,
        posinf=float(np.max(finite)),
        neginf=float(np.min(finite)),
    )


def _finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out = float(value)
        if np.isfinite(out):
            return out
    return None


def compute_profile_stats(intensity_array) -> dict[str, float]:
    """Compute pixel-space profile statistics from a 2D intensity array."""
    arr = np.clip(_safe_real_array(intensity_array), a_min=0.0, a_max=None)
    height, width = arr.shape
    total = float(np.sum(arr))
    peak = float(np.max(arr)) if arr.size else 0.0

    if total <= 0.0:
        centroid_x = (width - 1) / 2.0
        centroid_y = (height - 1) / 2.0
        sigma_x = 0.0
        sigma_y = 0.0
    else:
        yy, xx = np.indices(arr.shape, dtype=np.float64)
        centroid_x = float(np.sum(arr * xx) / total)
        centroid_y = float(np.sum(arr * yy) / total)
        var_x = float(np.sum(arr * (xx - centroid_x) ** 2) / total)
        var_y = float(np.sum(arr * (yy - centroid_y) ** 2) / total)
        sigma_x = float(np.sqrt(max(var_x, 0.0)))
        sigma_y = float(np.sqrt(max(var_y, 0.0)))

    return {
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "peak": float(peak),
        "total": float(total),
    }


def _thresholds(profile_thresholds: dict | None) -> dict:
    out = dict(DEFAULT_PROFILE_THRESHOLDS)
    if profile_thresholds:
        out.update(profile_thresholds)
    return out


def _compare_absolute(delta: float | None, threshold: float, negative_label: str, positive_label: str) -> str:
    if delta is None:
        return PROFILE_CHANGE_UNKNOWN
    if delta > threshold:
        return positive_label
    if delta < -threshold:
        return negative_label
    return PROFILE_CHANGE_APPROXIMATELY_UNCHANGED


def _compare_relative(current: float | None, target: float | None, thresholds: dict) -> str:
    if current is None or target is None:
        return PROFILE_CHANGE_UNKNOWN
    delta = target - current
    threshold = max(
        float(thresholds["absolute_intensity"]),
        abs(current) * float(thresholds["relative_intensity"]),
    )
    if delta > threshold:
        return PROFILE_CHANGE_INCREASES
    if delta < -threshold:
        return PROFILE_CHANGE_DECREASES
    return PROFILE_CHANGE_APPROXIMATELY_UNCHANGED


def classify_profile_change(
    current_stats: dict | None,
    target_stats: dict | None,
    thresholds: dict | None = None,
) -> dict[str, str]:
    """Classify target-current profile changes in pixel coordinates."""
    if not isinstance(current_stats, dict) or not isinstance(target_stats, dict):
        return {
            "centroid_x": PROFILE_CHANGE_UNKNOWN,
            "centroid_y": PROFILE_CHANGE_UNKNOWN,
            "beam_width_x": PROFILE_CHANGE_UNKNOWN,
            "beam_width_y": PROFILE_CHANGE_UNKNOWN,
            "peak_intensity": PROFILE_CHANGE_UNKNOWN,
            "total_intensity": PROFILE_CHANGE_UNKNOWN,
        }

    t = _thresholds(thresholds)
    cx = _finite_float(current_stats.get("centroid_x"))
    tx = _finite_float(target_stats.get("centroid_x"))
    cy = _finite_float(current_stats.get("centroid_y"))
    ty = _finite_float(target_stats.get("centroid_y"))
    sx = _finite_float(current_stats.get("sigma_x"))
    tsx = _finite_float(target_stats.get("sigma_x"))
    sy = _finite_float(current_stats.get("sigma_y"))
    tsy = _finite_float(target_stats.get("sigma_y"))

    return {
        "centroid_x": _compare_absolute(
            None if cx is None or tx is None else tx - cx,
            float(t["centroid_px"]),
            PROFILE_CHANGE_MOVES_LEFT,
            PROFILE_CHANGE_MOVES_RIGHT,
        ),
        "centroid_y": _compare_absolute(
            None if cy is None or ty is None else ty - cy,
            float(t["centroid_px"]),
            PROFILE_CHANGE_MOVES_UP,
            PROFILE_CHANGE_MOVES_DOWN,
        ),
        "beam_width_x": _compare_absolute(
            None if sx is None or tsx is None else tsx - sx,
            float(t["sigma_px"]),
            PROFILE_CHANGE_DECREASES,
            PROFILE_CHANGE_INCREASES,
        ),
        "beam_width_y": _compare_absolute(
            None if sy is None or tsy is None else tsy - sy,
            float(t["sigma_px"]),
            PROFILE_CHANGE_DECREASES,
            PROFILE_CHANGE_INCREASES,
        ),
        "peak_intensity": _compare_relative(
            _finite_float(current_stats.get("peak")),
            _finite_float(target_stats.get("peak")),
            t,
        ),
        "total_intensity": _compare_relative(
            _finite_float(current_stats.get("total")),
            _finite_float(target_stats.get("total")),
            t,
        ),
    }


def _setup_threshold(variable: str, thresholds: dict | None) -> dict[str, float]:
    group = "distance" if variable in _DISTANCE_VARIABLES else "offset"
    out = dict(_DEFAULT_SETUP_THRESHOLDS[group])
    if thresholds:
        if isinstance(thresholds.get(group), dict):
            out.update(thresholds[group])
        if isinstance(thresholds.get(variable), dict):
            out.update(thresholds[variable])
    return {key: float(value) for key, value in out.items()}


def classify_setup_delta(target_delta: dict | None, thresholds: dict | None = None) -> dict[str, Any]:
    """Classify setup delta direction and magnitude for canonical variables."""
    direction: dict[str, str] = {}
    magnitude_bucket: dict[str, str] = {}
    changed_variables: list[str] = []
    unchanged_variables: list[str] = []

    deltas = target_delta if isinstance(target_delta, dict) else {}
    for variable in CANONICAL_VARIABLE_ORDER:
        delta = _finite_float(deltas.get(variable))
        limits = _setup_threshold(variable, thresholds)
        abs_delta = abs(delta) if delta is not None else 0.0

        if delta is None or abs_delta <= limits["unchanged"]:
            direction[variable] = "unchanged"
            magnitude_bucket[variable] = "unchanged"
            unchanged_variables.append(variable)
            continue

        direction[variable] = "increase" if delta > 0.0 else "decrease"
        if abs_delta <= limits["small"]:
            magnitude_bucket[variable] = "small"
        elif abs_delta <= limits["medium"]:
            magnitude_bucket[variable] = "medium"
        else:
            magnitude_bucket[variable] = "large"
        changed_variables.append(variable)

    return {
        "changed_variables": changed_variables,
        "unchanged_variables": unchanged_variables,
        "direction": direction,
        "magnitude_bucket": magnitude_bucket,
    }


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def infer_prompt_constraints(prompt: str | None) -> dict[str, Any]:
    """Infer simple prompt constraints and prompt-level rejection reasons."""
    text = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    fixed_variables: list[str] = []
    allowed_variables: list[str] = []
    unsupported_requests: list[str] = []
    conflict_reasons: list[str] = []

    camera_fixed = _contains_any(
        text,
        (
            "keep camera fixed",
            "camera fixed",
            "do not move camera",
        ),
    )
    camera_only = _contains_any(
        text,
        (
            "move camera only",
            "only camera",
            "only move camera",
            "only move the camera",
            "move only camera",
            "move only the camera",
        ),
    )

    if camera_fixed:
        fixed_variables = ["camera_x", "camera_y"]
    if camera_only:
        allowed_variables = ["camera_x", "camera_y"]
    if camera_fixed and camera_only:
        conflict_reasons.append("camera variables are both fixed and the only allowed controls")

    contradictory_pairs = (
        ("left", "right"),
        ("wider", "narrower"),
        ("larger", "smaller"),
        ("up", "down"),
    )
    for first, second in contradictory_pairs:
        if first in text and second in text:
            conflict_reasons.append(f"contains both {first} and {second}")

    unsupported_terms = ("wavelength", "color", "laser power")
    for term in unsupported_terms:
        if term in text:
            unsupported_requests.append(term)

    valid = not conflict_reasons and not unsupported_requests
    return {
        "valid": valid,
        "fixed_variables": fixed_variables,
        "allowed_variables": allowed_variables,
        "conflict_detected": bool(conflict_reasons),
        "conflict_reasons": conflict_reasons,
        "unsupported_requests": unsupported_requests,
    }


def _profile_stats_from_path(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    return compute_profile_stats(load_intensity(path))


def _target_delta_from_record(record: dict) -> dict | None:
    target_delta = record.get("target_delta")
    if isinstance(target_delta, dict):
        return target_delta
    current_setup = record.get("current_setup")
    target_setup = record.get("target_setup")
    if isinstance(current_setup, dict) and isinstance(target_setup, dict):
        return compute_delta_setup(current_setup, target_setup)
    return None


def _change_mask_prior(setup_change: dict[str, Any], fixed_variables: list[str]) -> dict[str, str]:
    fixed = set(fixed_variables)
    changed = set(setup_change["changed_variables"])
    priors: dict[str, str] = {}
    for variable in CANONICAL_VARIABLE_ORDER:
        if variable in fixed:
            priors[variable] = "fixed"
        elif variable in changed:
            priors[variable] = "required"
        else:
            priors[variable] = "unlikely"
    return priors


def build_reasoning_command(record: dict, rendered_image_paths: dict | None = None) -> dict:
    """Build and validate the structured assistant JSON for one SFT example."""
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")

    prompt = str(record.get("prompt") or "").strip()
    constraints = infer_prompt_constraints(prompt)
    current_stats = _profile_stats_from_path(record.get("current_profile_path"))
    target_stats = _profile_stats_from_path(record.get("target_profile_path"))
    observed_change = classify_profile_change(current_stats, target_stats)
    setup_change = classify_setup_delta(_target_delta_from_record(record))

    command = default_reasoning_command()
    command["valid"] = bool(constraints["valid"])
    command["task_type"] = record.get("task_type")
    command["observed_profile_change"] = observed_change
    command["requested_goal"] = prompt
    command["constraints"] = {
        "fixed_variables": constraints["fixed_variables"],
        "allowed_variables": constraints["allowed_variables"],
        "change_mask_prior": _change_mask_prior(setup_change, constraints["fixed_variables"]),
    }
    command["control_plan"] = {
        "likely_relevant_variables": [
            variable
            for variable in setup_change["changed_variables"]
            if variable not in set(constraints["fixed_variables"])
        ],
        "avoid_variables": constraints["fixed_variables"],
        "setup_delta_summary": setup_change,
    }
    if rendered_image_paths:
        command["control_plan"]["rendered_images"] = dict(rendered_image_paths)
    command["canonical_prompt"] = prompt
    command["conflict_detected"] = bool(constraints["conflict_detected"])
    command["unsupported_requests"] = list(constraints["unsupported_requests"])
    command["reasoning_summary"] = (
        f"Task {command['task_type']} uses rendered intensity profiles and canonical setup variables only."
    )
    if not command["valid"]:
        reasons = list(constraints["conflict_reasons"]) + list(constraints["unsupported_requests"])
        command["rejection_reason"] = "; ".join(reasons) or "prompt is not supported"
    else:
        command["rejection_reason"] = ""

    return validate_reasoning_command(command)


def _setup_text(record: dict) -> str:
    parts = [
        f"task_type: {record.get('task_type')}",
        f"prompt: {record.get('prompt')}",
        "canonical_variables: " + ", ".join(VARIABLE_ORDER),
    ]
    if isinstance(record.get("current_setup"), dict):
        parts.append("current_setup: " + json.dumps(record["current_setup"], sort_keys=True))
    if isinstance(record.get("target_setup"), dict):
        parts.append("target_setup: " + json.dumps(record["target_setup"], sort_keys=True))
    if isinstance(record.get("target_delta"), dict):
        parts.append("target_delta: " + json.dumps(record["target_delta"], sort_keys=True))
    return "\n".join(parts)


def build_sft_record(record: dict, image_paths: dict) -> dict[str, list[dict[str, Any]]]:
    """Build a multimodal SFT-style message record."""
    command = build_reasoning_command(record, rendered_image_paths=image_paths)
    user_content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": (
                "Infer the profile2setup reasoning JSON from the prompt, canonical variables, "
                "available setup fields, and rendered intensity-profile images.\n"
                + _setup_text(record)
            ),
        }
    ]
    for key in ("current_profile", "target_profile", "difference_profile", "composite_profile"):
        if key in image_paths:
            user_content.append({"type": "image", "image_path": image_paths[key]})

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a profile2setup reasoning data generator. "
                    "Output only valid JSON that follows the provided schema and uses canonical variables."
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(command, sort_keys=True)},
        ]
    }
