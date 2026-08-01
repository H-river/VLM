"""Deterministic tools for setup, causal, and diagnosis reasoning routes."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from optics_sft.physics.sim_adapter import simulate_and_measure

from .core import apply_action_dict, classify_effects, setup_from_dict
from .tool_path_adapter import resolve_path


SETUP_TOOL = "normalize_optical_setup_summary_v1"
CAUSAL_TOOL = "classify_registered_transition_v1"
DIAGNOSIS_TOOL = "replay_registered_diagnosis_candidates_v1"

TOOL_SPECS: dict[str, dict[str, Any]] = {
    SETUP_TOOL: {
        "operation": "normalize the visible apparatus metadata, convert units, and list adjustable actuators",
        "required_source_roles": ["setup_path", "actuator_interface_path"],
    },
    CAUSAL_TOOL: {
        "operation": "classify registered before/after beam changes using the declared physical deadbands",
        "required_source_roles": [
            "before_state_handle_path",
            "after_state_handle_path",
            "thresholds_path",
        ],
    },
    DIAGNOSIS_TOOL: {
        "operation": "replay every candidate intervention and return all candidates matching the registered observation",
        "required_source_roles": [
            "setup_state_handle_path",
            "observed_state_handle_path",
            "candidate_interventions_path",
            "matching_tolerance_path",
        ],
    },
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def normalize_setup_summary(
    setup: Mapping[str, Any], actuator_interface: Mapping[str, Any]
) -> dict[str, Any]:
    adjustable = actuator_interface.get("adjustable_parameters")
    if not isinstance(adjustable, list) or not all(isinstance(v, str) for v in adjustable):
        raise ValueError("adjustable_parameters must be a list of strings")
    source_to_lens = _finite(setup.get("source_to_lens_mm"), "source_to_lens_mm")
    lens_to_camera = _finite(setup.get("lens_to_camera_mm"), "lens_to_camera_mm")
    focal_mm = _finite(setup.get("lens_focal_length_mm"), "lens_focal_length_mm")
    return {
        "component_order": ["gaussian_source", "thin_lens", "camera_sensor"],
        "adjustable_parameters": copy.deepcopy(adjustable),
        "total_source_to_sensor_mm": round(source_to_lens + lens_to_camera, 4),
        "lens_focal_length_m": round(focal_mm / 1000.0, 6),
    }


def classify_registered_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    label_cfg = {
        "centroid_effect_threshold_px": _finite(
            thresholds.get("centroid_px"), "centroid_px"
        ),
        "sigma_effect_threshold_px": _finite(thresholds.get("sigma_px"), "sigma_px"),
        "intensity_relative_threshold": _finite(
            thresholds.get("peak_relative"), "peak_relative"
        ),
    }
    if any(value < 0.0 for value in label_cfg.values()):
        raise ValueError("causal thresholds must be nonnegative")
    return {"effects": classify_effects(before, after, label_cfg)}


def _candidate_matches(
    candidate: Mapping[str, Any], observed: Mapping[str, Any], tolerance: Mapping[str, Any]
) -> bool:
    centroid_tolerance = _finite(tolerance.get("centroid_px"), "centroid_px")
    sigma_tolerance = _finite(tolerance.get("sigma_px"), "sigma_px")
    dx = _finite(candidate.get("centroid_x_px"), "candidate centroid_x_px") - _finite(
        observed.get("centroid_x_px"), "observed centroid_x_px"
    )
    dy = _finite(candidate.get("centroid_y_px"), "candidate centroid_y_px") - _finite(
        observed.get("centroid_y_px"), "observed centroid_y_px"
    )
    return (
        math.hypot(dx, dy) <= centroid_tolerance
        and abs(
            _finite(candidate.get("sigma_x_px"), "candidate sigma_x_px")
            - _finite(observed.get("sigma_x_px"), "observed sigma_x_px")
        )
        <= sigma_tolerance
        and abs(
            _finite(candidate.get("sigma_y_px"), "candidate sigma_y_px")
            - _finite(observed.get("sigma_y_px"), "observed sigma_y_px")
        )
        <= sigma_tolerance
    )


def replay_diagnosis_candidates(
    setup_config: Mapping[str, Any],
    observed_state: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    matching_tolerance: Mapping[str, Any],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("diagnosis candidates must be nonempty")
    setup = setup_from_dict(copy.deepcopy(dict(setup_config)))
    plausible: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_id = candidate.get("candidate_id")
        action = candidate.get("action")
        if not isinstance(candidate_id, str) or not candidate_id or not isinstance(action, Mapping):
            raise ValueError(f"candidate {index} has an invalid id or action")
        state = simulate_and_measure(apply_action_dict(setup, action))["state"]
        if _candidate_matches(state, observed_state, matching_tolerance):
            plausible.append(candidate_id)
    status = "unsupported" if not plausible else "unique" if len(plausible) == 1 else "ambiguous"
    return {"status": status, "plausible_causes": plausible}


def _mapped_values(
    tool_name: str,
    visible_evidence: Mapping[str, Any],
    source_map: Mapping[str, Any],
) -> dict[str, Any]:
    required = set(TOOL_SPECS[tool_name]["required_source_roles"])
    if set(source_map) != required or not all(isinstance(v, str) for v in source_map.values()):
        raise ValueError(f"{tool_name} source_map must contain exactly {sorted(required)}")
    return {role: resolve_path(visible_evidence, str(path)) for role, path in source_map.items()}


def run_mapped_direct_tool(
    tool_name: str,
    visible_evidence: Mapping[str, Any],
    source_map: Mapping[str, Any],
    *,
    setup_registry: Mapping[str, Mapping[str, Any]],
    observation_registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tool_name not in TOOL_SPECS:
        raise ValueError(f"unsupported direct reasoning tool: {tool_name}")
    values = _mapped_values(tool_name, visible_evidence, source_map)
    if tool_name == SETUP_TOOL:
        setup = values["setup_path"]
        interface = values["actuator_interface_path"]
        if not isinstance(setup, Mapping) or not isinstance(interface, Mapping):
            raise ValueError("setup and actuator interface must be objects")
        return normalize_setup_summary(setup, interface)
    if tool_name == CAUSAL_TOOL:
        before_handle = values["before_state_handle_path"]
        after_handle = values["after_state_handle_path"]
        thresholds = values["thresholds_path"]
        if before_handle not in observation_registry or after_handle not in observation_registry:
            raise ValueError("unknown registered causal observation handle")
        if not isinstance(thresholds, Mapping):
            raise ValueError("causal thresholds must be an object")
        return classify_registered_transition(
            observation_registry[str(before_handle)],
            observation_registry[str(after_handle)],
            thresholds,
        )
    setup_handle = values["setup_state_handle_path"]
    observed_handle = values["observed_state_handle_path"]
    candidates = values["candidate_interventions_path"]
    tolerance = values["matching_tolerance_path"]
    if setup_handle not in setup_registry or observed_handle not in observation_registry:
        raise ValueError("unknown registered diagnosis handle")
    if not isinstance(candidates, list) or not isinstance(tolerance, Mapping):
        raise ValueError("diagnosis candidates must be a list and tolerance must be an object")
    return replay_diagnosis_candidates(
        setup_registry[str(setup_handle)],
        observation_registry[str(observed_handle)],
        candidates,
        tolerance,
    )


def materialize_direct_answer(tool_name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    if tool_name == SETUP_TOOL:
        return {"status": "answerable", "answer": copy.deepcopy(dict(result))}
    if tool_name == CAUSAL_TOOL:
        return {
            "status": "answerable",
            "answer": {"effects": copy.deepcopy(result["effects"])},
        }
    if tool_name == DIAGNOSIS_TOOL:
        return {
            "status": result["status"],
            "answer": {"plausible_causes": copy.deepcopy(result["plausible_causes"])},
        }
    raise ValueError(f"unsupported direct reasoning tool: {tool_name}")
