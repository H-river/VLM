"""Adapter layer between optics_sft and optical_sim."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping

from optical_sim.src.metrics import BeamMetrics, compute_metrics
from optical_sim.src.optical_elements import OpticalSetup
from optical_sim.src.simulator import run_simulation


MM_TO_M = 1e-3
M_TO_MM = 1e3
M_TO_NM = 1e9
M_TO_UM = 1e6


@dataclass(frozen=True)
class Action:
    """Actuator deltas in millimeters."""

    lens_x_delta_mm: float
    lens_y_delta_mm: float
    camera_x_delta_mm: float
    camera_y_delta_mm: float


def _action_value(action: Action | Mapping[str, Any], key: str) -> float:
    if isinstance(action, Action):
        return float(getattr(action, key))
    return float(action[key])


def apply_action_to_setup(
    setup: OpticalSetup,
    action: Action | Mapping[str, Any],
) -> OpticalSetup:
    """Return a copied setup with actuator deltas applied."""
    copied = copy.deepcopy(setup)
    copied.lens.x_offset += _action_value(action, "lens_x_delta_mm") * MM_TO_M
    copied.lens.y_offset += _action_value(action, "lens_y_delta_mm") * MM_TO_M
    copied.camera.x_offset += _action_value(action, "camera_x_delta_mm") * MM_TO_M
    copied.camera.y_offset += _action_value(action, "camera_y_delta_mm") * MM_TO_M
    return copied


def metrics_to_state_m(metrics: BeamMetrics | Mapping[str, Any]) -> dict[str, float]:
    """Convert simulator metrics to the meter-based SFT state shape."""
    if isinstance(metrics, BeamMetrics):
        values = metrics.to_dict()
    else:
        values = dict(metrics)
    return {
        "centroid_x_m": float(values["centroid_x"]),
        "centroid_y_m": float(values["centroid_y"]),
        "sigma_x_m": float(values["sigma_x"]),
        "sigma_y_m": float(values["sigma_y"]),
        "peak_intensity": float(values["peak_intensity"]),
    }


def state_m_to_state_px(state_m: Mapping[str, Any], setup: OpticalSetup) -> dict[str, float]:
    """Convert lab-frame meter fields to the legacy pseudo-pixel coordinates.

    This preserves the published v1-v9 dataset convention.  It is not an
    image-array coordinate when camera offsets are nonzero; new visual data
    should use :func:`state_m_to_sensor_frame_px`.
    """
    height_px, width_px = setup.sensor.resolution
    pitch = float(setup.sensor.pixel_pitch)
    return {
        "centroid_x_px": float(state_m["centroid_x_m"]) / pitch + (width_px - 1) / 2.0,
        "centroid_y_px": float(state_m["centroid_y_m"]) / pitch + (height_px - 1) / 2.0,
        "sigma_x_px": float(state_m["sigma_x_m"]) / pitch,
        "sigma_y_px": float(state_m["sigma_y_m"]) / pitch,
    }


def state_m_to_sensor_frame_px(
    state_m: Mapping[str, Any], setup: OpticalSetup
) -> dict[str, float]:
    """Convert lab-frame beam metrics to camera sensor-array coordinates."""

    height_px, width_px = setup.sensor.resolution
    pitch = float(setup.sensor.pixel_pitch)
    return {
        "centroid_x_px": (
            float(state_m["centroid_x_m"]) - float(setup.camera.x_offset)
        )
        / pitch
        + (width_px - 1) / 2.0,
        "centroid_y_px": (
            float(state_m["centroid_y_m"]) - float(setup.camera.y_offset)
        )
        / pitch
        + (height_px - 1) / 2.0,
        "sigma_x_px": float(state_m["sigma_x_m"]) / pitch,
        "sigma_y_px": float(state_m["sigma_y_m"]) / pitch,
    }


def metrics_to_sensor_frame_state(
    metrics: BeamMetrics | Mapping[str, Any], setup: OpticalSetup
) -> dict[str, float]:
    """Return meter fields plus centroid/width fields aligned with PNG pixels."""

    state_m = metrics_to_state_m(metrics)
    return {**state_m, **state_m_to_sensor_frame_px(state_m, setup)}


def metrics_to_state(metrics: BeamMetrics | Mapping[str, Any], setup: OpticalSetup) -> dict[str, float]:
    """Return combined meter and pixel state fields for a simulation result."""
    state_m = metrics_to_state_m(metrics)
    return {
        **state_m,
        **state_m_to_state_px(state_m, setup),
    }


def simulate_and_measure(setup: OpticalSetup) -> dict[str, Any]:
    """Run optical_sim and return arrays, metrics, and compact state fields."""
    result = run_simulation(setup)
    metrics = compute_metrics(
        result["intensity"],
        result["sensor_X"],
        result["sensor_Y"],
    )
    metrics_dict = metrics.to_dict()
    return {
        "intensity": result["intensity"],
        "sensor_X": result["sensor_X"],
        "sensor_Y": result["sensor_Y"],
        "metrics": metrics_dict,
        "state": metrics_to_state(metrics, setup),
    }


def residual_error_px(
    current_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> float:
    """Return Euclidean centroid distance between two pixel states."""
    dx = float(current_state["centroid_x_px"]) - float(target_state["centroid_x_px"])
    dy = float(current_state["centroid_y_px"]) - float(target_state["centroid_y_px"])
    return math.hypot(dx, dy)


def setup_to_safe_metadata(setup: OpticalSetup) -> dict[str, Any]:
    """Convert an optical setup to prompt-safe metadata only."""
    return {
        "wavelength_nm": setup.source.wavelength * M_TO_NM,
        "beam_waist_mm": setup.source.beam_waist * M_TO_MM,
        "power_w": setup.source.power,
        "lens_focal_length_mm": setup.lens.focal_length * M_TO_MM,
        "lens_aperture_mm": setup.lens.clear_aperture * M_TO_MM,
        "source_to_lens_mm": setup.laser_to_lens * M_TO_MM,
        "lens_to_camera_mm": setup.lens_to_camera * M_TO_MM,
        "sensor_resolution": list(setup.sensor.resolution),
        "pixel_size_um": setup.sensor.pixel_pitch * M_TO_UM,
        "propagation_backend": setup.propagation_backend,
        "grid_size": setup.grid_size,
        "grid_extent_mm": setup.grid_extent * M_TO_MM,
    }
