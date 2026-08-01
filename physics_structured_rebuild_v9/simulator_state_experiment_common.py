"""Shared physics helpers for simulator state-sufficiency experiments."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from control_rebuild_v5.simulator_grid import simulate_fixed_action_grid
from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.metrics import compute_metrics
from optical_sim.src.optical_elements import OpticalSetup, setup_from_dict
from optical_sim.src.simulator import (
    _BACKENDS,
    _extract_sensor_region,
    _make_grid,
    apply_thin_lens,
)
from optics_sft.physics.sim_adapter import metrics_to_state
from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    config_from_visible,
)
from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = REPO_ROOT / "optical_sim/configs/base_config.yaml"
REGIMES = ("ordinary", "focusing", "clipping", "camera_boundary")


def stable_seed(*parts: object) -> int:
    text = ":".join(map(str, parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def sample_visible_setup(regime: str, index: int, seed: int) -> dict[str, float]:
    """Return a deterministic setup stratified by physical regime."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime}")
    rng = np.random.default_rng(stable_seed(seed, regime, index))
    result = {
        "wavelength_nm": float(rng.uniform(617.0, 648.0)),
        "beam_waist_mm": float(rng.uniform(0.70, 1.35)),
        "power_w": float(rng.uniform(0.75, 1.25)),
        "lens_focal_length_mm": float(rng.uniform(82.0, 120.0)),
        "lens_aperture_mm": float(rng.uniform(19.0, 32.0)),
        "source_to_lens_mm": float(rng.uniform(140.0, 260.0)),
        "lens_to_camera_mm": float(rng.uniform(92.0, 190.0)),
        "lens_x_offset_mm": float(rng.uniform(-0.23, 0.23)),
        "lens_y_offset_mm": float(rng.uniform(-0.23, 0.23)),
        "camera_x_offset_mm": float(rng.uniform(-0.18, 0.18)),
        "camera_y_offset_mm": float(rng.uniform(-0.18, 0.18)),
        "pixel_size_um": float(rng.choice((5.0, 5.5, 6.0))),
    }
    if regime == "focusing":
        focal = float(rng.uniform(70.0, 130.0))
        source = float(rng.uniform(max(1.18 * focal, 120.0), 285.0))
        image_distance = focal * source / (source - focal)
        result.update(
            {
                "beam_waist_mm": float(rng.uniform(0.65, 1.25)),
                "lens_focal_length_mm": focal,
                "source_to_lens_mm": source,
                "lens_to_camera_mm": float(
                    np.clip(
                        image_distance * rng.uniform(0.985, 1.015),
                        72.0,
                        260.0,
                    )
                ),
                "lens_aperture_mm": float(rng.uniform(18.0, 30.0)),
            }
        )
    elif regime == "clipping":
        result.update(
            {
                "beam_waist_mm": float(rng.uniform(0.75, 1.35)),
                "lens_aperture_mm": float(rng.uniform(1.2, 3.6)),
                "lens_x_offset_mm": float(
                    rng.choice((-1.0, 1.0)) * rng.uniform(0.20, 1.00)
                ),
                "lens_y_offset_mm": float(
                    rng.choice((-1.0, 1.0)) * rng.uniform(0.20, 1.00)
                ),
            }
        )
    elif regime == "camera_boundary":
        pitch_mm = result["pixel_size_um"] / 1000.0
        half_sensor_mm = 512.0 * pitch_mm
        axis = int(rng.integers(0, 2))
        sign = float(rng.choice((-1.0, 1.0)))
        boundary_offset = sign * rng.uniform(
            0.68 * half_sensor_mm,
            1.03 * half_sensor_mm,
        )
        if axis == 0:
            result["camera_x_offset_mm"] = float(boundary_offset)
            result["camera_y_offset_mm"] = float(
                rng.uniform(-0.25, 0.25) * half_sensor_mm
            )
        else:
            result["camera_y_offset_mm"] = float(boundary_offset)
            result["camera_x_offset_mm"] = float(
                rng.uniform(-0.25, 0.25) * half_sensor_mm
            )
    return result


def setup_from_visible(visible: Mapping[str, Any]) -> OpticalSetup:
    base = load_yaml(BASE_CONFIG_PATH)
    values = {**dict(visible), "sensor_resolution_px": [1024, 1024]}
    return setup_from_dict(config_from_visible(values, base))


def state_array(state: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(state[field]) for field in STATE_FIELDS],
        dtype=np.float64,
    )


def action_grid_states(setup: OpticalSetup) -> np.ndarray:
    simulated = simulate_fixed_action_grid(setup)
    return np.asarray(
        [
            [float(item["state"][field]) for field in STATE_FIELDS]
            for item in simulated
        ],
        dtype=np.float64,
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    draws: int = 2000,
) -> dict[str, float | int | None]:
    """Bootstrap groups, with one input row per independent context."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        size = min(200, draws - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        means[start : start + size] = array[indices].mean(axis=1)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def lens_transmission(setup: OpticalSetup) -> float:
    """Fraction of incident lens-plane power inside the clear aperture."""

    propagate = _BACKENDS.get(
        setup.propagation_backend,
        _BACKENDS["fresnel_numpy"],
    )
    field, x_grid, y_grid, spacing = custom_source_field(setup, {})
    at_lens = propagate(
        field,
        spacing,
        setup.laser_to_lens,
        setup.source.wavelength,
    )
    radius_squared = (
        np.square(x_grid - setup.lens.x_offset)
        + np.square(y_grid - setup.lens.y_offset)
    )
    mask = radius_squared <= np.square(setup.lens.clear_aperture / 2.0)
    intensity = np.square(np.abs(at_lens))
    return float(intensity[mask].sum() / max(float(intensity.sum()), 1e-30))


def geometric_focus_residual(setup: OpticalSetup) -> float:
    source = float(setup.laser_to_lens)
    focal = float(setup.lens.focal_length)
    camera = float(setup.effective_camera_distance)
    inverse_focal = 1.0 / max(abs(focal), 1e-12)
    return float(
        abs(inverse_focal - 1.0 / source - 1.0 / camera) / inverse_focal
    )


def custom_source_field(
    setup: OpticalSetup,
    hidden: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create a Gaussian source with omitted amplitude, shape, and phase state."""

    x_grid, y_grid, spacing = _make_grid(setup.grid_size, setup.grid_extent)
    waist = float(setup.source.beam_waist)
    x_offset = float(hidden.get("source_x_offset_mm", 0.0)) * 1e-3
    y_offset = float(hidden.get("source_y_offset_mm", 0.0)) * 1e-3
    waist_x = waist * float(hidden.get("waist_x_scale", 1.0))
    waist_y = waist * float(hidden.get("waist_y_scale", 1.0))
    amplitude = float(hidden.get("amplitude_scale", 1.0))
    shifted_x = x_grid - x_offset
    shifted_y = y_grid - y_offset
    envelope = amplitude * np.exp(
        -np.square(shifted_x) / np.square(waist_x)
        - np.square(shifted_y) / np.square(waist_y)
    )
    phase_at_waist = float(hidden.get("curvature_phase_rad", 0.0))
    astigmatism = float(hidden.get("astigmatic_phase_rad", 0.0))
    tilt_x = float(hidden.get("source_tilt_x_mrad", 0.0)) * 1e-3
    tilt_y = float(hidden.get("source_tilt_y_mrad", 0.0)) * 1e-3
    radial_phase = phase_at_waist * (
        np.square(shifted_x) + np.square(shifted_y)
    ) / max(waist * waist, 1e-30)
    astigmatic_phase = astigmatism * (
        np.square(shifted_x) - np.square(shifted_y)
    ) / max(waist * waist, 1e-30)
    linear_phase = setup.source.wavenumber * (
        tilt_x * shifted_x + tilt_y * shifted_y
    )
    field = envelope * np.exp(
        1j * (radial_phase + astigmatic_phase + linear_phase)
    )
    return field.astype(np.complex128), x_grid, y_grid, spacing


def _complex_sensor_region(
    field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    setup: OpticalSetup,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = setup.sensor.resolution
    pitch = setup.sensor.pixel_pitch
    sensor_x = np.linspace(
        setup.camera.x_offset - width / 2 * pitch,
        setup.camera.x_offset + width / 2 * pitch,
        width,
    )
    sensor_y = np.linspace(
        setup.camera.y_offset - height / 2 * pitch,
        setup.camera.y_offset + height / 2 * pitch,
        height,
    )
    ix = np.searchsorted(x_grid[0, :], sensor_x).clip(0, x_grid.shape[1] - 1)
    iy = np.searchsorted(y_grid[:, 0], sensor_y).clip(0, y_grid.shape[0] - 1)
    sampled = field[np.ix_(iy, ix)]
    sx, sy = np.meshgrid(sensor_x, sensor_y)
    return sampled, sx, sy


def custom_hidden_action_grid(
    setup: OpticalSetup,
    hidden: Mapping[str, float],
    return_current_field: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """Simulate all actions for an omitted complex source state."""

    propagate = _BACKENDS.get(
        setup.propagation_backend,
        _BACKENDS["fresnel_numpy"],
    )
    source, x_grid, y_grid, spacing = custom_source_field(setup, hidden)
    at_lens = propagate(
        source,
        spacing,
        setup.laser_to_lens,
        setup.source.wavelength,
    )
    states = []
    current_arrays: dict[str, np.ndarray] | None = None
    cached_fields: dict[tuple[float, float], np.ndarray] = {}
    for action_index, action in enumerate(ACTION_GRID):
        lens_key = (
            float(action["lens_x_delta_mm"]),
            float(action["lens_y_delta_mm"]),
        )
        field_at_camera = cached_fields.get(lens_key)
        if field_at_camera is None:
            lens_setup = copy.deepcopy(setup)
            lens_setup.lens.x_offset += lens_key[0] * 1e-3
            lens_setup.lens.y_offset += lens_key[1] * 1e-3
            after_lens = apply_thin_lens(
                at_lens,
                x_grid,
                y_grid,
                lens_setup,
            )
            field_at_camera = propagate(
                after_lens,
                spacing,
                lens_setup.effective_camera_distance,
                setup.source.wavelength,
            )
            cached_fields[lens_key] = field_at_camera
        candidate = copy.deepcopy(setup)
        candidate.lens.x_offset += lens_key[0] * 1e-3
        candidate.lens.y_offset += lens_key[1] * 1e-3
        candidate.camera.x_offset += (
            float(action["camera_x_delta_mm"]) * 1e-3
        )
        candidate.camera.y_offset += (
            float(action["camera_y_delta_mm"]) * 1e-3
        )
        intensity, sensor_x, sensor_y = _extract_sensor_region(
            field_at_camera,
            x_grid,
            y_grid,
            candidate,
        )
        metrics = compute_metrics(intensity, sensor_x, sensor_y)
        combined = metrics_to_state(metrics, candidate)
        states.append([float(combined[field]) for field in STATE_FIELDS])
        if return_current_field and action_index == 40:
            complex_sensor, _, _ = _complex_sensor_region(
                field_at_camera,
                x_grid,
                y_grid,
                candidate,
            )
            current_arrays = {
                "field": complex_sensor,
                "intensity": np.square(np.abs(complex_sensor)),
                "sensor_x": sensor_x,
                "sensor_y": sensor_y,
            }
    return np.asarray(states, dtype=np.float64), current_arrays


def custom_hidden_current_state(
    setup: OpticalSetup,
    hidden: Mapping[str, float],
) -> np.ndarray:
    """Simulate only the zero-action state for optimization."""

    propagate = _BACKENDS.get(
        setup.propagation_backend,
        _BACKENDS["fresnel_numpy"],
    )
    source, x_grid, y_grid, spacing = custom_source_field(setup, hidden)
    at_lens = propagate(
        source,
        spacing,
        setup.laser_to_lens,
        setup.source.wavelength,
    )
    after_lens = apply_thin_lens(at_lens, x_grid, y_grid, setup)
    field_at_camera = propagate(
        after_lens,
        spacing,
        setup.effective_camera_distance,
        setup.source.wavelength,
    )
    intensity, sensor_x, sensor_y = _extract_sensor_region(
        field_at_camera,
        x_grid,
        y_grid,
        setup,
    )
    metrics = compute_metrics(intensity, sensor_x, sensor_y)
    combined = metrics_to_state(metrics, setup)
    return state_array(combined)


def phase_descriptors(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return weighted phase tilt and curvature descriptors.

    Local phase increments avoid global phase and two-dimensional unwrapping.
    The five outputs are x/y tilt, x/y curvature, and cross-curvature.
    """

    field = np.asarray(arrays["field"], dtype=np.complex128)
    intensity = np.asarray(arrays["intensity"], dtype=np.float64)
    sensor_x = np.asarray(arrays["sensor_x"], dtype=np.float64)
    sensor_y = np.asarray(arrays["sensor_y"], dtype=np.float64)
    dx = float(np.median(np.diff(sensor_x[0])))
    dy = float(np.median(np.diff(sensor_y[:, 0])))
    phase_x = np.angle(field[:, 1:] * np.conj(field[:, :-1])) / dx
    phase_y = np.angle(field[1:, :] * np.conj(field[:-1, :])) / dy
    weight_x = np.sqrt(intensity[:, 1:] * intensity[:, :-1])
    weight_y = np.sqrt(intensity[1:, :] * intensity[:-1, :])
    x_mid = 0.5 * (sensor_x[:, 1:] + sensor_x[:, :-1])
    y_mid_x = 0.5 * (sensor_y[:, 1:] + sensor_y[:, :-1])
    x_mid_y = 0.5 * (sensor_x[1:, :] + sensor_x[:-1, :])
    y_mid = 0.5 * (sensor_y[1:, :] + sensor_y[:-1, :])

    def fit(
        value: np.ndarray,
        x_value: np.ndarray,
        y_value: np.ndarray,
        weight: np.ndarray,
    ) -> np.ndarray:
        mask = weight > max(float(weight.max()) * 1e-6, 1e-30)
        design = np.column_stack(
            [
                np.ones(int(mask.sum())),
                x_value[mask],
                y_value[mask],
            ]
        )
        root = np.sqrt(weight[mask] / max(float(weight[mask].max()), 1e-30))
        coefficient, *_ = np.linalg.lstsq(
            design * root[:, None],
            value[mask] * root,
            rcond=None,
        )
        return coefficient

    fit_x = fit(phase_x, x_mid, y_mid_x, weight_x)
    fit_y = fit(phase_y, x_mid_y, y_mid, weight_y)
    return np.asarray(
        [
            fit_x[0],
            fit_y[0],
            fit_x[1],
            fit_y[2],
            0.5 * (fit_x[2] + fit_y[1]),
        ],
        dtype=np.float64,
    )


def image_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float | bool]:
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    scale = max(float(left_value.max()), float(right_value.max()), 1e-30)
    difference = left_value - right_value
    rmse = float(np.sqrt(np.mean(np.square(difference))) / scale)
    maximum = float(np.max(np.abs(difference)) / scale)
    left_flat = left_value.ravel()
    right_flat = right_value.ravel()
    correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    return {
        "normalized_rmse": rmse,
        "normalized_maximum": maximum,
        "correlation": correlation,
        "matches_1pct_rmse": bool(rmse <= 0.01),
        "matches_5pct_maximum": bool(maximum <= 0.05),
    }


def matching_precision(current: np.ndarray) -> np.ndarray:
    """One tenth of the production tolerance for matching current moments."""

    return 0.1 * tolerance_from_current(current).astype(np.float64)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
