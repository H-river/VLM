"""Isolated helpers for auditing the optical simulator without changing it.

The functions in this module deliberately duplicate small pieces of sampling
and measurement logic.  They are comparison implementations, not production
alternatives.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from continuous_control_v12.contracts import (
    OUTPUT_FIELDS,
    Bounds,
    metrics_vector,
    tolerance_vector,
)
from continuous_control_v12.simulator import build_optical_setup
from optical_sim.src.metrics import BeamMetrics, compute_metrics
from optical_sim.src.optical_elements import OpticalSetup
from optical_sim.src.simulator import (
    _BACKENDS,
    _extract_sensor_region,
    apply_thin_lens,
    gaussian_source_field,
)
from optics_sft.physics.sim_adapter import (
    metrics_to_sensor_frame_state,
    metrics_to_state,
)


BASE_CONTEXT: dict[str, float] = {
    "wavelength_nm": 632.8,
    "beam_waist_mm": 1.0,
    "power_w": 1.0,
    "lens_focal_length_mm": 100.0,
    "lens_aperture_mm": 25.0,
    "source_to_lens_mm": 200.0,
    "lens_to_camera_mm": 150.0,
    "pixel_size_um": 5.5,
}

BASE_POSITIONS_MM: dict[str, float] = {
    "lens_x_mm": 0.11,
    "lens_y_mm": -0.08,
    "camera_x_mm": 0.07,
    "camera_y_mm": -0.04,
}

POSITION_TO_AXIS = {
    "lens_x_mm": "x",
    "lens_y_mm": "y",
    "camera_x_mm": "x",
    "camera_y_mm": "y",
}


def guarded_output_dir(path: Path, audit_root: Path) -> Path:
    """Resolve and validate an audit output path."""

    resolved = path.resolve()
    forbidden = ("physics_structured_rebuild_v10", "locked", "frozen_v9")
    lowered = tuple(part.lower() for part in resolved.parts)
    if any(token in part for part in lowered for token in forbidden):
        raise ValueError(f"refusing protected-looking output path: {resolved}")
    if audit_root.resolve() not in (resolved, *resolved.parents):
        raise ValueError("audit output must remain inside the versioned audit directory")
    return resolved


def build_setup(
    context: Mapping[str, Any],
    positions_mm: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    base_config_path: str,
) -> OpticalSetup:
    return build_optical_setup(
        context,
        positions_mm,
        simulator_fixed,
        base_config_path,
    )


def prepare_source_and_lens(
    setup: OpticalSetup,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Return source/grid/spacing and the propagated field at the lens."""

    propagate = _BACKENDS.get(
        setup.propagation_backend,
        _BACKENDS["fresnel_numpy"],
    )
    source, grid_x, grid_y, spacing = gaussian_source_field(setup)
    at_lens = propagate(
        source,
        spacing,
        setup.laser_to_lens,
        setup.source.wavelength,
    )
    return source, grid_x, grid_y, spacing, at_lens


def propagate_from_lens(
    setup: OpticalSetup,
    at_lens: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the lens and propagate to the camera plane."""

    propagate = _BACKENDS.get(
        setup.propagation_backend,
        _BACKENDS["fresnel_numpy"],
    )
    after_lens = apply_thin_lens(at_lens, grid_x, grid_y, setup)
    field_at_camera = propagate(
        after_lens,
        spacing,
        setup.effective_camera_distance,
        setup.source.wavelength,
    )
    return after_lens, field_at_camera


def sensor_coordinates(
    setup: OpticalSetup,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the production sensor coordinate construction exactly."""

    height, width = setup.sensor.resolution
    pitch = float(setup.sensor.pixel_pitch)
    sx = np.linspace(
        setup.camera.x_offset - width / 2.0 * pitch,
        setup.camera.x_offset + width / 2.0 * pitch,
        width,
    )
    sy = np.linspace(
        setup.camera.y_offset - height / 2.0 * pitch,
        setup.camera.y_offset + height / 2.0 * pitch,
        height,
    )
    sensor_x, sensor_y = np.meshgrid(sx, sy)
    return sx, sy, sensor_x, sensor_y


def sampling_metadata(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    setup: OpticalSetup,
    axis: str,
    method: str,
) -> dict[str, Any]:
    """Expose the otherwise hidden physical-to-array mapping."""

    sx, sy, _, _ = sensor_coordinates(setup)
    grid_x_1d = grid_x[0, :]
    grid_y_1d = grid_y[:, 0]
    ix = np.searchsorted(grid_x_1d, sx).clip(0, len(grid_x_1d) - 1)
    iy = np.searchsorted(grid_y_1d, sy).clip(0, len(grid_y_1d) - 1)
    x_fractional = (sx - grid_x_1d[0]) / (grid_x_1d[1] - grid_x_1d[0])
    y_fractional = (sy - grid_y_1d[0]) / (grid_y_1d[1] - grid_y_1d[0])
    selected = ix if axis == "x" else iy
    fractional = x_fractional if axis == "x" else y_fractional
    sensor_values = sx if axis == "x" else sy
    signature_values = selected if method == "production_searchsorted" else np.floor(
        fractional
    ).astype(np.int64)
    signature = hashlib.sha256(signature_values.tobytes()).hexdigest()[:16]
    center = len(selected) // 2
    return {
        "selected_center_index": int(selected[center]),
        "selected_index_min": int(selected.min()),
        "selected_index_max": int(selected.max()),
        "selected_index_unique_count": int(np.unique(selected).size),
        "selected_index_signature": signature,
        "center_sensor_coordinate_mm": float(sensor_values[center] * 1e3),
        "center_fractional_grid_index": float(fractional[center]),
    }


def sample_sensor(
    field_at_camera: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    setup: OpticalSetup,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the camera field using production or audit-only methods."""

    if method == "production_searchsorted":
        return _extract_sensor_region(field_at_camera, grid_x, grid_y, setup)

    sx, sy, sensor_x, sensor_y = sensor_coordinates(setup)
    x0 = float(grid_x[0, 0])
    y0 = float(grid_y[0, 0])
    dx = float(grid_x[0, 1] - grid_x[0, 0])
    dy = float(grid_y[1, 0] - grid_y[0, 0])
    x_index = (sx - x0) / dx
    y_index = (sy - y0) / dy
    yy, xx = np.meshgrid(y_index, x_index, indexing="ij")
    coordinates = np.stack([yy, xx], axis=0)

    if method == "bilinear_intensity":
        full_intensity = np.square(np.abs(field_at_camera))
        intensity = map_coordinates(
            full_intensity,
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        )
    elif method == "bilinear_complex_field":
        real = map_coordinates(
            field_at_camera.real,
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        imag = map_coordinates(
            field_at_camera.imag,
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        intensity = np.square(np.abs(real + 1j * imag))
    else:
        raise ValueError(f"unknown audit sampling method: {method}")
    return intensity, sensor_x, sensor_y


def independent_sensor_metrics(
    intensity: np.ndarray,
    pixel_pitch_m: float,
) -> dict[str, float]:
    """Compute physical metrics from array pixels at documented pixel centres."""

    values = np.asarray(intensity, dtype=np.float64)
    height, width = values.shape
    x = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * pixel_pitch_m
    y = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * pixel_pitch_m
    sensor_x, sensor_y = np.meshgrid(x, y)
    measured = compute_metrics(values, sensor_x, sensor_y)
    return {
        "total_intensity": float(values.sum()),
        "centroid_x_m": float(measured.centroid_x),
        "centroid_y_m": float(measured.centroid_y),
        "sigma_x_m": float(measured.sigma_x),
        "sigma_y_m": float(measured.sigma_y),
        "centroid_x_px": float(measured.centroid_x / pixel_pitch_m + (width - 1) / 2.0),
        "centroid_y_px": float(measured.centroid_y / pixel_pitch_m + (height - 1) / 2.0),
        "sigma_x_px": float(measured.sigma_x / pixel_pitch_m),
        "sigma_y_px": float(measured.sigma_y / pixel_pitch_m),
        "peak_intensity": float(measured.peak_intensity),
    }


def capture_from_field(
    field_at_camera: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    setup: OpticalSetup,
    method: str,
) -> dict[str, Any]:
    """Return production-compatible and independent measurements."""

    intensity, sensor_x, sensor_y = sample_sensor(
        field_at_camera,
        grid_x,
        grid_y,
        setup,
        method,
    )
    measured = compute_metrics(intensity, sensor_x, sensor_y)
    stored = metrics_to_state(measured, setup)
    sensor_frame = metrics_to_sensor_frame_state(measured, setup)
    independent = independent_sensor_metrics(
        intensity,
        float(setup.sensor.pixel_pitch),
    )
    return {
        "intensity": np.asarray(intensity, dtype=np.float64),
        "beam_metrics": measured,
        "stored_metrics": {
            field: float(stored[field]) for field in OUTPUT_FIELDS
        },
        "sensor_frame_metrics": {
            field: float(sensor_frame[field]) for field in OUTPUT_FIELDS
        },
        "image_metrics": independent,
        "captured_power": float(
            np.asarray(intensity, dtype=np.float64).sum()
            * float(setup.sensor.pixel_pitch) ** 2
        ),
    }


def normalized_image(intensity: np.ndarray) -> np.ndarray:
    values = np.asarray(intensity, dtype=np.float64)
    peak = max(float(values.max()), 1e-30)
    return values / peak


def metric_changes(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    step_mm: float,
) -> dict[str, float]:
    """Metric deltas, tolerance-normalized deltas, and derivatives."""

    output: dict[str, float] = {}
    if previous is None:
        for field in OUTPUT_FIELDS:
            output[f"delta_{field}"] = 0.0
            output[f"tolerance_normalized_delta_{field}"] = 0.0
            output[f"derivative_{field}_per_mm"] = 0.0
        output["max_tolerance_normalized_delta"] = 0.0
        return output
    now = metrics_vector(current)
    before = metrics_vector(previous)
    delta = now - before
    tolerances = tolerance_vector(previous)
    normalized = np.abs(delta) / tolerances
    for index, field in enumerate(OUTPUT_FIELDS):
        output[f"delta_{field}"] = float(delta[index])
        output[f"tolerance_normalized_delta_{field}"] = float(normalized[index])
        output[f"derivative_{field}_per_mm"] = float(delta[index] / step_mm)
    output["max_tolerance_normalized_delta"] = float(normalized.max())
    return output


def image_changes(
    intensity: np.ndarray,
    previous: np.ndarray | None,
) -> dict[str, Any]:
    """Raw and independently normalized adjacent image differences."""

    if previous is None:
        return {
            "exact_image_plateau": False,
            "raw_image_relative_l1_change": 0.0,
            "normalized_image_mean_abs_change": 0.0,
            "normalized_image_relative_l2_change": 0.0,
        }
    values = np.asarray(intensity, dtype=np.float64)
    before = np.asarray(previous, dtype=np.float64)
    now_normalized = normalized_image(values)
    before_normalized = normalized_image(before)
    denominator = max(float(np.abs(before).sum()), 1e-30)
    l2_denominator = max(float(np.linalg.norm(before_normalized)), 1e-30)
    return {
        "exact_image_plateau": bool(np.array_equal(values, before)),
        "raw_image_relative_l1_change": float(
            np.abs(values - before).sum() / denominator
        ),
        "normalized_image_mean_abs_change": float(
            np.abs(now_normalized - before_normalized).mean()
        ),
        "normalized_image_relative_l2_change": float(
            np.linalg.norm(now_normalized - before_normalized) / l2_denominator
        ),
    }


def one_sweep_row(
    *,
    sweep_id: str,
    position_field: str,
    requested_position_mm: float,
    effective_position_mm: float,
    reference_position_mm: float,
    step_mm: float,
    setup: OpticalSetup,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    capture: Mapping[str, Any],
    previous_capture: Mapping[str, Any] | None,
    previous_index_signature: str | None,
) -> dict[str, Any]:
    """Flatten one deterministic sweep capture for CSV output."""

    axis = POSITION_TO_AXIS[position_field]
    sampling = sampling_metadata(grid_x, grid_y, setup, axis, method)
    previous_metrics = (
        None if previous_capture is None else previous_capture["stored_metrics"]
    )
    previous_image = (
        None if previous_capture is None else previous_capture["intensity"]
    )
    stored = capture["stored_metrics"]
    sensor = capture["sensor_frame_metrics"]
    image = capture["image_metrics"]
    height, width = setup.sensor.resolution
    grid_pitch_mm = float(grid_x[0, 1] - grid_x[0, 0]) * 1e3
    sensor_axis_size = width if axis == "x" else height
    declared_pitch_mm = float(setup.sensor.pixel_pitch) * 1e3
    row: dict[str, Any] = {
        "sweep_id": sweep_id,
        "sampling_method": method,
        "position_field": position_field,
        "axis": axis,
        "requested_position_mm": float(requested_position_mm),
        "effective_position_mm": float(effective_position_mm),
        "displacement_from_reference_mm": float(
            effective_position_mm - reference_position_mm
        ),
        "step_mm": float(step_mm),
        "declared_sensor_pitch_mm": declared_pitch_mm,
        "constructed_sensor_coordinate_step_mm": float(
            declared_pitch_mm * sensor_axis_size / (sensor_axis_size - 1)
        ),
        "simulation_grid_pitch_mm": grid_pitch_mm,
        **sampling,
        "index_boundary_crossed": bool(
            previous_index_signature is not None
            and previous_index_signature != sampling["selected_index_signature"]
        ),
        "captured_power": float(capture["captured_power"]),
        "image_sum": float(np.asarray(capture["intensity"]).sum()),
        "image_max": float(np.asarray(capture["intensity"]).max()),
    }
    for field in OUTPUT_FIELDS:
        row[f"stored_{field}"] = float(stored[field])
        row[f"sensor_frame_{field}"] = float(sensor[field])
        row[f"image_derived_{field}"] = float(image[field])
    row.update(metric_changes(stored, previous_metrics, step_mm))
    row.update(image_changes(capture["intensity"], previous_image))
    row["exact_metric_plateau"] = bool(
        previous_metrics is not None
        and np.array_equal(metrics_vector(stored), metrics_vector(previous_metrics))
    )
    return row


def exact_array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def read_nonprotected_power_values(path: Path) -> np.ndarray:
    """Read only setup.power_w from an explicitly non-protected JSONL."""

    lowered = "/".join(part.lower() for part in path.resolve().parts)
    if "v10" in lowered or "locked" in lowered or "frozen" in lowered:
        raise ValueError(f"refusing protected-looking data path: {path}")
    values = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            setup = row.get("setup", row.get("setup_context", {}))
            if "power_w" in setup:
                values.append(float(setup["power_w"]))
    return np.asarray(values, dtype=np.float64)


def interpolation_spot_checks(
    field_at_camera: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    setup: OpticalSetup,
) -> list[dict[str, Any]]:
    """Compare the three interpolation semantics at the same state."""

    rows = []
    for method in (
        "production_searchsorted",
        "bilinear_intensity",
        "bilinear_complex_field",
    ):
        capture = capture_from_field(
            field_at_camera,
            grid_x,
            grid_y,
            setup,
            method,
        )
        rows.append(
            {
                "sampling_method": method,
                "image_sum": float(capture["intensity"].sum()),
                "captured_power": float(capture["captured_power"]),
                **{
                    field: float(capture["stored_metrics"][field])
                    for field in OUTPUT_FIELDS
                },
            }
        )
    return rows


def copy_setup_with_position(
    base_setup: OpticalSetup,
    position_field: str,
    position_mm: float,
) -> OpticalSetup:
    setup = copy.deepcopy(base_setup)
    if position_field == "lens_x_mm":
        setup.lens.x_offset = position_mm * 1e-3
    elif position_field == "lens_y_mm":
        setup.lens.y_offset = position_mm * 1e-3
    elif position_field == "camera_x_mm":
        setup.camera.x_offset = position_mm * 1e-3
    elif position_field == "camera_y_mm":
        setup.camera.y_offset = position_mm * 1e-3
    else:
        raise KeyError(position_field)
    return setup


def source_and_propagated_quantities(
    setup: OpticalSetup,
) -> dict[str, Any]:
    """Expose the complete absolute-amplitude data flow for a power audit."""

    source, grid_x, grid_y, spacing, at_lens = prepare_source_and_lens(setup)
    after_lens, field_at_camera = propagate_from_lens(
        setup,
        at_lens,
        grid_x,
        grid_y,
        spacing,
    )
    capture = capture_from_field(
        field_at_camera,
        grid_x,
        grid_y,
        setup,
        "production_searchsorted",
    )
    return {
        "source": source,
        "at_lens": at_lens,
        "after_lens": after_lens,
        "field_at_camera": field_at_camera,
        "capture": capture,
        "spacing": spacing,
        "source_peak_amplitude": float(np.abs(source).max()),
        "source_integrated_intensity": float(
            np.square(np.abs(source)).sum() * spacing**2
        ),
        "camera_grid_integrated_intensity": float(
            np.square(np.abs(field_at_camera)).sum() * spacing**2
        ),
    }


def current_bounds() -> Bounds:
    """The configured v12 action/position contract, for boundary tests."""

    return Bounds(
        action_low=np.asarray([-0.05, -0.05, -0.02, -0.02], dtype=np.float64),
        action_high=np.asarray([0.05, 0.05, 0.02, 0.02], dtype=np.float64),
        position_low=np.asarray([-3.0, -3.0, -3.0, -3.0], dtype=np.float64),
        position_high=np.asarray([3.0, 3.0, 3.0, 3.0], dtype=np.float64),
    )


def beam_metrics_to_mapping(metrics: BeamMetrics) -> dict[str, float]:
    return {
        "centroid_x_m": float(metrics.centroid_x),
        "centroid_y_m": float(metrics.centroid_y),
        "sigma_x_m": float(metrics.sigma_x),
        "sigma_y_m": float(metrics.sigma_y),
        "peak_intensity": float(metrics.peak_intensity),
    }
