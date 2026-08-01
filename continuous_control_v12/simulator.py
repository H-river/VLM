"""Explicit-mm adapter for the existing optical simulator."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from continuous_control_v12.contracts import (
    OUTPUT_FIELDS,
    POSITION_FIELDS,
    Bounds,
    metrics_dict,
    position_dict,
    position_vector,
    stable_seed,
    validate_positions,
)
from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.metrics import compute_metrics
from optical_sim.src.optical_elements import OpticalSetup, setup_from_dict
from optical_sim.src.simulator import (
    _BACKENDS,
    _extract_sensor_region,
    _extract_sensor_region_continuous,
    apply_thin_lens,
    gaussian_source_field,
    normalize_field_to_power,
)
from optics_sft.physics.sim_adapter import (
    metrics_to_sensor_frame_state,
    metrics_to_state,
)


REGIMES = (
    "ordinary",
    "focusing",
    "clipping",
    "camera_boundary",
    "tolerance_boundary",
    "high_offset_interaction",
)

LEGACY_SEMANTICS_VERSION = "legacy_v1_searchsorted_power_dead"
CORRECTED_SEMANTICS_VERSION = "v12_sensor_power_semantics_v1"
CORRECTED_SAMPLING_METHOD = "pixel_area_bilinear_intensity"


def default_simulator_fixed(
    base_config_path: str,
    *,
    grid_size: int | None = None,
    grid_extent_mm: float | None = None,
    sensor_resolution: list[int] | None = None,
    semantics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = load_yaml(base_config_path)
    fixed = {
        "sensor_resolution_px": list(
            sensor_resolution or base["sensor"]["resolution"]
        ),
        "grid_size": int(grid_size or base["simulation"]["grid_size"]),
        "grid_extent_mm": float(
            grid_extent_mm
            if grid_extent_mm is not None
            else float(base["simulation"]["grid_extent"]) * 1e3
        ),
        "propagation_backend": str(
            base["simulation"]["propagation_backend"]
        ),
        "alignment_defocus_mm": float(
            base.get("alignment", {}).get("defocus", 0.0)
        )
        * 1e3,
        "source_phase_mode": "fixed_zero",
        "source_shape": "fundamental_gaussian",
    }
    if semantics is None:
        return fixed
    version = str(semantics["simulator_semantics_version"])
    if version != CORRECTED_SEMANTICS_VERSION:
        raise ValueError(f"unknown requested simulator semantics: {version}")
    method = str(
        semantics.get("sensor_sampling_method", CORRECTED_SAMPLING_METHOD)
    )
    if method != CORRECTED_SAMPLING_METHOD:
        raise ValueError(
            "v12 corrected data must use pixel-area irradiance sampling; "
            "point samplers are diagnostic-only"
        )
    fixed.update(
        {
            "simulator_semantics_version": version,
            "sensor_sampling_method": method,
            "sensor_measurement_model": (
                "finite_pixel_area_average_of_bilinearly_interpolated_irradiance"
            ),
            "sensor_pixel_center_convention": (
                "camera_pose_plus_index_minus_half_extent_times_declared_pitch"
            ),
            "sensor_origin_convention": (
                "camera_pose_is_center_of_sensor_pixel_edge_rectangle"
            ),
            "axis_convention": "array_axis_0_plus_y_lab_axis_1_plus_x_lab",
            "outside_propagated_field_rule": "zero_padding",
            "pixel_area_quadrature_order": int(
                semantics.get("pixel_area_quadrature_order", 3)
            ),
            "power_semantics": "source_integrated_optical_power_w",
            "intensity_normalization": "none_raw_irradiance_w_per_m2",
            "metrics_frame": "lab_frame_legacy_pseudo_pixels",
            "image_frame": "sensor_frame",
            "target_control_frame": "lab_frame_legacy_pseudo_pixels",
            "lab_to_sensor_transform": (
                "centroid_sensor_px=centroid_lab_px-camera_pose_m/pitch_m;"
                "sigma_sensor_px=sigma_lab_px"
            ),
        }
    )
    if "minimum_initial_captured_power_fraction" in semantics:
        minimum_fraction = float(
            semantics["minimum_initial_captured_power_fraction"]
        )
        if not 0.0 <= minimum_fraction < 1.0:
            raise ValueError(
                "minimum initial captured-power fraction must be in [0, 1)"
            )
        fixed["minimum_initial_captured_power_fraction"] = minimum_fraction
    if "maximum_setup_resample_attempts" in semantics:
        attempts = int(semantics["maximum_setup_resample_attempts"])
        if attempts < 1:
            raise ValueError("maximum setup resample attempts must be positive")
        fixed["maximum_setup_resample_attempts"] = attempts
    return fixed


def is_corrected_semantics(simulator_fixed: Mapping[str, Any]) -> bool:
    version = simulator_fixed.get("simulator_semantics_version")
    if version is None:
        return False
    if version != CORRECTED_SEMANTICS_VERSION:
        raise ValueError(f"unknown simulator semantics in dataset: {version}")
    return True


def sample_group_setup(
    regime: str,
    group_id: str,
    seed: int,
    simulator_fixed: Mapping[str, Any],
    bounds: Bounds,
    *,
    setup_attempt: int = 0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Sample from repository-established v9/v10 simulator regimes."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime}")
    if setup_attempt < 0:
        raise ValueError("setup attempt must be non-negative")
    setup_seed = (
        stable_seed(seed, group_id, regime, "setup")
        if setup_attempt == 0
        else stable_seed(
            seed, group_id, regime, "setup_retry", setup_attempt
        )
    )
    rng = np.random.default_rng(setup_seed)
    setup = {
        "wavelength_nm": float(rng.uniform(617.0, 648.0)),
        "beam_waist_mm": float(rng.uniform(0.70, 1.35)),
        "power_w": float(rng.uniform(0.75, 1.25)),
        "lens_focal_length_mm": float(rng.uniform(82.0, 120.0)),
        "lens_aperture_mm": float(rng.uniform(19.0, 32.0)),
        "source_to_lens_mm": float(rng.uniform(140.0, 260.0)),
        "lens_to_camera_mm": float(rng.uniform(92.0, 190.0)),
        "pixel_size_um": float(rng.choice((5.0, 5.5, 6.0))),
    }
    positions = {
        "lens_x_mm": float(rng.uniform(-0.23, 0.23)),
        "lens_y_mm": float(rng.uniform(-0.23, 0.23)),
        "camera_x_mm": float(rng.uniform(-0.18, 0.18)),
        "camera_y_mm": float(rng.uniform(-0.18, 0.18)),
    }
    if regime == "ordinary":
        setup["pixel_size_um"] = 5.5
    elif regime in {"focusing", "tolerance_boundary"}:
        focal = float(rng.uniform(72.0, 128.0))
        source = float(rng.uniform(max(1.18 * focal, 120.0), 282.0))
        image_distance = focal * source / max(source - focal, 1e-6)
        width = 0.010 if regime == "tolerance_boundary" else 0.020
        setup.update(
            {
                "lens_focal_length_mm": focal,
                "source_to_lens_mm": source,
                "lens_to_camera_mm": float(
                    np.clip(
                        image_distance * rng.uniform(1.0 - width, 1.0 + width),
                        72.0,
                        255.0,
                    )
                ),
                "lens_aperture_mm": float(rng.uniform(17.0, 30.0)),
            }
        )
    elif regime == "clipping":
        setup["lens_aperture_mm"] = float(rng.uniform(1.2, 4.0))
        positions["lens_x_mm"] = float(
            rng.choice((-1.0, 1.0)) * rng.uniform(0.20, 1.00)
        )
        positions["lens_y_mm"] = float(
            rng.choice((-1.0, 1.0)) * rng.uniform(0.20, 1.00)
        )
    elif regime == "high_offset_interaction":
        setup["lens_aperture_mm"] = float(rng.uniform(14.0, 21.0))
        positions.update(
            {
                "lens_x_mm": float(
                    rng.choice((-1.0, 1.0)) * rng.uniform(0.24, 0.38)
                ),
                "lens_y_mm": float(
                    rng.choice((-1.0, 1.0)) * rng.uniform(0.24, 0.38)
                ),
                "camera_x_mm": float(
                    rng.choice((-1.0, 1.0)) * rng.uniform(0.17, 0.28)
                ),
                "camera_y_mm": float(
                    rng.choice((-1.0, 1.0)) * rng.uniform(0.17, 0.28)
                ),
            }
        )
    if regime in {"camera_boundary", "tolerance_boundary"}:
        height, width_px = simulator_fixed["sensor_resolution_px"]
        pitch_mm = setup["pixel_size_um"] / 1000.0
        half = (width_px if rng.random() < 0.5 else height) * pitch_mm / 2.0
        axis = int(rng.integers(0, 2))
        fraction_low = 0.68 if regime == "camera_boundary" else 0.72
        fraction_high = 1.0 if regime == "camera_boundary" else 0.92
        value = float(
            rng.choice((-1.0, 1.0))
            * rng.uniform(fraction_low, fraction_high)
            * half
        )
        positions["camera_x_mm" if axis == 0 else "camera_y_mm"] = value
    clipped = np.clip(
        position_vector(positions), bounds.position_low, bounds.position_high
    )
    positions = position_dict(clipped)
    validate_positions(positions, bounds)
    return setup, positions


def build_optical_setup(
    setup_context: Mapping[str, Any],
    positions_mm: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    base_config_path: str,
) -> OpticalSetup:
    cfg = copy.deepcopy(load_yaml(base_config_path))
    cfg["source"]["wavelength"] = float(setup_context["wavelength_nm"]) * 1e-9
    cfg["source"]["beam_waist"] = float(setup_context["beam_waist_mm"]) * 1e-3
    cfg["source"]["power"] = float(setup_context["power_w"])
    cfg["lens"]["focal_length"] = (
        float(setup_context["lens_focal_length_mm"]) * 1e-3
    )
    cfg["lens"]["clear_aperture"] = (
        float(setup_context["lens_aperture_mm"]) * 1e-3
    )
    cfg["geometry"]["laser_to_lens"] = (
        float(setup_context["source_to_lens_mm"]) * 1e-3
    )
    cfg["geometry"]["lens_to_camera"] = (
        float(setup_context["lens_to_camera_mm"]) * 1e-3
    )
    cfg["sensor"]["pixel_pitch"] = float(setup_context["pixel_size_um"]) * 1e-6
    cfg["sensor"]["resolution"] = [
        int(value) for value in simulator_fixed["sensor_resolution_px"]
    ]
    cfg["lens"]["x_offset"] = float(positions_mm["lens_x_mm"]) * 1e-3
    cfg["lens"]["y_offset"] = float(positions_mm["lens_y_mm"]) * 1e-3
    cfg.setdefault("camera", {})
    cfg["camera"]["x_offset"] = float(positions_mm["camera_x_mm"]) * 1e-3
    cfg["camera"]["y_offset"] = float(positions_mm["camera_y_mm"]) * 1e-3
    cfg.setdefault("alignment", {})
    cfg["alignment"]["defocus"] = (
        float(simulator_fixed["alignment_defocus_mm"]) * 1e-3
    )
    cfg["simulation"]["grid_size"] = int(simulator_fixed["grid_size"])
    cfg["simulation"]["grid_extent"] = (
        float(simulator_fixed["grid_extent_mm"]) * 1e-3
    )
    cfg["simulation"]["propagation_backend"] = str(
        simulator_fixed["propagation_backend"]
    )
    return setup_from_dict(cfg)


def simulate_state(
    setup_context: Mapping[str, Any],
    positions_mm: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    base_config_path: str,
    bounds: Bounds | None = None,
) -> dict[str, Any]:
    """Run explicitly selected legacy or corrected simulator semantics."""

    if bounds is not None:
        validate_positions(positions_mm, bounds)
    setup = build_optical_setup(
        setup_context, positions_mm, simulator_fixed, base_config_path
    )
    propagate = _BACKENDS.get(
        setup.propagation_backend, _BACKENDS["fresnel_numpy"]
    )
    source, grid_x, grid_y, spacing = gaussian_source_field(setup)
    corrected = is_corrected_semantics(simulator_fixed)
    source_scale = 1.0
    if corrected:
        source, source_scale = normalize_field_to_power(
            source,
            spacing,
            float(setup_context["power_w"]),
        )
    source_integrated_power = float(
        np.square(np.abs(source)).sum() * spacing**2
    )
    at_lens = propagate(
        source,
        spacing,
        setup.laser_to_lens,
        setup.source.wavelength,
    )
    incoming_power = max(float(np.square(np.abs(at_lens)).sum()), 1e-30)
    after_lens = apply_thin_lens(at_lens, grid_x, grid_y, setup)
    transmission = float(
        np.square(np.abs(after_lens)).sum() / incoming_power
    )
    field_at_camera = propagate(
        after_lens,
        spacing,
        setup.effective_camera_distance,
        setup.source.wavelength,
    )
    if corrected:
        (
            intensity,
            sensor_x,
            sensor_y,
            valid_region,
            sampling_metadata,
        ) = _extract_sensor_region_continuous(
            field_at_camera,
            grid_x,
            grid_y,
            setup,
            method=str(simulator_fixed["sensor_sampling_method"]),
            quadrature_order=int(
                simulator_fixed["pixel_area_quadrature_order"]
            ),
        )
    else:
        intensity, sensor_x, sensor_y = _extract_sensor_region(
            field_at_camera, grid_x, grid_y, setup
        )
        valid_region = np.ones_like(intensity, dtype=bool)
        sampling_metadata = {
            "method": "legacy_left_searchsorted_integer_lookup",
            "measurement_model": "point_lookup_of_grid_intensity",
            "quadrature_order": 0,
            "outside_propagated_field_rule": "clipped_to_edge_sample",
            "simulation_grid_pitch_x_m": float(spacing),
            "simulation_grid_pitch_y_m": float(spacing),
            "sensor_pixel_pitch_m": float(setup.sensor.pixel_pitch),
            "axis_0_direction": "+y_lab",
            "axis_1_direction": "+x_lab",
            "valid_region_fraction": 1.0,
        }
    raw_image = np.asarray(intensity, dtype=np.float32)
    measured = compute_metrics(raw_image, sensor_x, sensor_y)
    legacy = metrics_to_state(measured, setup)
    sensor_frame = metrics_to_sensor_frame_state(measured, setup)
    metrics = metrics_dict([legacy[field] for field in OUTPUT_FIELDS])
    metrics_sensor_frame = metrics_dict(
        [sensor_frame[field] for field in OUTPUT_FIELDS]
    )
    height, width = setup.sensor.resolution
    edge_distance = float(
        min(
            sensor_frame["centroid_x_px"],
            (width - 1) - sensor_frame["centroid_x_px"],
            sensor_frame["centroid_y_px"],
            (height - 1) - sensor_frame["centroid_y_px"],
        )
    )
    values = np.asarray(list(metrics.values()), dtype=np.float64)
    valid = bool(
        np.isfinite(values).all()
        and np.isfinite(raw_image).all()
        and np.all(raw_image >= 0)
    )
    at_limit = None
    if bounds is not None:
        positions = position_vector(positions_mm)
        at_limit = bool(
            np.any(np.isclose(positions, bounds.position_low, atol=1e-10))
            or np.any(np.isclose(positions, bounds.position_high, atol=1e-10))
        )
    peak = float(raw_image.max())
    normalized_image = (
        raw_image / peak if peak > 0.0 else np.zeros_like(raw_image)
    )
    captured_power_w = float(
        np.asarray(raw_image, dtype=np.float64).sum()
        * setup.sensor.pixel_pitch**2
    )
    camera_pose = {
        "x_mm": float(setup.camera.x_offset * 1e3),
        "y_mm": float(setup.camera.y_offset * 1e3),
    }
    sensor_origin = {
        "meaning": "first_pixel_center_in_lab_frame",
        "x_mm": float(sensor_x[0, 0] * 1e3),
        "y_mm": float(sensor_y[0, 0] * 1e3),
    }
    coordinate_semantics = {
        "metrics_lab_frame": "legacy_pseudo_pixels_at_declared_pitch",
        "image_sensor_frame": "array_centred_on_camera_pose",
        "camera_pose_lab": camera_pose,
        "sensor_pixel_pitch_um": float(setup.sensor.pixel_pitch * 1e6),
        "sensor_origin": sensor_origin,
        "axis_convention": "axis_0_plus_y_lab_axis_1_plus_x_lab",
        "lab_to_sensor_transform": (
            "centroid_sensor_px=centroid_lab_px-camera_pose_m/pitch_m;"
            "sigma_sensor_px=sigma_lab_px"
        ),
        "target_control_frame": (
            "lab_frame_legacy_pseudo_pixels"
            if corrected
            else "legacy_implicit_lab_frame_pseudo_pixels"
        ),
    }
    auxiliary = {
        "captured_power": captured_power_w,
        "clipping_fraction": float(1.0 - transmission),
        "camera_boundary_indicator": bool(edge_distance <= 0.0),
        "actuator_limit_indicator": at_limit,
        "phase_descriptor": None,
        "simulator_valid": valid,
        "distance_to_camera_boundary_px": edge_distance,
    }
    if corrected:
        auxiliary.update(
            {
                "captured_power_w": captured_power_w,
                "valid_region_fraction": float(valid_region.mean()),
                "source_integrated_power_w": source_integrated_power,
                "source_amplitude_scale": float(source_scale),
                "peak_intensity_abs": float(metrics["peak_intensity"]),
            }
        )
    return {
        "metrics": metrics,
        "metrics_lab_frame": metrics,
        "metrics_sensor_frame": metrics_sensor_frame,
        "intensity": raw_image,
        "image_raw": raw_image,
        "image_normalized": np.asarray(normalized_image, dtype=np.float32),
        "valid_region_mask": valid_region,
        "sampling_metadata": sampling_metadata,
        "coordinate_semantics": coordinate_semantics,
        "power_semantics": (
            "source_integrated_optical_power_w"
            if corrected
            else "legacy_declared_but_ignored"
        ),
        "intensity_normalization": (
            "none_raw_irradiance_w_per_m2"
            if corrected
            else "legacy_unnormalized_simulator_units"
        ),
        "auxiliary": auxiliary,
    }
