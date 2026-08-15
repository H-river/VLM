#!/usr/bin/env python3
"""Quantitative legacy-versus-v12 sensor/power/frame acceptance audit."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import (
    OUTPUT_FIELDS,
    Bounds,
    metrics_dict,
    tolerance_vector,
)
from continuous_control_v12.simulator import (
    REGIMES,
    build_optical_setup,
    default_simulator_fixed,
    sample_group_setup,
    simulate_state,
)
from optical_sim.src.metrics import compute_metrics
from optical_sim.src.simulator import (
    _BACKENDS,
    _extract_sensor_region,
    _extract_sensor_region_continuous,
    apply_thin_lens,
    gaussian_source_field,
    normalize_field_to_power,
)
from legacy.experiments.audits.simulator_three_issue_audit_v1.scripts import (
    audit_core as simulator_audit_core,
)

BASE_CONTEXT = simulator_audit_core.BASE_CONTEXT
BASE_POSITIONS_MM = simulator_audit_core.BASE_POSITIONS_MM
independent_sensor_metrics = simulator_audit_core.independent_sensor_metrics
from optics_sft.physics.sim_adapter import (
    metrics_to_sensor_frame_state,
    metrics_to_state,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "continuous_control_v12/config_v12_semantics_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--grid-extent-mm", type=float)
    parser.add_argument("--sensor-resolution", type=int, default=1024)
    parser.add_argument("--sweep-half-range-mm", type=float, default=0.09)
    parser.add_argument("--sweep-step-mm", type=float, default=0.0015)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_fields(
    setup: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    source, grid_x, grid_y, spacing = gaussian_source_field(setup)
    source, _ = normalize_field_to_power(source, spacing, setup.source.power)
    propagate = _BACKENDS.get(
        setup.propagation_backend, _BACKENDS["fresnel_numpy"]
    )
    at_lens = propagate(
        source, spacing, setup.laser_to_lens, setup.source.wavelength
    )
    after_lens = apply_thin_lens(at_lens, grid_x, grid_y, setup)
    field = propagate(
        after_lens,
        spacing,
        setup.effective_camera_distance,
        setup.source.wavelength,
    )
    return source, grid_x, grid_y, spacing, at_lens, field


def capture(
    field: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    setup: Any,
    method: str,
    *,
    quadrature_order: int = 3,
) -> dict[str, Any]:
    if method == "legacy_left_searchsorted":
        intensity, sensor_x, sensor_y = _extract_sensor_region(
            field, grid_x, grid_y, setup
        )
        x_axis = grid_x[0, :]
        y_axis = grid_y[:, 0]
        sx = sensor_x[0, :]
        sy = sensor_y[:, 0]
        selected = np.searchsorted(x_axis, sx).clip(0, len(x_axis) - 1)
        selected_y = np.searchsorted(y_axis, sy).clip(0, len(y_axis) - 1)
        metadata = {
            "center_effective_coordinate_x": float(
                selected[len(selected) // 2]
            ),
            "center_effective_coordinate_y": float(
                selected_y[len(selected_y) // 2]
            ),
            "coordinate_kind": "integer_grid_index",
            "coordinate_signature": hashlib.sha256(
                selected.tobytes() + selected_y.tobytes()
            ).hexdigest()[:16],
            "valid_region_fraction": 1.0,
        }
    else:
        (
            intensity,
            sensor_x,
            sensor_y,
            valid,
            sampling,
        ) = _extract_sensor_region_continuous(
            field,
            grid_x,
            grid_y,
            setup,
            method=method,
            quadrature_order=quadrature_order,
        )
        metadata = {
            "center_effective_coordinate_x": sampling[
                "center_fractional_grid_index_x"
            ],
            "center_effective_coordinate_y": sampling[
                "center_fractional_grid_index_y"
            ],
            "coordinate_kind": "continuous_fractional_grid_index",
            "coordinate_signature": (
                f"{sampling['center_fractional_grid_index_x']:.15g}:"
                f"{sampling['center_fractional_grid_index_y']:.15g}"
            ),
            "valid_region_fraction": float(valid.mean()),
        }
    raw = np.asarray(intensity, dtype=np.float32)
    measured = compute_metrics(raw, sensor_x, sensor_y)
    lab = metrics_to_state(measured, setup)
    sensor = metrics_to_sensor_frame_state(measured, setup)
    lab_metrics = metrics_dict([lab[field] for field in OUTPUT_FIELDS])
    sensor_metrics = metrics_dict(
        [sensor[field] for field in OUTPUT_FIELDS]
    )
    peak = max(float(raw.max()), 1e-30)
    return {
        "raw": raw,
        "normalized": raw / peak,
        "lab_metrics": lab_metrics,
        "sensor_metrics": sensor_metrics,
        "captured_power_w": float(
            np.asarray(raw, dtype=np.float64).sum()
            * setup.sensor.pixel_pitch**2
        ),
        "metadata": metadata,
    }


def moved_setup(setup: Any, field: str, value_mm: float) -> Any:
    output = copy.deepcopy(setup)
    object_name, axis, _ = field.split("_", 2)
    target = output.camera if object_name == "camera" else output.lens
    setattr(target, f"{axis}_offset", float(value_mm) * 1e-3)
    return output


def sweep_rows(
    setup: Any,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    spacing: float,
    at_lens: np.ndarray,
    base_field: np.ndarray,
    *,
    half_range_mm: float,
    step_mm: float,
) -> list[dict[str, Any]]:
    count = int(round(2.0 * half_range_mm / step_mm)) + 1
    offsets = np.linspace(-half_range_mm, half_range_mm, count)
    methods = (
        "legacy_left_searchsorted",
        "pixel_area_bilinear_intensity",
    )
    propagate = _BACKENDS.get(
        setup.propagation_backend, _BACKENDS["fresnel_numpy"]
    )
    rows: list[dict[str, Any]] = []
    for position_field in (
        "camera_x_mm",
        "camera_y_mm",
        "lens_x_mm",
        "lens_y_mm",
    ):
        reference = float(BASE_POSITIONS_MM[position_field])
        previous: dict[str, dict[str, Any] | None] = {
            method: None for method in methods
        }
        for offset in offsets:
            value = reference + float(offset)
            active_setup = moved_setup(setup, position_field, value)
            if position_field.startswith("lens"):
                after_lens = apply_thin_lens(
                    at_lens, grid_x, grid_y, active_setup
                )
                field = propagate(
                    after_lens,
                    spacing,
                    active_setup.effective_camera_distance,
                    active_setup.source.wavelength,
                )
            else:
                field = base_field
            for method in methods:
                current = capture(
                    field, grid_x, grid_y, active_setup, method
                )
                row: dict[str, Any] = {
                    "sweep_id": f"interior_{position_field}",
                    "position_field": position_field,
                    "position_mm": value,
                    "step_mm": step_mm,
                    "sampling_method": method,
                    **current["metadata"],
                    "captured_power_w": current["captured_power_w"],
                }
                for metric in OUTPUT_FIELDS:
                    row[metric] = current["lab_metrics"][metric]
                prior = previous[method]
                if prior is None:
                    row.update(
                        {
                            "exact_image_plateau": False,
                            "exact_metric_plateau": False,
                            "index_boundary_crossed": False,
                            "raw_image_relative_l1_change": 0.0,
                            "normalized_image_mean_abs_change": 0.0,
                            "max_tolerance_normalized_delta": 0.0,
                        }
                    )
                    for metric in OUTPUT_FIELDS:
                        row[f"delta_{metric}"] = 0.0
                        row[f"tolerance_normalized_delta_{metric}"] = 0.0
                        row[f"derivative_{metric}_per_mm"] = 0.0
                else:
                    delta = np.asarray(
                        [
                            current["lab_metrics"][metric]
                            - prior["lab_metrics"][metric]
                            for metric in OUTPUT_FIELDS
                        ],
                        dtype=np.float64,
                    )
                    normalized = np.abs(delta) / tolerance_vector(
                        prior["lab_metrics"]
                    )
                    row.update(
                        {
                            "exact_image_plateau": bool(
                                np.array_equal(current["raw"], prior["raw"])
                            ),
                            "exact_metric_plateau": bool(
                                current["lab_metrics"]
                                == prior["lab_metrics"]
                            ),
                            "index_boundary_crossed": bool(
                                current["metadata"]["coordinate_signature"]
                                != prior["metadata"]["coordinate_signature"]
                            ),
                            "raw_image_relative_l1_change": float(
                                np.abs(current["raw"] - prior["raw"]).sum()
                                / max(
                                    float(np.abs(prior["raw"]).sum()), 1e-30
                                )
                            ),
                            "normalized_image_mean_abs_change": float(
                                np.abs(
                                    current["normalized"]
                                    - prior["normalized"]
                                ).mean()
                            ),
                            "max_tolerance_normalized_delta": float(
                                normalized.max()
                            ),
                        }
                    )
                    for index, metric in enumerate(OUTPUT_FIELDS):
                        row[f"delta_{metric}"] = float(delta[index])
                        row[
                            f"tolerance_normalized_delta_{metric}"
                        ] = float(normalized[index])
                        row[f"derivative_{metric}_per_mm"] = float(
                            delta[index] / step_mm
                        )
                rows.append(row)
                previous[method] = current
    return rows


def boundary_rows(
    setup: Any,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    field: np.ndarray,
    *,
    step_mm: float,
) -> list[dict[str, Any]]:
    values = np.arange(2.65, 2.95 + step_mm / 2.0, step_mm)
    rows: list[dict[str, Any]] = []
    previous: dict[str, dict[str, Any] | None] = {
        method: None
        for method in (
            "legacy_left_searchsorted",
            "pixel_area_bilinear_intensity",
        )
    }
    for value in values:
        active = moved_setup(setup, "camera_x_mm", float(value))
        for method in previous:
            current = capture(field, grid_x, grid_y, active, method)
            prior = previous[method]
            row = {
                "sweep_id": "clipping_boundary_camera_x_mm",
                "position_field": "camera_x_mm",
                "position_mm": float(value),
                "step_mm": step_mm,
                "sampling_method": method,
                **current["metadata"],
                "captured_power_w": current["captured_power_w"],
            }
            for metric in OUTPUT_FIELDS:
                row[metric] = current["lab_metrics"][metric]
            if prior is None:
                delta = np.zeros(5)
                normalized = np.zeros(5)
            else:
                delta = np.asarray(
                    [
                        current["lab_metrics"][metric]
                        - prior["lab_metrics"][metric]
                        for metric in OUTPUT_FIELDS
                    ]
                )
                normalized = np.abs(delta) / tolerance_vector(
                    prior["lab_metrics"]
                )
            row.update(
                {
                    "exact_image_plateau": bool(
                        prior is not None
                        and np.array_equal(current["raw"], prior["raw"])
                    ),
                    "exact_metric_plateau": bool(
                        prior is not None
                        and current["lab_metrics"] == prior["lab_metrics"]
                    ),
                    "index_boundary_crossed": bool(
                        prior is not None
                        and current["metadata"]["coordinate_signature"]
                        != prior["metadata"]["coordinate_signature"]
                    ),
                    "raw_image_relative_l1_change": (
                        0.0
                        if prior is None
                        else float(
                            np.abs(current["raw"] - prior["raw"]).sum()
                            / max(float(np.abs(prior["raw"]).sum()), 1e-30)
                        )
                    ),
                    "normalized_image_mean_abs_change": (
                        0.0
                        if prior is None
                        else float(
                            np.abs(
                                current["normalized"] - prior["normalized"]
                            ).mean()
                        )
                    ),
                    "max_tolerance_normalized_delta": float(normalized.max()),
                }
            )
            for index, metric in enumerate(OUTPUT_FIELDS):
                row[f"delta_{metric}"] = float(delta[index])
                row[f"tolerance_normalized_delta_{metric}"] = float(
                    normalized[index]
                )
                row[f"derivative_{metric}_per_mm"] = float(
                    delta[index] / step_mm
                )
            rows.append(row)
            previous[method] = current
    return rows


def continuity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def longest_plateau_points(flags: list[bool]) -> int:
        longest_transitions = 0
        current_transitions = 0
        for flag in flags:
            if flag:
                current_transitions += 1
                longest_transitions = max(
                    longest_transitions, current_transitions
                )
            else:
                current_transitions = 0
        return longest_transitions + 1

    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(row["sweep_id"], row["sampling_method"])].append(row)
    output = []
    for (sweep_id, method), group in grouped.items():
        adjacent = np.asarray(
            [row["max_tolerance_normalized_delta"] for row in group[1:]],
            dtype=np.float64,
        )
        image_plateaus = [
            bool(row["exact_image_plateau"]) for row in group[1:]
        ]
        metric_plateaus = [
            bool(row["exact_metric_plateau"]) for row in group[1:]
        ]
        coordinate_plateaus = [
            not bool(row["index_boundary_crossed"]) for row in group[1:]
        ]
        summary: dict[str, Any] = {
            "sweep_id": sweep_id,
            "sampling_method": method,
            "points": len(group),
            "step_mm": group[0]["step_mm"],
            "exact_image_plateau_count": int(sum(image_plateaus)),
            "longest_exact_image_plateau_points": longest_plateau_points(
                image_plateaus
            ),
            "exact_metric_plateau_count": int(sum(metric_plateaus)),
            "longest_exact_metric_plateau_points": longest_plateau_points(
                metric_plateaus
            ),
            "effective_coordinate_plateau_transition_count": int(
                sum(coordinate_plateaus)
            ),
            "longest_effective_coordinate_plateau_points": (
                longest_plateau_points(coordinate_plateaus)
            ),
            "index_boundary_transition_count": int(
                sum(bool(row["index_boundary_crossed"]) for row in group[1:])
            ),
            "max_adjacent_tolerance_change": float(adjacent.max()),
            "p50_adjacent_tolerance_change": float(
                np.quantile(adjacent, 0.50)
            ),
            "p95_adjacent_tolerance_change": float(
                np.quantile(adjacent, 0.95)
            ),
            "p99_adjacent_tolerance_change": float(
                np.quantile(adjacent, 0.99)
            ),
        }
        for metric in OUTPUT_FIELDS:
            changes = np.asarray(
                [
                    row[f"tolerance_normalized_delta_{metric}"]
                    for row in group[1:]
                ],
                dtype=np.float64,
            )
            derivative = np.asarray(
                [
                    row[f"derivative_{metric}_per_mm"]
                    for row in group[1:]
                ],
                dtype=np.float64,
            )
            nonzero = derivative[np.abs(derivative) > 1e-12]
            median = (
                float(np.median(np.abs(nonzero))) if len(nonzero) else 0.0
            )
            summary[f"max_tolerance_change_{metric}"] = float(changes.max())
            summary[f"p95_tolerance_change_{metric}"] = float(
                np.quantile(changes, 0.95)
            )
            summary[f"derivative_sign_changes_{metric}"] = int(
                np.sum(np.sign(nonzero[1:]) != np.sign(nonzero[:-1]))
                if len(nonzero) > 1
                else 0
            )
            summary[f"derivative_spikes_{metric}"] = int(
                np.sum(np.abs(nonzero) > 10.0 * median)
                if median > 0.0
                else 0
            )
        output.append(summary)
    return output


def power_rows(
    config: dict[str, Any],
    fixed: dict[str, Any],
    base_config: str,
    bounds: Bounds,
) -> list[dict[str, Any]]:
    rows = []
    reference: dict[str, Any] | None = None
    for power in (0.25, 0.5, 1.0, 2.0, 4.0):
        context = dict(BASE_CONTEXT)
        context["power_w"] = power
        result = simulate_state(
            context, BASE_POSITIONS_MM, fixed, base_config, bounds
        )
        if reference is None and power == 1.0:
            reference = result
        rows.append({"power_w": power, "capture": result})
    reference = next(
        row["capture"] for row in rows if row["power_w"] == 1.0
    )
    output = []
    for item in rows:
        power = item["power_w"]
        result = item["capture"]
        output.append(
            {
                "power_w": power,
                "expected_amplitude_ratio": float(np.sqrt(power)),
                "source_amplitude_ratio": float(
                    result["auxiliary"]["source_amplitude_scale"]
                    / reference["auxiliary"]["source_amplitude_scale"]
                ),
                "expected_intensity_ratio": power,
                "raw_image_sum_ratio": float(
                    result["image_raw"].sum()
                    / reference["image_raw"].sum()
                ),
                "captured_power_ratio": float(
                    result["auxiliary"]["captured_power_w"]
                    / reference["auxiliary"]["captured_power_w"]
                ),
                "peak_intensity_ratio": float(
                    result["metrics"]["peak_intensity"]
                    / reference["metrics"]["peak_intensity"]
                ),
                "normalized_image_max_abs_difference": float(
                    np.abs(
                        result["image_normalized"]
                        - reference["image_normalized"]
                    ).max()
                ),
                **{
                    f"{field}_difference": float(
                        result["metrics"][field]
                        - reference["metrics"][field]
                    )
                    for field in OUTPUT_FIELDS[:4]
                },
            }
        )
    return output


def coordinate_rows(
    fixed: dict[str, Any],
    base_config: str,
    bounds: Bounds,
) -> list[dict[str, Any]]:
    interventions = [
        ("base", {}),
        ("camera_x_plus_0p02_mm", {"camera_x_mm": 0.02}),
        ("camera_y_plus_0p02_mm", {"camera_y_mm": 0.02}),
        ("lens_x_plus_0p02_mm", {"lens_x_mm": 0.02}),
        ("lens_y_plus_0p02_mm", {"lens_y_mm": 0.02}),
    ]
    rows = []
    for name, deltas in interventions:
        positions = dict(BASE_POSITIONS_MM)
        for field, delta in deltas.items():
            positions[field] += delta
        result = simulate_state(
            BASE_CONTEXT, positions, fixed, base_config, bounds
        )
        independent = independent_sensor_metrics(
            result["image_raw"], BASE_CONTEXT["pixel_size_um"] * 1e-6
        )
        row: dict[str, Any] = {"intervention": name}
        tolerance = tolerance_vector(result["metrics"])
        for index, field in enumerate(OUTPUT_FIELDS):
            row[f"lab_{field}"] = result["metrics_lab_frame"][field]
            row[f"sensor_{field}"] = result["metrics_sensor_frame"][field]
            row[f"image_recomputed_{field}"] = independent[field]
            row[f"sensor_image_discrepancy_tolerance_{field}"] = float(
                abs(result["metrics_sensor_frame"][field] - independent[field])
                / tolerance[index]
            )
        rows.append(row)
    return rows


def reference_rows(
    setup: Any,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    spacing: float,
    at_lens: np.ndarray,
    base_field: np.ndarray,
) -> list[dict[str, Any]]:
    cases = [
        ("base", "camera_x_mm", BASE_POSITIONS_MM["camera_x_mm"]),
        (
            "camera_x_plus_0p003_mm",
            "camera_x_mm",
            BASE_POSITIONS_MM["camera_x_mm"] + 0.003,
        ),
        ("clipping_camera_x_2p8_mm", "camera_x_mm", 2.8),
        (
            "lens_x_plus_0p003_mm",
            "lens_x_mm",
            BASE_POSITIONS_MM["lens_x_mm"] + 0.003,
        ),
    ]
    propagate = _BACKENDS.get(
        setup.propagation_backend, _BACKENDS["fresnel_numpy"]
    )
    rows = []
    for case, field_name, value in cases:
        active = moved_setup(setup, field_name, value)
        if field_name.startswith("lens"):
            field = propagate(
                apply_thin_lens(at_lens, grid_x, grid_y, active),
                spacing,
                active.effective_camera_distance,
                active.source.wavelength,
            )
        else:
            field = base_field
        reference = capture(
            field,
            grid_x,
            grid_y,
            active,
            "pixel_area_bilinear_intensity",
            quadrature_order=9,
        )
        reference_tolerance = tolerance_vector(reference["lab_metrics"])
        for method, order in (
            ("pixel_area_bilinear_intensity", 3),
            ("point_bilinear_intensity", 1),
            ("point_bilinear_complex_field", 1),
        ):
            candidate = capture(
                field,
                grid_x,
                grid_y,
                active,
                method,
                quadrature_order=order,
            )
            delta = np.asarray(
                [
                    candidate["lab_metrics"][metric]
                    - reference["lab_metrics"][metric]
                    for metric in OUTPUT_FIELDS
                ]
            )
            row = {
                "case": case,
                "method": method,
                "reference": "pixel_area_bilinear_intensity_order_9",
                "raw_image_relative_l1_error": float(
                    np.abs(candidate["raw"] - reference["raw"]).sum()
                    / max(float(np.abs(reference["raw"]).sum()), 1e-30)
                ),
                "captured_power_ratio_to_reference": float(
                    candidate["captured_power_w"]
                    / reference["captured_power_w"]
                ),
                "max_metric_error_in_tolerances": float(
                    np.max(np.abs(delta) / reference_tolerance)
                ),
            }
            for index, metric in enumerate(OUTPUT_FIELDS):
                row[f"{metric}_error_in_tolerances"] = float(
                    abs(delta[index]) / reference_tolerance[index]
                )
            rows.append(row)
    return rows


def resolution_rows(
    config: dict[str, Any],
    base_config: str,
    bounds: Bounds,
    sensor_resolution: int,
    grid_extent_mm: float,
    configured_grid_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for grid_size in sorted({512, 1024, configured_grid_size, 2048}):
        fixed = default_simulator_fixed(
            base_config,
            grid_size=grid_size,
            grid_extent_mm=grid_extent_mm,
            sensor_resolution=[sensor_resolution, sensor_resolution],
            semantics=config["simulator"]["semantics"],
        )
        started = time.perf_counter()
        result = simulate_state(
            BASE_CONTEXT, BASE_POSITIONS_MM, fixed, base_config, bounds
        )
        row = {
            "grid_size": grid_size,
            "simulation_grid_pitch_mm": (
                2.0 * grid_extent_mm / (grid_size - 1)
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "captured_power_w": result["auxiliary"]["captured_power_w"],
            **result["metrics"],
        }
        rows.append(row)
        del result
        gc.collect()
    reference = rows[-1]
    reference_tolerance = tolerance_vector(reference)
    for row in rows:
        delta = np.asarray(
            [row[field] - reference[field] for field in OUTPUT_FIELDS]
        )
        row["reference_grid_size"] = 2048
        row["max_metric_difference_in_tolerances"] = float(
            np.max(np.abs(delta) / reference_tolerance)
        )
        row["captured_power_relative_difference"] = float(
            abs(row["captured_power_w"] - reference["captured_power_w"])
            / reference["captured_power_w"]
        )
    return rows


def regime_resolution_rows(
    config: dict[str, Any],
    base_config: str,
    bounds: Bounds,
    sensor_resolution: int,
    grid_extent_mm: float,
    configured_grid_size: int,
) -> list[dict[str, Any]]:
    fixed_by_size = {
        size: default_simulator_fixed(
            base_config,
            grid_size=size,
            grid_extent_mm=grid_extent_mm,
            sensor_resolution=[sensor_resolution, sensor_resolution],
            semantics=config["simulator"]["semantics"],
        )
        for size in (configured_grid_size, 2048)
    }
    rows = []
    for regime in REGIMES:
        for replicate in range(2):
            group_id = f"resolution_{regime}_{replicate}"
            context, positions = sample_group_setup(
                regime,
                group_id,
                int(config["seed"]),
                fixed_by_size[configured_grid_size],
                bounds,
            )
            captures = {
                size: simulate_state(
                    context,
                    positions,
                    fixed,
                    base_config,
                    bounds,
                )
                for size, fixed in fixed_by_size.items()
            }
            reference = captures[2048]
            tolerance = tolerance_vector(reference["metrics"])
            delta = np.asarray(
                [
                    captures[configured_grid_size]["metrics"][field]
                    - reference["metrics"][field]
                    for field in OUTPUT_FIELDS
                ]
            )
            row = {
                "regime": regime,
                "replicate": replicate,
                "group_id": group_id,
                "grid_extent_mm": grid_extent_mm,
                "max_metric_difference_in_tolerances": float(
                    np.max(np.abs(delta) / tolerance)
                ),
                "captured_power_relative_difference": float(
                    abs(
                        captures[configured_grid_size]["auxiliary"][
                            "captured_power_w"
                        ]
                        - reference["auxiliary"]["captured_power_w"]
                    )
                    / max(
                        reference["auxiliary"]["captured_power_w"], 1e-30
                    )
                ),
                "configured_grid_size": configured_grid_size,
                "valid_region_fraction_configured": captures[
                    configured_grid_size
                ][
                    "sampling_metadata"
                ]["valid_region_fraction"],
                "valid_region_fraction_2048": reference[
                    "sampling_metadata"
                ]["valid_region_fraction"],
            }
            for index, field in enumerate(OUTPUT_FIELDS):
                row[f"{field}_difference_in_tolerances"] = float(
                    abs(delta[index]) / tolerance[index]
                )
            rows.append(row)
            del captures
            gc.collect()
    return rows


def plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(row["sweep_id"], row["sampling_method"])].append(row)
    for sweep_id in sorted({row["sweep_id"] for row in rows}):
        figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
        for method in (
            "legacy_left_searchsorted",
            "pixel_area_bilinear_intensity",
        ):
            group = grouped.get((sweep_id, method), [])
            if not group:
                continue
            x = [row["position_mm"] for row in group]
            for index, metric in enumerate(OUTPUT_FIELDS):
                axis = axes.flat[index]
                axis.plot(x, [row[metric] for row in group], label=method)
                axis.set_ylabel(metric)
                axis.grid(alpha=0.25)
        axes.flat[-1].plot(
            [
                row["position_mm"]
                for row in grouped[
                    (sweep_id, "pixel_area_bilinear_intensity")
                ]
            ],
            [
                row["max_tolerance_normalized_delta"]
                for row in grouped[
                    (sweep_id, "pixel_area_bilinear_intensity")
                ]
            ],
            color="tab:green",
        )
        axes.flat[-1].set_ylabel("max adjacent change [tol]")
        axes.flat[-1].set_xlabel("position [mm]")
        axes.flat[-1].grid(alpha=0.25)
        axes.flat[0].legend(fontsize=8)
        figure.suptitle(sweep_id)
        figure.tight_layout()
        figure.savefig(output_dir / "plots" / f"{sweep_id}.png", dpi=150)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite audit: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "plots").mkdir()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    bounds = Bounds.from_config(config)
    base_config = str(
        (REPO_ROOT / config["simulator"]["base_config"]).resolve()
    )
    fixed = default_simulator_fixed(
        base_config,
        grid_size=args.grid_size,
        grid_extent_mm=(
            float(args.grid_extent_mm)
            if args.grid_extent_mm is not None
            else float(config["simulator"]["data_grid_extent_mm"])
        ),
        sensor_resolution=[
            args.sensor_resolution,
            args.sensor_resolution,
        ],
        semantics=config["simulator"]["semantics"],
    )
    setup = build_optical_setup(
        BASE_CONTEXT, BASE_POSITIONS_MM, fixed, base_config
    )
    started = time.perf_counter()
    _, grid_x, grid_y, spacing, at_lens, base_field = prepare_fields(setup)
    interior = sweep_rows(
        setup,
        grid_x,
        grid_y,
        spacing,
        at_lens,
        base_field,
        half_range_mm=float(args.sweep_half_range_mm),
        step_mm=float(args.sweep_step_mm),
    )
    boundary = boundary_rows(
        setup,
        grid_x,
        grid_y,
        base_field,
        step_mm=float(args.sweep_step_mm),
    )
    sweeps = interior + boundary
    summary = continuity_summary(sweeps)
    power = power_rows(config, fixed, base_config, bounds)
    coordinates = coordinate_rows(fixed, base_config, bounds)
    references = reference_rows(
        setup, grid_x, grid_y, spacing, at_lens, base_field
    )
    resolution = resolution_rows(
        config,
        base_config,
        bounds,
        args.sensor_resolution,
        float(fixed["grid_extent_mm"]),
        args.grid_size,
    )
    regime_resolution = regime_resolution_rows(
        config,
        base_config,
        bounds,
        args.sensor_resolution,
        float(fixed["grid_extent_mm"]),
        args.grid_size,
    )
    write_csv(output_dir / "continuity_sweeps.csv", sweeps)
    write_csv(output_dir / "continuity_summary.csv", summary)
    write_csv(output_dir / "power_scaling.csv", power)
    write_csv(output_dir / "coordinate_frames.csv", coordinates)
    write_csv(output_dir / "sampling_reference.csv", references)
    write_csv(output_dir / "resolution_reference.csv", resolution)
    write_csv(
        output_dir / "regime_resolution_reference.csv",
        regime_resolution,
    )
    plots(output_dir, sweeps)

    corrected_summary = [
        row
        for row in summary
        if row["sampling_method"] == "pixel_area_bilinear_intensity"
    ]
    interior_camera = [
        row
        for row in corrected_summary
        if row["sweep_id"]
        in {"interior_camera_x_mm", "interior_camera_y_mm"}
    ]
    boundary_corrected = next(
        row
        for row in corrected_summary
        if row["sweep_id"] == "clipping_boundary_camera_x_mm"
    )
    power_pass = all(
        abs(row["source_amplitude_ratio"] - row["expected_amplitude_ratio"])
        < 2e-6
        and abs(row["raw_image_sum_ratio"] - row["expected_intensity_ratio"])
        < 2e-5
        and abs(row["captured_power_ratio"] - row["expected_intensity_ratio"])
        < 2e-5
        and abs(row["peak_intensity_ratio"] - row["expected_intensity_ratio"])
        < 2e-5
        and row["normalized_image_max_abs_difference"] < 2e-6
        for row in power
    )
    coordinate_pass = max(
        row[f"sensor_image_discrepancy_tolerance_{field}"]
        for row in coordinates
        for field in OUTPUT_FIELDS
    ) < 0.1
    selected_reference_error = max(
        row["max_metric_error_in_tolerances"]
        for row in references
        if row["method"] == "pixel_area_bilinear_intensity"
    )
    configured_resolution_row = next(
        row for row in resolution if row["grid_size"] == args.grid_size
    )
    gates = {
        "legacy_compatibility_requires_separate_bitwise_test": True,
        "continuous_effective_coordinates": all(
            row["index_boundary_transition_count"]
            == row["points"] - 1
            for row in corrected_summary
            if "camera_" in row["sweep_id"]
        ),
        "no_corrected_exact_image_plateaus": all(
            row["exact_image_plateau_count"] == 0
            for row in corrected_summary
        ),
        "no_corrected_exact_metric_plateaus": all(
            row["exact_metric_plateau_count"] == 0
            for row in corrected_summary
        ),
        "interior_camera_below_0p25_tolerance": max(
            row["max_adjacent_tolerance_change"]
            for row in interior_camera
        )
        < 0.25,
        "boundary_below_one_tolerance": (
            boundary_corrected["max_adjacent_tolerance_change"] < 1.0
        ),
        "power_scaling": power_pass,
        "coordinate_transform_below_0p1_tolerance": coordinate_pass,
        "selected_sampling_matches_order9_reference_below_0p05_tolerance": (
            selected_reference_error < 0.05
        ),
        "propagation_resolution_converged_below_0p25_tolerance": (
            configured_resolution_row[
                "max_metric_difference_in_tolerances"
            ]
            < 0.25
        ),
        "propagation_power_converged_below_1pct": (
            configured_resolution_row[
                "captured_power_relative_difference"
            ]
            < 0.01
        ),
        "all_regimes_resolution_converged_below_0p25_tolerance": all(
            row["max_metric_difference_in_tolerances"] < 0.25
            for row in regime_resolution
        ),
        "all_regimes_power_converged_below_1pct": all(
            row["captured_power_relative_difference"] < 0.01
            for row in regime_resolution
        ),
        "finite_outputs": all(
            np.isfinite(
                [
                    row[field]
                    for field in OUTPUT_FIELDS
                    if field in row
                ]
            ).all()
            for row in sweeps
        ),
    }
    gates["acceptance_gate_pass"] = all(gates.values())
    report = {
        "version": "simulator_semantics_v12_acceptance_v1",
        "config": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "grid_size": args.grid_size,
        "grid_extent_mm": fixed["grid_extent_mm"],
        "sensor_resolution": args.sensor_resolution,
        "step_mm": args.sweep_step_mm,
        "elapsed_seconds": time.perf_counter() - started,
        "gates": gates,
        "selected_reference_max_metric_error_in_tolerances": (
            selected_reference_error
        ),
        "configured_to_2048_max_metric_difference_in_tolerances": (
            configured_resolution_row[
                "max_metric_difference_in_tolerances"
            ]
        ),
        "legacy_comparison_baselines": {
            "interior_max_tolerance_jump": 0.8377,
            "boundary_max_tolerance_jump": 1.6948,
            "smoke_peak_max_tolerance_jump": 6.9384,
        },
        "files": sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    (output_dir / "acceptance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
