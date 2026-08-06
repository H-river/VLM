#!/usr/bin/env python3
"""Run the deterministic three-issue optical-simulator audit."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audits.simulator_three_issue_audit_v1.scripts.audit_core import (
    BASE_CONTEXT,
    BASE_POSITIONS_MM,
    build_setup,
    capture_from_field,
    copy_setup_with_position,
    current_bounds,
    exact_array_hash,
    guarded_output_dir,
    interpolation_spot_checks,
    one_sweep_row,
    prepare_source_and_lens,
    propagate_from_lens,
    read_nonprotected_power_values,
    source_and_propagated_quantities,
)
from continuous_control_v12.contracts import (
    OUTPUT_FIELDS,
    action_dict,
    position_dict,
    project_action,
)
from continuous_control_v12.simulator import default_simulator_fixed


AUDIT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = AUDIT_ROOT / "data"
DEFAULT_PLOT_DIR = AUDIT_ROOT / "plots"
BASE_CONFIG = REPO_ROOT / "optical_sim/configs/base_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--sensor-resolution", type=int, default=1024)
    parser.add_argument("--sweep-half-range-mm", type=float, default=0.09)
    parser.add_argument("--sweep-step-mm", type=float, default=0.0015)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main_sweeps(
    base_setup: Any,
    source: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    spacing: float,
    at_lens: np.ndarray,
    base_camera_field: np.ndarray,
    half_range_mm: float,
    step_mm: float,
) -> list[dict[str, Any]]:
    points = int(round(2.0 * half_range_mm / step_mm)) + 1
    offsets = np.linspace(-half_range_mm, half_range_mm, points)
    methods = ("production_searchsorted", "bilinear_intensity")
    rows: list[dict[str, Any]] = []
    del source
    for position_field in (
        "camera_x_mm",
        "camera_y_mm",
        "lens_x_mm",
        "lens_y_mm",
    ):
        reference = float(BASE_POSITIONS_MM[position_field])
        previous: dict[str, Any | None] = {method: None for method in methods}
        signatures: dict[str, str | None] = {method: None for method in methods}
        for offset in offsets:
            requested = reference + float(offset)
            setup = copy_setup_with_position(base_setup, position_field, requested)
            if position_field.startswith("lens"):
                _, field_at_camera = propagate_from_lens(
                    setup,
                    at_lens,
                    grid_x,
                    grid_y,
                    spacing,
                )
            else:
                field_at_camera = base_camera_field
            for method in methods:
                capture = capture_from_field(
                    field_at_camera,
                    grid_x,
                    grid_y,
                    setup,
                    method,
                )
                row = one_sweep_row(
                    sweep_id=f"main_{position_field}",
                    position_field=position_field,
                    requested_position_mm=requested,
                    effective_position_mm=requested,
                    reference_position_mm=reference,
                    step_mm=step_mm,
                    setup=setup,
                    grid_x=grid_x,
                    grid_y=grid_y,
                    method=method,
                    capture=capture,
                    previous_capture=previous[method],
                    previous_index_signature=signatures[method],
                )
                rows.append(row)
                previous[method] = capture
                signatures[method] = str(row["selected_index_signature"])
    return rows


def camera_boundary_sweep(
    base_setup: Any,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    base_camera_field: np.ndarray,
) -> list[dict[str, Any]]:
    """Sweep through expected beam/sensor clipping near +x sensor edge."""

    values = np.linspace(2.65, 2.95, 81)
    step = float(values[1] - values[0])
    methods = ("production_searchsorted", "bilinear_intensity")
    previous: dict[str, Any | None] = {method: None for method in methods}
    signatures: dict[str, str | None] = {method: None for method in methods}
    rows = []
    for value in values:
        setup = copy_setup_with_position(base_setup, "camera_x_mm", float(value))
        for method in methods:
            capture = capture_from_field(
                base_camera_field,
                grid_x,
                grid_y,
                setup,
                method,
            )
            row = one_sweep_row(
                sweep_id="sensor_clipping_boundary_camera_x_mm",
                position_field="camera_x_mm",
                requested_position_mm=float(value),
                effective_position_mm=float(value),
                reference_position_mm=float(values[0]),
                step_mm=step,
                setup=setup,
                grid_x=grid_x,
                grid_y=grid_y,
                method=method,
                capture=capture,
                previous_capture=previous[method],
                previous_index_signature=signatures[method],
            )
            rows.append(row)
            previous[method] = capture
            signatures[method] = str(row["selected_index_signature"])
    return rows


def actuator_limit_sweep(
    context: dict[str, float],
    simulator_fixed: dict[str, Any],
    base_setup: Any,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    base_camera_field: np.ndarray,
) -> list[dict[str, Any]]:
    """Separate expected plateau from v12 projection at +3 mm."""

    del context, simulator_fixed
    bounds = current_bounds()
    current = np.asarray([0.0, 0.0, 2.99, 0.0], dtype=np.float64)
    requests = np.linspace(-0.005, 0.02, 51)
    previous = None
    signature = None
    rows = []
    for request in requests:
        action = np.asarray([0.0, 0.0, float(request), 0.0], dtype=np.float64)
        projected = project_action(current, action, bounds)
        effective_positions = current + projected
        setup = copy.deepcopy(base_setup)
        setup.camera.x_offset = float(effective_positions[2]) * 1e-3
        setup.camera.y_offset = 0.0
        capture = capture_from_field(
            base_camera_field,
            grid_x,
            grid_y,
            setup,
            "production_searchsorted",
        )
        row = one_sweep_row(
            sweep_id="v12_absolute_actuator_limit_camera_x_mm",
            position_field="camera_x_mm",
            requested_position_mm=float(current[2] + request),
            effective_position_mm=float(effective_positions[2]),
            reference_position_mm=float(current[2]),
            step_mm=float(requests[1] - requests[0]),
            setup=setup,
            grid_x=grid_x,
            grid_y=grid_y,
            method="production_searchsorted",
            capture=capture,
            previous_capture=previous,
            previous_index_signature=signature,
        )
        row["requested_action_json"] = json.dumps(action_dict(action))
        row["effective_action_json"] = json.dumps(action_dict(projected))
        rows.append(row)
        previous = capture
        signature = str(row["selected_index_signature"])
    return rows


def plateau_run_length(flags: list[bool]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest + 1 if longest else 1


def equality_run_length(values: list[Any]) -> int:
    if not values:
        return 0
    longest = current = 1
    for previous, value in zip(values, values[1:], strict=False):
        if value == previous:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def derivative_diagnostics(
    rows: list[dict[str, Any]],
    field: str,
) -> tuple[int, int]:
    values = np.asarray(
        [float(row[f"derivative_{field}_per_mm"]) for row in rows[1:]],
        dtype=np.float64,
    )
    nonzero = values[np.abs(values) > 1e-12]
    sign_changes = (
        int(np.sum(np.sign(nonzero[1:]) != np.sign(nonzero[:-1])))
        if len(nonzero) > 1
        else 0
    )
    median = float(np.median(np.abs(nonzero))) if len(nonzero) else 0.0
    spikes = int(np.sum(np.abs(nonzero) > 10.0 * median)) if median > 0 else 0
    return sign_changes, spikes


def continuity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sweep_id"]), str(row["sampling_method"]))].append(row)
    output = []
    for (sweep_id, method), group in grouped.items():
        summary: dict[str, Any] = {
            "sweep_id": sweep_id,
            "sampling_method": method,
            "points": len(group),
            "step_mm": float(group[0]["step_mm"]),
            "simulation_grid_pitch_mm": float(
                group[0]["simulation_grid_pitch_mm"]
            ),
            "declared_sensor_pitch_mm": float(
                group[0]["declared_sensor_pitch_mm"]
            ),
            "exact_image_plateau_transitions": int(
                sum(as_bool(row["exact_image_plateau"]) for row in group)
            ),
            "longest_exact_image_plateau_points": plateau_run_length(
                [as_bool(row["exact_image_plateau"]) for row in group[1:]]
            ),
            "exact_metric_plateau_transitions": int(
                sum(as_bool(row["exact_metric_plateau"]) for row in group)
            ),
            "index_boundary_transitions": int(
                sum(as_bool(row["index_boundary_crossed"]) for row in group)
            ),
            "selected_center_index_change_transitions": int(
                sum(
                    int(group[index]["selected_center_index"])
                    != int(group[index - 1]["selected_center_index"])
                    for index in range(1, len(group))
                )
            ),
            "longest_selected_center_index_plateau_points": equality_run_length(
                [int(row["selected_center_index"]) for row in group]
            ),
            "largest_raw_image_relative_l1_change": max(
                float(row["raw_image_relative_l1_change"]) for row in group
            ),
            "largest_normalized_image_mean_abs_change": max(
                float(row["normalized_image_mean_abs_change"]) for row in group
            ),
            "largest_max_tolerance_normalized_delta": max(
                float(row["max_tolerance_normalized_delta"]) for row in group
            ),
        }
        for field in OUTPUT_FIELDS:
            normalized_key = f"tolerance_normalized_delta_{field}"
            derivative_key = f"derivative_{field}_per_mm"
            sign_changes, spikes = derivative_diagnostics(group, field)
            summary[f"largest_jump_{field}"] = max(
                abs(float(row[f"delta_{field}"])) for row in group
            )
            summary[f"largest_tolerance_normalized_jump_{field}"] = max(
                float(row[normalized_key]) for row in group
            )
            summary[f"largest_abs_derivative_{field}_per_mm"] = max(
                abs(float(row[derivative_key])) for row in group
            )
            plateau_flags = [
                abs(float(row[f"delta_{field}"])) <= 1e-14
                for row in group[1:]
            ]
            summary[f"constant_transition_count_{field}"] = int(
                sum(plateau_flags)
            )
            summary[f"longest_constant_plateau_points_{field}"] = (
                plateau_run_length(plateau_flags)
            )
            summary[f"derivative_sign_changes_{field}"] = sign_changes
            summary[f"derivative_spikes_{field}"] = spikes
        output.append(summary)
    return output


def coordinate_rows(
    base_setup: Any,
    at_lens: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    spacing: float,
    base_camera_field: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run no-op plus four pure interventions and compare frames."""

    cases = [("no_op", None)]
    cases.extend(
        [
            ("camera_x_plus_0p02_mm", "camera_x_mm"),
            ("camera_y_plus_0p02_mm", "camera_y_mm"),
            ("lens_x_plus_0p02_mm", "lens_x_mm"),
            ("lens_y_plus_0p02_mm", "lens_y_mm"),
        ]
    )
    methods = ("production_searchsorted", "bilinear_intensity")
    captures: dict[tuple[str, str], dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    setups_and_fields = []
    for case_name, field in cases:
        setup = copy.deepcopy(base_setup)
        if field is not None:
            reference = float(BASE_POSITIONS_MM[field])
            setup = copy_setup_with_position(base_setup, field, reference + 0.02)
        if field is not None and field.startswith("lens"):
            _, camera_field = propagate_from_lens(
                setup,
                at_lens,
                grid_x,
                grid_y,
                spacing,
            )
        else:
            camera_field = base_camera_field
        setups_and_fields.append((case_name, field, setup, camera_field))
        for method in methods:
            capture = capture_from_field(
                camera_field,
                grid_x,
                grid_y,
                setup,
                method,
            )
            captures[(method, case_name)] = capture
            stored = capture["stored_metrics"]
            sensor = capture["sensor_frame_metrics"]
            image = capture["image_metrics"]
            pitch = float(setup.sensor.pixel_pitch)
            for field_name in OUTPUT_FIELDS:
                physical_discrepancy = (
                    (float(sensor[field_name]) - float(image[field_name])) * pitch
                    if field_name != "peak_intensity"
                    else float(sensor[field_name]) - float(image[field_name])
                )
                tolerance = (
                    1.0
                    if field_name.startswith("centroid")
                    else 2.0
                    if field_name.startswith("sigma")
                    else max(0.05 * abs(float(stored["peak_intensity"])), 1e-6)
                )
                comparison_rows.append(
                    {
                        "sampling_method": method,
                        "case": case_name,
                        "quantity": field_name,
                        "stored_legacy_value": float(stored[field_name]),
                        "stored_transformed_sensor_value": float(sensor[field_name]),
                        "image_derived_value": float(image[field_name]),
                        "sensor_minus_image": float(
                            sensor[field_name] - image[field_name]
                        ),
                        "physical_discrepancy_m_or_intensity": physical_discrepancy,
                        "tolerance": tolerance,
                        "absolute_discrepancy_in_tolerance_units": float(
                            abs(float(sensor[field_name]) - float(image[field_name]))
                            / tolerance
                        ),
                    }
                )
    intervention_rows = []
    for method in methods:
        baseline = captures[(method, "no_op")]
        for case_name, field, _, _ in setups_and_fields:
            capture = captures[(method, case_name)]
            row: dict[str, Any] = {
                "sampling_method": method,
                "case": case_name,
                "changed_position_field": field or "none",
                "position_delta_mm": 0.0 if field is None else 0.02,
            }
            for representation, key in (
                ("stored", "stored_metrics"),
                ("sensor_frame", "sensor_frame_metrics"),
                ("image_derived", "image_metrics"),
            ):
                for metric in OUTPUT_FIELDS:
                    row[f"delta_{representation}_{metric}"] = float(
                        capture[key][metric] - baseline[key][metric]
                    )
            intervention_rows.append(row)
    return comparison_rows, intervention_rows


def frame_table() -> list[dict[str, Any]]:
    return [
        {
            "quantity": "centroid_x_px",
            "current_frame": "lab-frame pseudo-pixel",
            "image_derived_frame": "sensor-array pixel centres",
            "unit": "px",
            "transform_available": "subtract camera_x_m/pitch_m",
            "consistent": "only after transform; residual W/(W-1) scale",
        },
        {
            "quantity": "centroid_y_px",
            "current_frame": "lab-frame pseudo-pixel",
            "image_derived_frame": "sensor-array pixel centres",
            "unit": "px",
            "transform_available": "subtract camera_y_m/pitch_m",
            "consistent": "only after transform; residual H/(H-1) scale",
        },
        {
            "quantity": "sigma_x_px",
            "current_frame": "lab-derived width divided by declared pitch",
            "image_derived_frame": "sensor-array pixel centres",
            "unit": "px",
            "transform_available": "no translation; scale by (W-1)/W",
            "consistent": "yes after coordinate-step correction",
        },
        {
            "quantity": "sigma_y_px",
            "current_frame": "lab-derived width divided by declared pitch",
            "image_derived_frame": "sensor-array pixel centres",
            "unit": "px",
            "transform_available": "no translation; scale by (H-1)/H",
            "consistent": "yes after coordinate-step correction",
        },
        {
            "quantity": "peak_intensity",
            "current_frame": "sampled sensor image",
            "image_derived_frame": "same sampled sensor image",
            "unit": "simulator intensity units",
            "transform_available": "identity",
            "consistent": "yes",
        },
    ]


def power_rows(
    context: dict[str, float],
    positions: dict[str, float],
    simulator_fixed: dict[str, Any],
) -> list[dict[str, Any]]:
    raw = []
    for power in (0.25, 0.5, 1.0, 2.0, 4.0):
        local_context = dict(context)
        local_context["power_w"] = power
        setup = build_setup(
            local_context,
            positions,
            simulator_fixed,
            str(BASE_CONFIG),
        )
        quantities = source_and_propagated_quantities(setup)
        capture = quantities["capture"]
        row = {
            "power_w": power,
            "expected_field_amplitude_ratio_if_active": float(np.sqrt(power)),
            "expected_intensity_ratio_if_active": power,
            "source_peak_amplitude": quantities["source_peak_amplitude"],
            "source_integrated_intensity": quantities[
                "source_integrated_intensity"
            ],
            "camera_grid_integrated_intensity": quantities[
                "camera_grid_integrated_intensity"
            ],
            "captured_power": capture["captured_power"],
            "unnormalized_sensor_sum": float(capture["intensity"].sum()),
            "unnormalized_sensor_max": float(capture["intensity"].max()),
            "normalized_sensor_sum": float(
                capture["intensity"].sum()
                / max(float(capture["intensity"].max()), 1e-30)
            ),
            "source_hash": exact_array_hash(quantities["source"]),
            "camera_field_hash": exact_array_hash(quantities["field_at_camera"]),
            "sensor_image_hash": exact_array_hash(capture["intensity"]),
        }
        for field in OUTPUT_FIELDS:
            row[field] = float(capture["stored_metrics"][field])
        raw.append(row)
    baseline = next(row for row in raw if row["power_w"] == 1.0)
    ratio_fields = (
        "source_peak_amplitude",
        "source_integrated_intensity",
        "camera_grid_integrated_intensity",
        "captured_power",
        "unnormalized_sensor_sum",
        "unnormalized_sensor_max",
        "normalized_sensor_sum",
        *OUTPUT_FIELDS,
    )
    for row in raw:
        for field in ratio_fields:
            denominator = max(abs(float(baseline[field])), 1e-300)
            row[f"ratio_to_1x_{field}"] = float(row[field]) / denominator
        row["exact_source_equal_to_1x"] = (
            row["source_hash"] == baseline["source_hash"]
        )
        row["exact_camera_field_equal_to_1x"] = (
            row["camera_field_hash"] == baseline["camera_field_hash"]
        )
        row["exact_sensor_image_equal_to_1x"] = (
            row["sensor_image_hash"] == baseline["sensor_image_hash"]
        )
    return raw


def dataset_power_rows() -> list[dict[str, Any]]:
    paths = [
        Path(
            "/home/jiamo/VLM_data/specialist_rebuild_v2/grids/train.jsonl"
        ),
        Path(
            "/home/jiamo/VLM_data/control_rebuild_v4_quickcheck/grids/train.jsonl"
        ),
        Path(
            "/home/jiamo/VLM_data/control_rebuild_v5_numerical/grids/train.jsonl"
        ),
    ]
    rows = []
    for path in paths:
        if not path.is_file():
            rows.append(
                {
                    "dataset_path": str(path),
                    "exists": False,
                    "rows_with_power": 0,
                    "distinct_power_values": 0,
                    "minimum_power_w": "",
                    "maximum_power_w": "",
                    "power_in_setup_features": True,
                }
            )
            continue
        values = read_nonprotected_power_values(path)
        rows.append(
            {
                "dataset_path": str(path),
                "exists": True,
                "rows_with_power": int(len(values)),
                "distinct_power_values": int(np.unique(values).size),
                "minimum_power_w": float(values.min()),
                "maximum_power_w": float(values.max()),
                "power_in_setup_features": True,
            }
        )
    return rows


def make_plots(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["sweep_id"]).startswith("main_"):
            grouped[str(row["sweep_id"])].append(row)
    colors = {
        "production_searchsorted": "#b22222",
        "bilinear_intensity": "#1f77b4",
    }
    for sweep_id, group in grouped.items():
        fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
        axes_flat = axes.flat
        for metric_index, metric in enumerate(OUTPUT_FIELDS):
            ax = axes_flat[metric_index]
            for method in colors:
                selected = [
                    row for row in group if row["sampling_method"] == method
                ]
                ax.plot(
                    [float(row["displacement_from_reference_mm"]) for row in selected],
                    [float(row[f"stored_{metric}"]) for row in selected],
                    label=method,
                    color=colors[method],
                    linewidth=1.2,
                )
            ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
        change_axis = axes_flat[5]
        for method in colors:
            selected = [row for row in group if row["sampling_method"] == method]
            change_axis.plot(
                [float(row["displacement_from_reference_mm"]) for row in selected],
                [float(row["max_tolerance_normalized_delta"]) for row in selected],
                label=method,
                color=colors[method],
                linewidth=1.2,
            )
        change_axis.set_ylabel("adjacent max change [tolerance]")
        change_axis.set_yscale("symlog", linthresh=1e-8)
        change_axis.grid(alpha=0.25)
        for ax in axes[-1, :]:
            ax.set_xlabel("physical displacement [mm]")
        axes_flat[0].legend(fontsize=8)
        fig.suptitle(sweep_id)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{sweep_id}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for method in colors:
            selected = [row for row in group if row["sampling_method"] == method]
            ax.plot(
                [float(row["displacement_from_reference_mm"]) for row in selected],
                [
                    float(row["normalized_image_mean_abs_change"])
                    for row in selected
                ],
                label=method,
                color=colors[method],
            )
        ax.set_xlabel("physical displacement [mm]")
        ax.set_ylabel("adjacent normalized-image mean |difference|")
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{sweep_id}_adjacent_change.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = guarded_output_dir(args.data_dir, AUDIT_ROOT)
    plot_dir = guarded_output_dir(args.plot_dir, AUDIT_ROOT)
    data_dir.mkdir(parents=True, exist_ok=True)
    context = dict(BASE_CONTEXT)
    positions = dict(BASE_POSITIONS_MM)
    simulator_fixed = default_simulator_fixed(
        str(BASE_CONFIG),
        grid_size=int(args.grid_size),
        sensor_resolution=[
            int(args.sensor_resolution),
            int(args.sensor_resolution),
        ],
    )
    base_setup = build_setup(
        context,
        positions,
        simulator_fixed,
        str(BASE_CONFIG),
    )
    source, grid_x, grid_y, spacing, at_lens = prepare_source_and_lens(base_setup)
    _, base_camera_field = propagate_from_lens(
        base_setup,
        at_lens,
        grid_x,
        grid_y,
        spacing,
    )

    sweep_rows = main_sweeps(
        base_setup,
        source,
        grid_x,
        grid_y,
        spacing,
        at_lens,
        base_camera_field,
        float(args.sweep_half_range_mm),
        float(args.sweep_step_mm),
    )
    boundary_rows = camera_boundary_sweep(
        base_setup,
        grid_x,
        grid_y,
        base_camera_field,
    )
    actuator_rows = actuator_limit_sweep(
        context,
        simulator_fixed,
        base_setup,
        grid_x,
        grid_y,
        base_camera_field,
    )
    all_continuity_rows = [*sweep_rows, *boundary_rows]
    summary_rows = continuity_summary(all_continuity_rows)
    coordinate_comparison, interventions = coordinate_rows(
        base_setup,
        at_lens,
        grid_x,
        grid_y,
        spacing,
        base_camera_field,
    )
    power = power_rows(context, positions, simulator_fixed)
    interpolation = interpolation_spot_checks(
        base_camera_field,
        grid_x,
        grid_y,
        base_setup,
    )

    write_csv(data_dir / "continuity_sweeps.csv", all_continuity_rows)
    write_csv(data_dir / "continuity_summary.csv", summary_rows)
    write_csv(data_dir / "actuator_limit_boundary.csv", actuator_rows)
    write_csv(data_dir / "sensor_metric_comparison.csv", coordinate_comparison)
    write_csv(data_dir / "coordinate_interventions.csv", interventions)
    write_csv(data_dir / "coordinate_frame_table.csv", frame_table())
    write_csv(data_dir / "power_causal_scaling.csv", power)
    write_csv(data_dir / "dataset_power_variation.csv", dataset_power_rows())
    write_csv(data_dir / "interpolation_semantics.csv", interpolation)
    make_plots(all_continuity_rows, plot_dir)

    run_metadata = {
        "version": "simulator_three_issue_audit_v1",
        "base_config": str(BASE_CONFIG),
        "base_context": context,
        "base_positions_mm": positions,
        "simulator_fixed": simulator_fixed,
        "sweep_half_range_mm": float(args.sweep_half_range_mm),
        "sweep_step_mm": float(args.sweep_step_mm),
        "simulation_grid_pitch_mm": float(spacing * 1e3),
        "declared_sensor_pitch_mm": float(base_setup.sensor.pixel_pitch * 1e3),
        "constructed_sensor_x_step_mm": float(
            (base_setup.sensor.resolution[1] / (base_setup.sensor.resolution[1] - 1))
            * base_setup.sensor.pixel_pitch
            * 1e3
        ),
        "source_seed": "not applicable; deterministic simulator contains no RNG",
        "production_code_modified": False,
        "protected_data_accessed": False,
        "files": sorted(
            str(path.relative_to(AUDIT_ROOT))
            for path in [*data_dir.glob("*.csv"), *plot_dir.glob("*.png")]
        ),
    }
    (data_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
