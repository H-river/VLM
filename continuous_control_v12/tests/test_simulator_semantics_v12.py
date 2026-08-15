from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from continuous_control_v12.contracts import Bounds, OUTPUT_FIELDS
from continuous_control_v12.simulator import (
    CORRECTED_SEMANTICS_VERSION,
    build_optical_setup,
    default_simulator_fixed,
    sample_group_setup,
    simulate_state,
)
from optical_sim.src.simulator import (
    _BACKENDS,
    _extract_sensor_region,
    _extract_sensor_region_continuous,
    apply_thin_lens,
    gaussian_source_field,
)
from legacy.experiments.audits.simulator_three_issue_audit_v1.scripts import (
    audit_core as simulator_audit_core,
)

BASE_CONTEXT = simulator_audit_core.BASE_CONTEXT
BASE_POSITIONS_MM = simulator_audit_core.BASE_POSITIONS_MM
independent_sensor_metrics = simulator_audit_core.independent_sensor_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CONFIG = json.loads(
    (REPO_ROOT / "continuous_control_v12/config_v12.json").read_text()
)
CORRECTED_CONFIG = json.loads(
    (
        REPO_ROOT
        / "continuous_control_v12/config_v12_semantics_v1.json"
    ).read_text()
)
DATA_QUALITY_CONFIG = json.loads(
    (
        REPO_ROOT
        / "continuous_control_v12/config_v12_semantics_v2.json"
    ).read_text()
)
BASE = str((REPO_ROOT / "optical_sim/configs/base_config.yaml").resolve())


@pytest.fixture(scope="module")
def bounds() -> Bounds:
    return Bounds.from_config(CORRECTED_CONFIG)


@pytest.fixture(scope="module")
def corrected_fixed() -> dict:
    return default_simulator_fixed(
        BASE,
        grid_size=256,
        grid_extent_mm=8.0,
        sensor_resolution=[128, 128],
        semantics=CORRECTED_CONFIG["simulator"]["semantics"],
    )


@pytest.fixture(scope="module")
def corrected_capture(corrected_fixed: dict, bounds: Bounds) -> dict:
    return simulate_state(
        BASE_CONTEXT,
        BASE_POSITIONS_MM,
        corrected_fixed,
        BASE,
        bounds,
    )


def test_data_quality_setup_retries_are_opt_in_and_deterministic() -> None:
    legacy_v12_fixed = default_simulator_fixed(
        BASE,
        grid_size=128,
        sensor_resolution=[64, 64],
        semantics=CORRECTED_CONFIG["simulator"]["semantics"],
    )
    assert "minimum_initial_captured_power_fraction" not in legacy_v12_fixed
    assert "maximum_setup_resample_attempts" not in legacy_v12_fixed

    quality_fixed = default_simulator_fixed(
        BASE,
        grid_size=128,
        sensor_resolution=[64, 64],
        semantics=DATA_QUALITY_CONFIG["simulator"]["semantics"],
    )
    assert quality_fixed["minimum_initial_captured_power_fraction"] == 0.01
    assert quality_fixed["maximum_setup_resample_attempts"] == 16

    arguments = (
        "tolerance_boundary",
        "v12_train_000055",
        int(DATA_QUALITY_CONFIG["seed"]),
        quality_fixed,
        Bounds.from_config(DATA_QUALITY_CONFIG),
    )
    first = sample_group_setup(*arguments, setup_attempt=0)
    first_replay = sample_group_setup(*arguments, setup_attempt=0)
    retry = sample_group_setup(*arguments, setup_attempt=1)
    retry_replay = sample_group_setup(*arguments, setup_attempt=1)
    assert first == first_replay
    assert retry == retry_replay
    assert retry != first


def test_legacy_128_outputs_remain_bitwise_unchanged() -> None:
    fixed = default_simulator_fixed(
        BASE, grid_size=128, sensor_resolution=[128, 128]
    )
    setup = build_optical_setup(BASE_CONTEXT, BASE_POSITIONS_MM, fixed, BASE)
    source, grid_x, grid_y, spacing = gaussian_source_field(setup)
    propagate = _BACKENDS[setup.propagation_backend]
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
    intensity, _, _ = _extract_sensor_region(
        field, grid_x, grid_y, setup
    )
    expected = {
        "source": "f12df84fb780df7e0bfc2fa7bc2e02719be73f8ae61e91a0a58498c820ed8927",
        "at_lens": "957862c738a05ac973dcec730e9a644005acd7a3101f9be10bc8bc3562ce2b75",
        "after_lens": "2d987f25661a16a25e1f167ef5b4385c3338e193408eb3e74e2194855945cb20",
        "field": "494996cb2bfdba5af692e59916cc4d2203c29b10ce70952cf5a0f509222e9486",
        "intensity": "d7a98a57304dc6cc0c0e5c09e67b9c1eea75b63993886cddb38bb5f0eccec272",
    }
    actual = {
        name: hashlib.sha256(array.tobytes()).hexdigest()
        for name, array in (
            ("source", source),
            ("at_lens", at_lens),
            ("after_lens", after_lens),
            ("field", field),
            ("intensity", intensity),
        )
    }
    assert actual == expected


def test_corrected_no_op_is_bitwise_repeatable(
    corrected_fixed: dict, bounds: Bounds
) -> None:
    left = simulate_state(
        BASE_CONTEXT, BASE_POSITIONS_MM, corrected_fixed, BASE, bounds
    )
    right = simulate_state(
        BASE_CONTEXT, BASE_POSITIONS_MM, corrected_fixed, BASE, bounds
    )
    assert np.array_equal(left["image_raw"], right["image_raw"])
    assert np.array_equal(
        left["image_normalized"], right["image_normalized"]
    )
    assert np.array_equal(
        left["valid_region_mask"], right["valid_region_mask"]
    )
    assert left["metrics"] == right["metrics"]
    assert left["sampling_metadata"] == right["sampling_metadata"]


def test_corrected_camera_subpixel_sweep_has_no_exact_plateaus(
    corrected_fixed: dict, bounds: Bounds
) -> None:
    images = []
    metrics = []
    fractional = []
    for delta_mm in np.linspace(-0.004, 0.004, 17):
        positions = dict(BASE_POSITIONS_MM)
        positions["camera_x_mm"] += float(delta_mm)
        capture = simulate_state(
            BASE_CONTEXT, positions, corrected_fixed, BASE, bounds
        )
        images.append(capture["image_raw"])
        metrics.append(tuple(capture["metrics"].values()))
        fractional.append(
            capture["sampling_metadata"][
                "center_fractional_grid_index_x"
            ]
        )
    assert all(
        not np.array_equal(images[index - 1], images[index])
        for index in range(1, len(images))
    )
    assert all(
        metrics[index - 1] != metrics[index]
        for index in range(1, len(metrics))
    )
    assert np.allclose(
        np.diff(fractional),
        np.diff(fractional)[0],
        rtol=1e-11,
        atol=1e-13,
    )
    assert np.all(np.diff(fractional) > 0.0)


def test_corrected_lens_sweeps_are_continuous_and_axis_signs_are_explicit(
    corrected_fixed: dict, bounds: Bounds
) -> None:
    base = simulate_state(
        BASE_CONTEXT, BASE_POSITIONS_MM, corrected_fixed, BASE, bounds
    )
    for position_field, centroid_field in (
        ("lens_x_mm", "centroid_x_px"),
        ("lens_y_mm", "centroid_y_px"),
    ):
        captures = []
        for delta_mm in (-0.003, 0.0, 0.003):
            positions = dict(BASE_POSITIONS_MM)
            positions[position_field] += delta_mm
            captures.append(
                simulate_state(
                    BASE_CONTEXT, positions, corrected_fixed, BASE, bounds
                )
            )
        assert all(
            not np.array_equal(
                captures[index - 1]["image_raw"],
                captures[index]["image_raw"],
            )
            for index in range(1, len(captures))
        )
        values = [
            capture["metrics"][centroid_field] for capture in captures
        ]
        assert values[0] < values[1] < values[2]

    for position_field, centroid_field in (
        ("camera_x_mm", "centroid_x_px"),
        ("camera_y_mm", "centroid_y_px"),
    ):
        positions = dict(BASE_POSITIONS_MM)
        positions[position_field] += 0.003
        moved = simulate_state(
            BASE_CONTEXT, positions, corrected_fixed, BASE, bounds
        )
        assert (
            moved["metrics_sensor_frame"][centroid_field]
            < base["metrics_sensor_frame"][centroid_field]
        )


def test_corrected_coordinate_transform_matches_sensor_image(
    corrected_capture: dict,
) -> None:
    independent = independent_sensor_metrics(
        corrected_capture["image_raw"],
        BASE_CONTEXT["pixel_size_um"] * 1e-6,
    )
    for field in OUTPUT_FIELDS:
        assert corrected_capture["metrics_sensor_frame"][field] == pytest.approx(
            independent[field], abs=2e-6
        )
    semantics = corrected_capture["coordinate_semantics"]
    assert semantics["target_control_frame"] == (
        "lab_frame_legacy_pseudo_pixels"
    )
    assert semantics["axis_convention"] == (
        "axis_0_plus_y_lab_axis_1_plus_x_lab"
    )


def test_corrected_power_scales_field_and_raw_intensity(
    corrected_fixed: dict, bounds: Bounds
) -> None:
    captures = {}
    for power in (0.25, 1.0, 4.0):
        context = dict(BASE_CONTEXT)
        context["power_w"] = power
        captures[power] = simulate_state(
            context, BASE_POSITIONS_MM, corrected_fixed, BASE, bounds
        )
        assert captures[power]["auxiliary"][
            "source_integrated_power_w"
        ] == pytest.approx(power, rel=2e-12, abs=1e-12)
    reference = captures[1.0]
    for power in (0.25, 4.0):
        capture = captures[power]
        assert capture["auxiliary"]["source_amplitude_scale"] / reference[
            "auxiliary"
        ]["source_amplitude_scale"] == pytest.approx(np.sqrt(power), rel=2e-12)
        assert capture["image_raw"] == pytest.approx(
            reference["image_raw"] * power, rel=3e-6, abs=3e-5
        )
        assert capture["image_normalized"] == pytest.approx(
            reference["image_normalized"], rel=3e-6, abs=3e-7
        )
        assert capture["auxiliary"]["captured_power_w"] / reference[
            "auxiliary"
        ]["captured_power_w"] == pytest.approx(power, rel=3e-7)
        assert capture["metrics"]["peak_intensity"] / reference["metrics"][
            "peak_intensity"
        ] == pytest.approx(power, rel=3e-7)
        for field in OUTPUT_FIELDS[:4]:
            assert capture["metrics"][field] == pytest.approx(
                reference["metrics"][field], abs=2e-10
            )


def test_peak_is_maximum_of_raw_image(corrected_capture: dict) -> None:
    assert corrected_capture["metrics"]["peak_intensity"] == pytest.approx(
        float(corrected_capture["image_raw"].max()), rel=2e-7
    )
    assert corrected_capture["auxiliary"]["peak_intensity_abs"] == (
        corrected_capture["metrics"]["peak_intensity"]
    )


def test_selected_quadrature_agrees_with_oversampled_reference(
    corrected_fixed: dict,
) -> None:
    setup = build_optical_setup(
        BASE_CONTEXT, BASE_POSITIONS_MM, corrected_fixed, BASE
    )
    source, grid_x, grid_y, spacing = gaussian_source_field(setup)
    source_power = float(np.square(np.abs(source)).sum() * spacing**2)
    source *= np.sqrt(BASE_CONTEXT["power_w"] / source_power)
    propagate = _BACKENDS[setup.propagation_backend]
    at_lens = propagate(
        source, spacing, setup.laser_to_lens, setup.source.wavelength
    )
    field = propagate(
        apply_thin_lens(at_lens, grid_x, grid_y, setup),
        spacing,
        setup.effective_camera_distance,
        setup.source.wavelength,
    )
    selected, sensor_x, sensor_y, _, _ = _extract_sensor_region_continuous(
        field,
        grid_x,
        grid_y,
        setup,
        method="pixel_area_bilinear_intensity",
        quadrature_order=3,
    )
    reference, _, _, _, _ = _extract_sensor_region_continuous(
        field,
        grid_x,
        grid_y,
        setup,
        method="pixel_area_bilinear_intensity",
        quadrature_order=9,
    )
    relative_l1 = float(
        np.abs(selected - reference).sum()
        / max(float(np.abs(reference).sum()), 1e-30)
    )
    assert relative_l1 < 2e-4
    selected_metrics = independent_sensor_metrics(
        selected, setup.sensor.pixel_pitch
    )
    reference_metrics = independent_sensor_metrics(
        reference, setup.sensor.pixel_pitch
    )
    tolerances = (1.0, 1.0, 2.0, 2.0, 0.05 * reference_metrics["peak_intensity"])
    for field_name, tolerance in zip(
        OUTPUT_FIELDS, tolerances, strict=True
    ):
        assert (
            abs(selected_metrics[field_name] - reference_metrics[field_name])
            / max(float(tolerance), 1e-30)
            < 0.02
        )
    assert sensor_x.shape == selected.shape
    assert sensor_y.shape == selected.shape


def test_zero_padding_and_valid_region_mask_are_explicit(
    corrected_fixed: dict,
) -> None:
    setup = build_optical_setup(
        BASE_CONTEXT, BASE_POSITIONS_MM, corrected_fixed, BASE
    )
    moved = copy.deepcopy(setup)
    moved.camera.x_offset = setup.grid_extent + 1e-3
    source, grid_x, grid_y, _ = gaussian_source_field(moved)
    intensity, _, _, valid, metadata = _extract_sensor_region_continuous(
        source,
        grid_x,
        grid_y,
        moved,
        method="pixel_area_bilinear_intensity",
        quadrature_order=3,
    )
    assert not valid.all()
    assert np.any(intensity == 0.0)
    assert metadata["outside_propagated_field_rule"] == "zero_padding"
    assert metadata["valid_region_fraction"] < 1.0


def test_corrected_semantics_are_explicit(corrected_fixed: dict) -> None:
    assert (
        corrected_fixed["simulator_semantics_version"]
        == CORRECTED_SEMANTICS_VERSION
    )
    assert (
        corrected_fixed["sensor_sampling_method"]
        == "pixel_area_bilinear_intensity"
    )
    assert corrected_fixed["power_semantics"] == (
        "source_integrated_optical_power_w"
    )
    assert corrected_fixed["metrics_frame"] == (
        "lab_frame_legacy_pseudo_pixels"
    )
