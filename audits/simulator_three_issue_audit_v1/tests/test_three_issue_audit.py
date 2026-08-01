"""Deterministic diagnostic expectations for current and desired semantics."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from audits.simulator_three_issue_audit_v1.scripts.audit_core import (
    BASE_CONTEXT,
    BASE_POSITIONS_MM,
    build_setup,
    capture_from_field,
    copy_setup_with_position,
    exact_array_hash,
    independent_sensor_metrics,
    prepare_source_and_lens,
    propagate_from_lens,
    sampling_metadata,
    source_and_propagated_quantities,
)
from continuous_control_v12.contracts import stable_seed
from continuous_control_v12.simulator import default_simulator_fixed


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = REPO_ROOT / "optical_sim/configs/base_config.yaml"


@pytest.fixture(scope="module")
def deterministic_fixture() -> dict[str, object]:
    fixed = default_simulator_fixed(
        str(BASE_CONFIG),
        grid_size=128,
        sensor_resolution=[128, 128],
    )
    setup = build_setup(
        BASE_CONTEXT,
        BASE_POSITIONS_MM,
        fixed,
        str(BASE_CONFIG),
    )
    source, grid_x, grid_y, spacing, at_lens = prepare_source_and_lens(setup)
    after_lens, field = propagate_from_lens(
        setup,
        at_lens,
        grid_x,
        grid_y,
        spacing,
    )
    capture = capture_from_field(
        field,
        grid_x,
        grid_y,
        setup,
        "production_searchsorted",
    )
    return {
        "fixed": fixed,
        "setup": setup,
        "source": source,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "spacing": spacing,
        "at_lens": at_lens,
        "after_lens": after_lens,
        "field": field,
        "capture": capture,
    }


@pytest.fixture(scope="module")
def production_fixture() -> dict[str, object]:
    fixed = default_simulator_fixed(str(BASE_CONFIG))
    setup = build_setup(
        BASE_CONTEXT,
        BASE_POSITIONS_MM,
        fixed,
        str(BASE_CONFIG),
    )
    source, grid_x, grid_y, spacing, at_lens = prepare_source_and_lens(setup)
    _, field = propagate_from_lens(
        setup,
        at_lens,
        grid_x,
        grid_y,
        spacing,
    )
    capture = capture_from_field(
        field,
        grid_x,
        grid_y,
        setup,
        "production_searchsorted",
    )
    return {
        "fixed": fixed,
        "setup": setup,
        "source": source,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "spacing": spacing,
        "at_lens": at_lens,
        "field": field,
        "capture": capture,
    }


def test_01_no_op_repeatability(deterministic_fixture: dict[str, object]) -> None:
    """No-op repeats must remain bitwise deterministic."""

    capture_a = capture_from_field(
        deterministic_fixture["field"],
        deterministic_fixture["grid_x"],
        deterministic_fixture["grid_y"],
        deterministic_fixture["setup"],
        "production_searchsorted",
    )
    capture_b = capture_from_field(
        deterministic_fixture["field"],
        deterministic_fixture["grid_x"],
        deterministic_fixture["grid_y"],
        deterministic_fixture["setup"],
        "production_searchsorted",
    )
    assert np.array_equal(capture_a["intensity"], capture_b["intensity"])
    assert capture_a["stored_metrics"] == capture_b["stored_metrics"]


def _subpixel_camera_captures(
    deterministic_fixture: dict[str, object],
) -> list[dict[str, object]]:
    captures = []
    setup = deterministic_fixture["setup"]
    reference = float(BASE_POSITIONS_MM["camera_x_mm"])
    for delta_mm in np.linspace(-0.004, 0.004, 33):
        moved = copy_setup_with_position(
            setup,
            "camera_x_mm",
            reference + float(delta_mm),
        )
        capture = capture_from_field(
            deterministic_fixture["field"],
            deterministic_fixture["grid_x"],
            deterministic_fixture["grid_y"],
            moved,
            "production_searchsorted",
        )
        captures.append({"setup": moved, "capture": capture})
    return captures


def test_02_subpixel_camera_displacement_sweep(
    deterministic_fixture: dict[str, object],
) -> None:
    """The diagnostic sweep is far finer than pitch and still reproducible."""

    captures = _subpixel_camera_captures(deterministic_fixture)
    assert len(captures) == 33
    assert 0.00025 < float(deterministic_fixture["setup"].sensor.pixel_pitch * 1e3)


def test_03_detects_integer_index_plateaus_and_jumps(
    deterministic_fixture: dict[str, object],
) -> None:
    captures = _subpixel_camera_captures(deterministic_fixture)
    images = [item["capture"]["intensity"] for item in captures]
    equal = [
        np.array_equal(images[index - 1], images[index])
        for index in range(1, len(images))
    ]
    assert any(equal), "nearest-index sweep should expose at least one plateau"
    assert not all(equal), "the sweep should also cross an index boundary"


@pytest.mark.xfail(
    strict=True,
    reason="desired smooth subpixel camera response; production searchsorted is piecewise constant",
)
def test_03b_desired_subpixel_camera_response_has_no_plateaus(
    deterministic_fixture: dict[str, object],
) -> None:
    captures = _subpixel_camera_captures(deterministic_fixture)
    images = [item["capture"]["intensity"] for item in captures]
    assert all(
        not np.array_equal(images[index - 1], images[index])
        for index in range(1, len(images))
    )


def test_04_sensor_image_metric_recomputation(
    deterministic_fixture: dict[str, object],
) -> None:
    capture = deterministic_fixture["capture"]
    setup = deterministic_fixture["setup"]
    independent = independent_sensor_metrics(
        capture["intensity"],
        float(setup.sensor.pixel_pitch),
    )
    assert independent == capture["image_metrics"]


def test_05_lab_to_sensor_centroid_transformation(
    deterministic_fixture: dict[str, object],
) -> None:
    capture = deterministic_fixture["capture"]
    setup = deterministic_fixture["setup"]
    pitch = float(setup.sensor.pixel_pitch)
    assert capture["stored_metrics"]["centroid_x_px"] - capture[
        "sensor_frame_metrics"
    ]["centroid_x_px"] == pytest.approx(setup.camera.x_offset / pitch, abs=1e-12)
    assert capture["stored_metrics"]["centroid_y_px"] - capture[
        "sensor_frame_metrics"
    ]["centroid_y_px"] == pytest.approx(setup.camera.y_offset / pitch, abs=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="legacy stored centroid is lab-frame pseudo-pixel, not image-array pixel",
)
def test_05b_desired_stored_centroid_directly_matches_image(
    deterministic_fixture: dict[str, object],
) -> None:
    capture = deterministic_fixture["capture"]
    assert capture["stored_metrics"]["centroid_x_px"] == pytest.approx(
        capture["image_metrics"]["centroid_x_px"],
        abs=1e-9,
    )


def test_06_camera_xy_sign_convention(
    production_fixture: dict[str, object],
) -> None:
    baseline = production_fixture["capture"]["image_metrics"]
    setup = production_fixture["setup"]
    for position_field, metric in (
        ("camera_x_mm", "centroid_x_px"),
        ("camera_y_mm", "centroid_y_px"),
    ):
        moved = copy_setup_with_position(
            setup,
            position_field,
            float(BASE_POSITIONS_MM[position_field]) + 0.02,
        )
        capture = capture_from_field(
            production_fixture["field"],
            production_fixture["grid_x"],
            production_fixture["grid_y"],
            moved,
            "production_searchsorted",
        )
        assert capture["image_metrics"][metric] < baseline[metric]


def test_07_lens_xy_sign_convention(
    production_fixture: dict[str, object],
) -> None:
    baseline = production_fixture["capture"]["sensor_frame_metrics"]
    setup = production_fixture["setup"]
    for position_field, metric in (
        ("lens_x_mm", "centroid_x_px"),
        ("lens_y_mm", "centroid_y_px"),
    ):
        moved = copy_setup_with_position(
            setup,
            position_field,
            float(BASE_POSITIONS_MM[position_field]) + 0.02,
        )
        _, field = propagate_from_lens(
            moved,
            production_fixture["at_lens"],
            production_fixture["grid_x"],
            production_fixture["grid_y"],
            production_fixture["spacing"],
        )
        capture = capture_from_field(
            field,
            production_fixture["grid_x"],
            production_fixture["grid_y"],
            moved,
            "production_searchsorted",
        )
        assert capture["sensor_frame_metrics"][metric] > baseline[metric]


def test_08_width_unit_conversion_exposes_endpoint_scale(
    deterministic_fixture: dict[str, object],
) -> None:
    capture = deterministic_fixture["capture"]
    height, width = deterministic_fixture["setup"].sensor.resolution
    assert capture["sensor_frame_metrics"]["sigma_x_px"] == pytest.approx(
        capture["image_metrics"]["sigma_x_px"] * width / (width - 1),
        rel=1e-12,
    )
    assert capture["sensor_frame_metrics"]["sigma_y_px"] == pytest.approx(
        capture["image_metrics"]["sigma_y_px"] * height / (height - 1),
        rel=1e-12,
    )


def test_09_peak_intensity_source_consistency(
    deterministic_fixture: dict[str, object],
) -> None:
    capture = deterministic_fixture["capture"]
    assert capture["stored_metrics"]["peak_intensity"] == float(
        capture["intensity"].max()
    )
    assert capture["image_metrics"]["peak_intensity"] == float(
        capture["intensity"].max()
    )


def test_10_power_w_perturbation_is_currently_dead(
    deterministic_fixture: dict[str, object],
) -> None:
    fixed = deterministic_fixture["fixed"]
    hashes = []
    for power in (0.25, 1.0, 4.0):
        context = dict(BASE_CONTEXT)
        context["power_w"] = power
        setup = build_setup(
            context,
            BASE_POSITIONS_MM,
            fixed,
            str(BASE_CONFIG),
        )
        quantities = source_and_propagated_quantities(setup)
        hashes.append(exact_array_hash(quantities["capture"]["intensity"]))
    assert len(set(hashes)) == 1


@pytest.mark.xfail(
    strict=True,
    reason="desired absolute-intensity semantics; gaussian_source_field ignores source.power",
)
def test_10b_desired_power_w_scales_raw_intensity(
    deterministic_fixture: dict[str, object],
) -> None:
    fixed = deterministic_fixture["fixed"]
    sums = []
    for power in (1.0, 4.0):
        context = dict(BASE_CONTEXT)
        context["power_w"] = power
        setup = build_setup(
            context,
            BASE_POSITIONS_MM,
            fixed,
            str(BASE_CONFIG),
        )
        quantities = source_and_propagated_quantities(setup)
        sums.append(float(quantities["capture"]["intensity"].sum()))
    assert sums[1] / sums[0] == pytest.approx(4.0, rel=1e-12)


def test_11_raw_vs_normalized_intensity_scaling(
    deterministic_fixture: dict[str, object],
) -> None:
    """In a passive linear model, intensity scaling cancels only on normalization."""

    image = np.asarray(deterministic_fixture["capture"]["intensity"])
    scaled = 4.0 * image
    assert scaled.sum() / image.sum() == pytest.approx(4.0)
    assert np.array_equal(scaled / scaled.max(), image / image.max())


def test_12_identical_seeds_reproduce_identical_audit_results(
    deterministic_fixture: dict[str, object],
) -> None:
    assert stable_seed(20260730, "audit") == stable_seed(20260730, "audit")
    copied = copy.deepcopy(deterministic_fixture["setup"])
    metadata_a = sampling_metadata(
        deterministic_fixture["grid_x"],
        deterministic_fixture["grid_y"],
        deterministic_fixture["setup"],
        "x",
        "production_searchsorted",
    )
    metadata_b = sampling_metadata(
        deterministic_fixture["grid_x"],
        deterministic_fixture["grid_y"],
        copied,
        "x",
        "production_searchsorted",
    )
    assert metadata_a == metadata_b
