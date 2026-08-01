from __future__ import annotations

import math

import numpy as np
from PIL import Image

from vlm_optics_benchmark.reflection_width_relative import (
    directional_sigma,
    inject_width_relative_reflection,
    primary_centroid_covariance,
)


def _beam(
    *,
    shape: tuple[int, int] = (128, 128),
    center: tuple[float, float] = (63.0, 65.0),
    sigma: tuple[float, float] = (7.0, 13.0),
) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    cx, cy = center
    sx, sy = sigma
    return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2)).astype(
        np.float32
    )


def test_directional_sigma_horizontal_vertical_and_diagonal() -> None:
    covariance = np.asarray([[9.0, 2.0], [2.0, 16.0]])
    assert directional_sigma(covariance, [1.0, 0.0]) == 3.0
    assert directional_sigma(covariance, [0.0, 1.0]) == 4.0
    expected = math.sqrt(0.5 * (9.0 + 2.0 + 2.0 + 16.0))
    assert math.isclose(
        directional_sigma(covariance, [1.0, 1.0]), expected, rel_tol=1e-15
    )


def test_injection_separation_uses_clean_primary_covariance_in_all_directions() -> None:
    primary = _beam()
    _, covariance = primary_centroid_covariance(primary)
    for angle in (0.0, math.pi / 2.0, math.pi / 4.0):
        _, metadata = inject_width_relative_reflection(
            primary,
            k=2.25,
            amplitude=0.4,
            width_ratio=1.0,
            angle_radians=angle,
        )
        unit = [math.cos(angle), math.sin(angle)]
        expected = 2.25 * directional_sigma(covariance, unit)
        assert metadata["separation_px"] == expected
        assert metadata["covariance_source"] == "clean_primary_before_reflection_injection"


def test_narrow_and_broad_beams_have_same_dimensionless_separation() -> None:
    for sigma in ((4.0, 6.0), (14.0, 21.0)):
        _, metadata = inject_width_relative_reflection(
            _beam(sigma=sigma),
            k=2.75,
            amplitude=0.5,
            width_ratio=0.8,
            angle_radians=0.35,
        )
        assert metadata["separation_px"] / metadata["sigma_direction_px"] == 2.75


def test_reproducible_injection_from_fixed_parameters() -> None:
    primary = _beam()
    kwargs = {
        "k": 2.25,
        "amplitude": 0.4,
        "width_ratio": 1.2,
        "angle_radians": 1.7,
    }
    first, first_metadata = inject_width_relative_reflection(primary, **kwargs)
    second, second_metadata = inject_width_relative_reflection(primary, **kwargs)
    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata


def test_no_dependence_on_combined_anomalous_covariance() -> None:
    primary = _beam(sigma=(5.0, 15.0))
    anomaly, metadata = inject_width_relative_reflection(
        primary,
        k=2.75,
        amplitude=0.5,
        width_ratio=1.2,
        angle_radians=0.6,
    )
    _, clean_covariance = primary_centroid_covariance(primary)
    _, anomalous_covariance = primary_centroid_covariance(anomaly)
    expected = directional_sigma(clean_covariance, metadata["direction_unit_xy"])
    wrong_combined_value = directional_sigma(
        anomalous_covariance, metadata["direction_unit_xy"]
    )
    assert metadata["sigma_direction_px"] == expected
    assert not np.isclose(metadata["sigma_direction_px"], wrong_combined_value)


def test_near_sensor_boundary_is_finite_and_zero_fill_does_not_wrap() -> None:
    primary = _beam(center=(119.0, 62.0), sigma=(5.0, 8.0))
    anomaly, metadata = inject_width_relative_reflection(
        primary,
        k=2.75,
        amplitude=0.5,
        width_ratio=1.0,
        angle_radians=0.0,
    )
    assert np.all(np.isfinite(anomaly))
    assert 0.0 <= metadata["component_power_retained_fraction"] < 1.0
    assert float(anomaly[:, :8].max()) < 1e-4


def test_serialization_and_replay_consistency(tmp_path) -> None:
    primary = _beam()
    kwargs = {
        "k": 2.25,
        "amplitude": 0.4,
        "width_ratio": 0.8,
        "angle_radians": math.pi / 4.0,
    }
    anomaly, metadata = inject_width_relative_reflection(primary, **kwargs)
    path = tmp_path / "anomaly.png"
    Image.fromarray(np.rint(anomaly * 255).astype(np.uint8), mode="L").save(path)
    stored = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    replay, replay_metadata = inject_width_relative_reflection(primary, **kwargs)
    replay_stored = np.rint(replay * 255).astype(np.uint8).astype(np.float32) / 255.0
    np.testing.assert_array_equal(stored, replay_stored)
    assert metadata == replay_metadata
