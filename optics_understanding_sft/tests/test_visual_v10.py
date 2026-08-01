from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from optics_sft.physics.sim_adapter import state_m_to_sensor_frame_px
from optics_understanding_sft.visual_v10 import (
    render_calibrated_images,
    render_signed_difference,
    sensor_frame_state,
)


def dummy_setup() -> SimpleNamespace:
    return SimpleNamespace(
        sensor=SimpleNamespace(resolution=(100, 200), pixel_pitch=5e-6),
        camera=SimpleNamespace(x_offset=10e-6, y_offset=-15e-6),
    )


def test_sensor_frame_conversion_subtracts_camera_offset() -> None:
    setup = dummy_setup()
    state = state_m_to_sensor_frame_px(
        {
            "centroid_x_m": 15e-6,
            "centroid_y_m": -5e-6,
            "sigma_x_m": 20e-6,
            "sigma_y_m": 30e-6,
        },
        setup,
    )
    assert state == {
        "centroid_x_px": 100.5,
        "centroid_y_px": 51.5,
        "sigma_x_px": 4.0,
        "sigma_y_px": 6.0,
    }


def test_visual_state_uses_same_sensor_frame() -> None:
    setup = dummy_setup()
    result = {
        "metrics": {
            "centroid_x": 15e-6,
            "centroid_y": -5e-6,
            "sigma_x": 20e-6,
            "sigma_y": 30e-6,
            "peak_intensity": 2.5,
        }
    }
    state = sensor_frame_state(result, setup)
    assert state["centroid_x_px"] == 100.5
    assert state["centroid_y_px"] == 51.5
    assert state["peak_intensity"] == 2.5


def test_pair_render_uses_shared_absolute_scale(tmp_path) -> None:
    first = np.linspace(0.0, 2.0, 64, dtype=np.float64).reshape(8, 8)
    second = first * 0.5
    paths, calibration = render_calibrated_images(
        [("first", first), ("second", second)],
        tmp_path,
        size_px=8,
        percentile_clip=(0.0, 100.0),
    )
    first_pixels = np.asarray(Image.open(tmp_path / paths[0]))[..., 0]
    second_pixels = np.asarray(Image.open(tmp_path / paths[1]))[..., 0]
    assert abs(int(first_pixels.max()) - 2 * int(second_pixels.max())) <= 2
    assert calibration["pair_shared_calibration"] is True
    assert calibration["shared_normalization_bounds"] == [0.0, 2.0]


def test_instrumented_crop_and_signed_difference(tmp_path) -> None:
    first = np.zeros((16, 16), dtype=np.float64)
    second = np.zeros((16, 16), dtype=np.float64)
    first[8, 6] = 1.0
    second[8, 10] = 1.0
    paths, calibration = render_calibrated_images(
        [("first", first), ("second", second)],
        tmp_path,
        size_px=16,
        percentile_clip=(0.0, 100.0),
        sensor_crop_px=16,
        instrumented=True,
    )
    assert calibration["instrumented_overlay"] is True
    assert calibration["sensor_crop_px"] == 16
    image = np.asarray(Image.open(tmp_path / paths[0]))
    assert image[7, 7, 1] > 100  # cyan center aid

    metadata = render_signed_difference(
        first, second, tmp_path / "difference.png", size_px=16, sensor_crop_px=16
    )
    difference = np.asarray(Image.open(tmp_path / "difference.png"))
    assert difference[8, 10, 0] > difference[8, 10, 2]
    assert difference[8, 6, 2] > difference[8, 6, 0]
    assert metadata["color_key"]["red"] == "second_brighter"
