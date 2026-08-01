from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from measurement_rebuild_v3.build_dataset import sensor_frame_state
from measurement_rebuild_v3.common import analytic_measurement
from measurement_rebuild_v3.models import measurement_model_v3, require_torch
from measurement_rebuild_v3.train import apply_transform, prepare_view, train_loss


def test_spatial_model_shape_and_preserved_grid() -> None:
    torch = require_torch()
    model = measurement_model_v3(torch)
    output = model(
        torch.rand(2, 1, 128, 128),
        torch.rand(2, 1, 128, 128),
        torch.ones(2, 1, 128, 128),
        torch.rand(2, 11),
        torch.rand(2, 9),
    )
    assert output.shape == (2, 5)
    assert tuple(model.spatial[-1].output_size) == (4, 4)


def test_analytic_measurement_recovers_gaussian_state() -> None:
    size = 512
    source = 1024
    y, x = np.mgrid[:size, :size]
    source_x = (x + 0.5) * source / size - 0.5
    source_y = (y + 0.5) * source / size - 0.5
    truth = np.asarray([505.4, 497.7, 116.0, 121.0, 2.5])
    image = np.exp(
        -0.5
        * (
            ((source_x - truth[0]) / truth[2]) ** 2
            + ((source_y - truth[1]) / truth[3]) ** 2
        )
    )
    baseline, _ = analytic_measurement(
        image, np.ones_like(image), truth[4], (source, source)
    )
    tolerance = np.asarray([1.0, 1.0, 2.0, 2.0, 0.05 * truth[4]])
    assert np.all(np.abs(baseline - truth) <= tolerance)


def test_crop_masks_without_rescaling_coordinates() -> None:
    image = np.ones((64, 64), dtype=np.float32)
    transform = {
        "exposure": 1.0,
        "gamma": 1.0,
        "noise_std": 0.0,
        "blur_sigma_px": 0.0,
        "saturation_level": 1.0,
        "crop_left_px": 8,
        "crop_right_px": 0,
        "crop_top_px": 0,
        "crop_bottom_px": 8,
    }
    observed, linearized, valid = apply_transform(image, transform, 1)
    assert observed.shape == image.shape
    assert np.all(valid[:, :8] == 0.0)
    assert np.all(valid[-8:, :] == 0.0)
    assert np.all(linearized[valid == 0.0] == 0.0)


def test_sensor_frame_label_subtracts_final_camera_offset() -> None:
    state = {
        "centroid_x_px": 510.0,
        "centroid_y_px": 514.0,
        "sigma_x_px": 80.0,
        "sigma_y_px": 90.0,
        "peak_intensity": 4.0,
    }
    setup = {
        "pixel_size_um": 5.0,
        "camera_x_offset_mm": 0.010,
        "camera_y_offset_mm": -0.020,
    }
    action = {"camera_x_delta_mm": 0.005, "camera_y_delta_mm": -0.005}
    converted = sensor_frame_state(state, setup, action)
    assert converted["centroid_x_px"] == pytest.approx(507.0)
    assert converted["centroid_y_px"] == pytest.approx(519.0)
    assert converted["sigma_x_px"] == state["sigma_x_px"]


def test_prediction_scale_does_not_use_ground_truth_peak(tmp_path) -> None:
    torch = require_torch()
    image = np.zeros((64, 64), dtype=np.uint16)
    image[32, 32] = 32768
    Image.fromarray(image).save(tmp_path / "beam.png")
    row = {
        "state_id": "state",
        "base_image": "beam.png",
        "image_calibration": {
            "linear_intensity_high": 2.0,
            "source_sensor_resolution_px": [64, 64],
            "stored_resolution_px": [64, 64],
        },
        "target_state": {
            "centroid_x_px": 32.0,
            "centroid_y_px": 32.0,
            "sigma_x_px": 1.0,
            "sigma_y_px": 1.0,
            "peak_intensity": 100.0,
        },
    }
    transform = {
        "exposure": 1.0,
        "gamma": 1.0,
        "noise_std": 0.0,
        "blur_sigma_px": 0.0,
        "saturation_level": 1.0,
        "crop_left_px": 0,
        "crop_right_px": 0,
        "crop_top_px": 0,
        "crop_bottom_px": 0,
    }
    prepared = prepare_view(torch, tmp_path, row, "clean", transform)
    baseline = prepared[5].numpy()
    prediction_scale = prepared[6].numpy()
    target_tolerance = prepared[8].numpy()
    assert prediction_scale[4] == pytest.approx(0.05 * baseline[4])
    assert target_tolerance[4] == pytest.approx(5.0)


def test_tolerance_loss_penalizes_larger_errors() -> None:
    torch = require_torch()
    small = train_loss(torch, torch.full((4, 5), 0.25))
    large = train_loss(torch, torch.full((4, 5), 2.0))
    assert float(large) > float(small)
