from __future__ import annotations

import numpy as np
from PIL import Image

from optics_understanding_sft.direction_inverse_v1.evaluate_visual_pipeline import measure_image


def test_measure_image_recovers_calibrated_moments(tmp_path) -> None:
    y, x = np.mgrid[0:64, 0:64]
    intensity = np.exp(-0.5 * (((x - 30.0) / 5.0) ** 2 + ((y - 34.0) / 7.0) ** 2))
    path = tmp_path / "beam.png"
    Image.fromarray(np.rint(intensity * 255).astype(np.uint8), mode="L").convert("RGB").save(path)
    calibration = {"gamma": 1.0, "linear_intensity_low": 0.0, "linear_intensity_high": 1.0,
                   "source_sensor_resolution_px": [64, 64]}
    state = measure_image(path, calibration)
    assert abs(state["centroid_x_px"] - 30.0) < 0.1
    assert abs(state["centroid_y_px"] - 34.0) < 0.1
    assert abs(state["sigma_x_px"] - 5.0) < 0.2
    assert abs(state["sigma_y_px"] - 7.0) < 0.2
