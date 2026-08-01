from __future__ import annotations

import numpy as np

from optics_understanding_sft.direction_inverse_v1.train_forward_small import (
    FEATURE_FIELDS, engineered_features, strict_metrics, target_arrays,
)


def _record() -> dict:
    return {"inputs": {"current_beam_state": {"peak_intensity": 100.0}}, "target": {
        "change": {"centroid_x_px": 1.0, "centroid_y_px": -2.0, "sigma_x_px": 3.0,
                   "sigma_y_px": -4.0, "peak_intensity": 10.0},
        "directions": {"centroid_x": "no_change", "centroid_y": "decrease",
                       "sigma_x": "increase", "sigma_y": "decrease", "peak_intensity": "increase"}}}


def test_target_scaling_uses_sensor_tolerances() -> None:
    _, scaled, directions = target_arrays([_record()])
    np.testing.assert_allclose(scaled[0], [1.0, -2.0, 1.5, -2.0, 2.0])
    assert directions.shape == (1, 5)


def test_strict_success_requires_every_field() -> None:
    record = _record(); _, target_scaled, directions = target_arrays([record])
    logits = np.full((1, 5, 3), -10.0, dtype=np.float32)
    for index, label in enumerate(directions[0]):
        logits[0, index, label] = 10.0
    assert strict_metrics([record], target_scaled.copy(), logits)["strict_all_five_success"] == 1.0
    one_bad = target_scaled.copy(); one_bad[0, 3] += 1.01
    assert strict_metrics([record], one_bad, logits)["strict_all_five_success"] == 0.0


def test_engineered_features_add_final_positions() -> None:
    x = np.zeros((1, len(FEATURE_FIELDS)), dtype=np.float32)
    index = {name: i for i, name in enumerate(FEATURE_FIELDS)}
    x[0, index["lens_x_offset_mm"]] = 0.1
    x[0, index["lens_x_delta_mm"]] = 0.05
    x[0, index["lens_focal_length_mm"]] = 100.0
    x[0, index["lens_to_camera_mm"]] = 100.0
    x[0, index["pixel_size_um"]] = 5.0
    output = engineered_features(x)
    assert output.shape[1] == x.shape[1] + 25
    assert abs(output[0, x.shape[1]] - 0.15) < 1e-6
