from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from optics_understanding_sft.visual_state_tool_v10_1 import (
    classify,
    classify_pair,
    classify_knn,
    extract_features,
    fit_ordered_thresholds,
    grayscale_without_overlays,
    neighbor_indices_with_ties,
)


def test_negative_denoise_passes_are_rejected(tmp_path) -> None:
    image = np.full((5, 5, 3), 10, dtype=np.uint8)
    path = tmp_path / "constant.png"
    Image.fromarray(image).save(path)
    with pytest.raises(ValueError, match="denoise_passes"):
        extract_features(path, denoise_passes=-1)


def test_fit_ordered_thresholds_and_classify() -> None:
    accuracy, low, high = fit_ordered_thresholds(
        [(1.0, "narrow"), (2.0, "narrow"), (4.0, "medium"), (7.0, "wide")],
        ["narrow", "medium", "wide"],
    )
    assert accuracy == 1.0
    calibration = {
        "centroid_x": {
            "thresholds": [500.0, 520.0],
            "labels": ["left_of_center", "centered", "right_of_center"],
        },
        "centroid_y": {
            "thresholds": [500.0, 520.0],
            "labels": ["above_center", "centered", "below_center"],
        },
        "sigma_x": {"thresholds": [low, high], "labels": ["narrow", "medium", "wide"]},
        "sigma_y": {"thresholds": [low, high], "labels": ["narrow", "medium", "wide"]},
    }
    result = classify(
        {
            "centroid_x_px": 490.0,
            "centroid_y_px": 530.0,
            "rendered_sigma_x_px": 4.0,
            "rendered_sigma_y_px": 7.0,
        },
        calibration,
    )
    assert result == {
        "centroid_horizontal_region": "left_of_center",
        "centroid_vertical_region": "below_center",
        "sigma_x_band": "medium",
        "sigma_y_band": "wide",
    }


def test_classify_pair_uses_calibrated_deadbands() -> None:
    calibration = {
        field: {"thresholds": [-0.5, 0.5]}
        for field in (
            "centroid_x",
            "centroid_y",
            "sigma_x",
            "sigma_y",
            "peak_intensity",
        )
    }
    result = classify_pair(
        {
            "centroid_x": -0.51,
            "centroid_y": 0.51,
            "sigma_x": -0.5,
            "sigma_y": 0.5,
            "peak_intensity": 0.0,
        },
        calibration,
    )
    assert result == {
        "centroid_x": "decrease",
        "centroid_y": "increase",
        "sigma_x": "no_change",
        "sigma_y": "no_change",
        "peak_intensity": "no_change",
    }


def test_classify_knn_uses_field_specific_k() -> None:
    calibration = {
        "classifier": "group_cv_knn_v1",
        "feature_order": ["centroid_x_px"],
        "normalization": {"mean": [0.0], "scale": [1.0]},
        "fields": {"region": {"k": 3}},
        "prototypes": [
            {"features": [0.0], "labels": {"region": "left"}},
            {"features": [0.1], "labels": {"region": "center"}},
            {"features": [0.2], "labels": {"region": "center"}},
        ],
    }
    assert classify_knn({"centroid_x_px": 0.0}, calibration) == {"region": "center"}


def test_neighbor_indices_include_boundary_ties() -> None:
    assert neighbor_indices_with_ties(np.asarray([0.0, 0.0, 1.0]), 1).tolist() == [0, 1]


def test_colored_overlay_pixel_is_interpolated_from_gray_neighbors(tmp_path) -> None:
    rgb = np.full((5, 5, 3), 100, dtype=np.uint8)
    rgb[2, 2] = (0, 220, 220)
    path = tmp_path / "overlay.png"
    Image.fromarray(rgb, mode="RGB").save(path)
    zeroed = grayscale_without_overlays(path, overlay_handling="mask_zero")
    interpolated = grayscale_without_overlays(
        path, overlay_handling="interpolate_colored_aids"
    )
    assert zeroed[2, 2] == 0.0
    assert interpolated[2, 2] == 100.0
