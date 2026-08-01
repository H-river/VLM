from __future__ import annotations

import numpy as np

from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    change_and_directions,
    direction_feature,
    fixed_action_grid,
    forward_feature,
    inverse_context,
    matching_mask,
    minimum_motion_index,
)


SETUP = {
    "wavelength_nm": 532.0,
    "beam_waist_mm": 0.8,
    "power_w": 0.01,
    "lens_focal_length_mm": 100.0,
    "lens_aperture_mm": 25.4,
    "source_to_lens_mm": 120.0,
    "lens_to_camera_mm": 100.0,
    "lens_x_offset_mm": 0.01,
    "lens_y_offset_mm": -0.01,
    "camera_x_offset_mm": 0.005,
    "camera_y_offset_mm": -0.005,
    "pixel_size_um": 5.0,
}
CURRENT = {
    "centroid_x_px": 96.0,
    "centroid_y_px": 96.0,
    "sigma_x_px": 12.0,
    "sigma_y_px": 13.0,
    "peak_intensity": 1000.0,
}


def test_fixed_action_grid_is_complete_and_centered() -> None:
    grid = fixed_action_grid()
    assert len(grid) == 81
    assert len({tuple(row[key] for key in ACTION_FIELDS) for row in grid}) == 81
    assert grid[40] == {key: 0.0 for key in ACTION_FIELDS}


def test_feature_dimensions_are_frozen() -> None:
    action = fixed_action_grid()[0]
    desired = dict(CURRENT, centroid_x_px=97.0)
    assert direction_feature(SETUP, CURRENT, action).shape == (21,)
    assert forward_feature(SETUP, CURRENT, action).shape == (46,)
    assert inverse_context(SETUP, CURRENT, desired).shape == (31,)


def test_matching_and_minimum_motion_policy() -> None:
    desired = np.asarray([96.0, 96.0, 12.0, 13.0, 1000.0])
    states = np.stack(
        [
            desired,
            desired + np.asarray([0.2, 0.2, 0.2, 0.2, 10.0]),
            desired + np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )
    assert matching_mask(states, desired).tolist() == [True, True, False]
    assert minimum_motion_index([0, 40, 80]) == 40


def test_direction_labels_use_defined_dead_bands() -> None:
    after = dict(
        CURRENT,
        centroid_x_px=97.1,
        centroid_y_px=95.0,
        sigma_x_px=9.9,
        sigma_y_px=15.0,
        peak_intensity=1051.0,
    )
    _, labels = change_and_directions(CURRENT, after)
    assert labels == {
        "centroid_x": "increase",
        "centroid_y": "no_change",
        "width_x": "decrease",
        "width_y": "no_change",
        "peak_intensity": "increase",
    }
