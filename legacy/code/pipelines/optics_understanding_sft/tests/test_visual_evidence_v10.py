from __future__ import annotations

from optics_understanding_sft.build_visual_evidence_v10 import (
    pair_directions,
    state_regions,
    transformed_directions,
    transformed_regions,
)


def test_state_regions_and_flips() -> None:
    regions = state_regions(
        {
            "centroid_x_px": 500.0,
            "centroid_y_px": 520.0,
            "sigma_x_px": 100.0,
            "sigma_y_px": 145.0,
        }
    )
    assert regions == {
        "centroid_horizontal_region": "left_of_center",
        "centroid_vertical_region": "below_center",
        "sigma_x_band": "narrow",
        "sigma_y_band": "wide",
    }
    assert transformed_regions(regions, "horizontal_flip")[
        "centroid_horizontal_region"
    ] == "right_of_center"
    assert transformed_regions(regions, "vertical_flip")[
        "centroid_vertical_region"
    ] == "above_center"


def test_pair_thresholds_and_flip_semantics() -> None:
    before = {
        "centroid_x_px": 500.0,
        "centroid_y_px": 500.0,
        "sigma_x_px": 120.0,
        "sigma_y_px": 120.0,
        "peak_intensity": 10.0,
    }
    after = {
        "centroid_x_px": 503.0,
        "centroid_y_px": 499.5,
        "sigma_x_px": 116.0,
        "sigma_y_px": 121.0,
        "peak_intensity": 9.0,
    }
    values = pair_directions(before, after)
    assert values == {
        "centroid_x": "increase",
        "centroid_y": "no_change",
        "sigma_x": "decrease",
        "sigma_y": "no_change",
        "peak_intensity": "decrease",
    }
    assert transformed_directions(values, "horizontal_flip")["centroid_x"] == "decrease"
    assert transformed_directions(values, "vertical_flip")["centroid_x"] == "increase"

