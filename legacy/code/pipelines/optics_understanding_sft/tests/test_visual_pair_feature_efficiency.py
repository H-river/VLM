from pathlib import Path

import numpy as np

from optics_understanding_sft import visual_state_tool_v10_1 as tool


def test_equal_axis_powers_reuse_weighted_feature_extraction(monkeypatch):
    calls = []

    def fake_extract(path, **kwargs):
        calls.append((path, kwargs.get("intensity_power", 1.0)))
        offset = 1.0 if path.name == "second.png" else 0.0
        return {
            "centroid_x_px": offset,
            "centroid_y_px": offset,
            "rendered_sigma_x_px": 2.0 + offset,
            "rendered_sigma_y_px": 3.0 + offset,
        }

    monkeypatch.setattr(tool, "extract_features", fake_extract)
    monkeypatch.setattr(
        tool,
        "grayscale_without_overlays",
        lambda *args, **kwargs: np.ones((2, 2), dtype=np.float64),
    )
    result = tool.extract_pair_features(
        Path("first.png"),
        Path("second.png"),
        sigma_x_power=2.0,
        sigma_y_power=2.0,
    )
    assert len(calls) == 4
    assert result["sigma_x"] == 1.0
    assert result["sigma_y"] == 1.0
    assert result["sigma_x_first"] == 2.0
    assert result["sigma_x_second"] == 3.0
    assert result["peak_proxy_first"] == 4.0
    assert result["peak_proxy_second"] == 4.0


def test_signed_difference_width_contrast_tracks_wider_gaussian(tmp_path):
    from PIL import Image

    yy, xx = np.indices((64, 64), dtype=np.float64)
    narrow = np.exp(-((xx - 31.5) ** 2 + (yy - 31.5) ** 2) / (2.0 * 6.0**2))
    wide = np.exp(-((xx - 31.5) ** 2 + (yy - 31.5) ** 2) / (2.0 * 9.0**2))
    narrow /= narrow.sum()
    wide /= wide.sum()
    paths = []
    for name, values in (("narrow", narrow), ("wide", wide)):
        path = tmp_path / f"{name}.png"
        pixels = np.rint(values / max(narrow.max(), wide.max()) * 255.0).astype(np.uint8)
        Image.fromarray(pixels, mode="L").convert("RGB").save(path)
        paths.append(path)
    widening = tool.extract_differential_features(
        paths[0], paths[1], noise_floor_sigma=0.0
    )
    narrowing = tool.extract_differential_features(
        paths[1], paths[0], noise_floor_sigma=0.0
    )
    assert widening["diff_sigma_x_contrast"] > 0.0
    assert widening["diff_sigma_y_contrast"] > 0.0
    assert narrowing["diff_sigma_x_contrast"] < 0.0
    assert narrowing["diff_sigma_y_contrast"] < 0.0
