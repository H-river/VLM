from pathlib import Path

from optics_understanding_sft import run_adaptive_visual_tool_v10_15 as adaptive
from optics_understanding_sft.run_adaptive_visual_tool_v10_15 import quality_route


def test_quality_route_prefers_blur_then_noise_then_clean():
    assert quality_route([{"colored_fraction": 0.01, "border_noise_scale": 0.0}]) == "clean"
    assert quality_route([{"colored_fraction": 0.01, "border_noise_scale": 2.0}]) == "noisy"
    assert (
        quality_route([{"colored_fraction": 0.04, "border_noise_scale": 2.0}])
        == "blurred"
    )


def test_quality_route_separates_dim_plus_noise_from_full_gain_noise():
    assert quality_route(
        [
            {
                "colored_fraction": 0.01,
                "border_noise_scale": 2.0,
                "max_channel_value": 181.0,
            }
        ]
    ) == "dim_noisy"
    assert quality_route(
        [
            {
                "colored_fraction": 0.01,
                "border_noise_scale": 2.0,
                "max_channel_value": 225.0,
            }
        ]
    ) == "noisy"


def test_pair_width_hybrid_changes_only_width_fields(monkeypatch):
    monkeypatch.setattr(
        adaptive,
        "image_quality",
        lambda _: {
            "colored_fraction": 0.01,
            "border_noise_scale": 2.0,
            "max_channel_value": 225.0,
        },
    )
    monkeypatch.setattr(
        adaptive,
        "pair_features",
        lambda _first, _second, calibration: {"calibration": calibration["name"]},
    )
    answers = {
        "clean": {
            "centroid_x": "clean_x",
            "centroid_y": "clean_y",
            "sigma_x": "clean_sx",
            "sigma_y": "clean_sy",
            "peak_intensity": "clean_peak",
        },
        "robust": {
            "centroid_x": "robust_x",
            "centroid_y": "robust_y",
            "sigma_x": "robust_sx",
            "sigma_y": "robust_sy",
            "peak_intensity": "robust_peak",
        },
        "width": {
            "centroid_x": "width_x",
            "centroid_y": "width_y",
            "sigma_x": "width_sx",
            "sigma_y": "width_sy",
            "peak_intensity": "width_peak",
        },
    }
    monkeypatch.setattr(
        adaptive,
        "classify_pair",
        lambda features, _calibration: answers[features["calibration"]],
    )

    answer, evidence = adaptive.adaptive_pair_answer(
        Path("first.png"),
        Path("second.png"),
        {"name": "clean"},
        {"name": "robust"},
        {"name": "width"},
    )

    assert answer == {
        "centroid_x": "robust_x",
        "centroid_y": "robust_y",
        "sigma_x": "width_sx",
        "sigma_y": "width_sy",
        "peak_intensity": "clean_peak",
    }
    assert evidence["route"] == "noisy"


def test_pair_feature_options_reach_pair_extractor(monkeypatch):
    captured = {}

    def fake_pair(*args, **kwargs):
        captured.update(kwargs)
        return {"ok": 1.0}

    monkeypatch.setattr(adaptive, "extract_pair_features", fake_pair)
    result = adaptive.pair_features(
        Path("first.png"),
        Path("second.png"),
        {
            "width_estimator": "projected_1d",
            "include_differential_features": True,
            "differential_floor_sigma": 2.5,
        },
    )
    assert result == {"ok": 1.0}
    assert captured["width_estimator"] == "projected_1d"
    assert captured["include_differential_features"] is True
    assert captured["differential_floor_sigma"] == 2.5


def test_state_denoise_option_reaches_state_extractor(monkeypatch):
    captured = {}

    def fake_state(*args, **kwargs):
        captured.update(kwargs)
        return {"ok": 1.0}

    monkeypatch.setattr(adaptive, "extract_features", fake_state)
    result = adaptive.state_features(Path("state.png"), {"denoise_passes": 2})
    assert result == {"ok": 1.0}
    assert captured["denoise_passes"] == 2
