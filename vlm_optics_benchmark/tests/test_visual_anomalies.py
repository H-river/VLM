from __future__ import annotations

import numpy as np

from vlm_optics_benchmark.visual_anomalies import (
    inject_anomaly,
    matched_clean_counterfactual,
    moment_metrics,
    normalized_metric_distance,
    severity_for,
)


def _beam() -> np.ndarray:
    y, x = np.indices((128, 128), dtype=float)
    image = np.exp(-0.5 * (((x - 63.0) / 9.0) ** 2 + ((y - 65.0) / 12.0) ** 2))
    return image.astype(np.float32)


def test_anomaly_injection_is_reproducible() -> None:
    for family in ("sensor_saturation", "secondary_reflection"):
        severity = severity_for("case", family, "train")
        assert severity == severity_for("case", family, "train")
        first = inject_anomaly(_beam(), family, severity)
        second = inject_anomaly(_beam(), family, severity)
        np.testing.assert_array_equal(first, second)


def test_counterfactual_matches_frozen_metric_criterion() -> None:
    base = _beam()
    anomaly = inject_anomaly(
        base,
        "secondary_reflection",
        {
            "reflection_amplitude_fraction": 0.35,
            "reflection_offset_px": 18.0,
            "reflection_angle_radians": 0.7,
        },
    )
    target = moment_metrics(anomaly)
    clean = matched_clean_counterfactual(base, target)
    match = normalized_metric_distance(moment_metrics(clean), target)
    assert match["passes_frozen_match"]

