from __future__ import annotations

import numpy as np

from vlm_optics_benchmark.visual_anomalies import severity_for
from vlm_optics_benchmark.visual_control import _full_anomaly


def test_full_anomaly_replay_is_deterministic_per_family() -> None:
    y, x = np.indices((256, 256), dtype=float)
    raw = np.exp(-0.5 * (((x - 127) / 18) ** 2 + ((y - 129) / 22) ** 2)).astype(np.float32)
    for family in ("sensor_saturation", "secondary_reflection"):
        severity = severity_for("case", family, "severity_ood")
        first_full, first_view = _full_anomaly(raw, family, severity)
        second_full, second_view = _full_anomaly(raw, family, severity)
        np.testing.assert_array_equal(first_full, second_full)
        np.testing.assert_array_equal(first_view, second_view)

