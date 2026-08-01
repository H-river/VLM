from __future__ import annotations

import numpy as np

from control_rebuild_v4.orchestrated_runtime import (
    DIRECTION_V4_ROUTE_ARTIFACTS,
    DIRECTION_V4_ROUTE_BACKENDS,
    OrchestratedSpecialistRuntimeV4,
    ROUTE_ARTIFACTS,
    ROUTE_BACKENDS,
)
from direction_rebuild_v4.data import (
    balance_table,
    direction_metrics,
    distance_bins,
    labels_from_normalized_change,
)


def test_direction_thresholds_and_boundary_distance_bins() -> None:
    changes = np.asarray(
        [
            [-1.50, -1.00, -0.80, 0.00, 1.00],
            [1.01, 1.20, 1.80, -1.01, -2.00],
        ],
        dtype=np.float32,
    )
    labels = labels_from_normalized_change(changes)
    assert labels.tolist() == [
        [0, 1, 1, 1, 1],
        [2, 2, 2, 0, 0],
    ]
    bins = distance_bins(changes)
    assert bins.tolist() == [
        [1, 0, 0, 2, 0],
        [0, 0, 2, 0, 2],
    ]


def test_balance_table_normalizes_observed_sample_weights() -> None:
    labels = np.asarray(
        [
            [0, 1, 2, 1, 1],
            [0, 1, 2, 1, 1],
            [1, 2, 0, 2, 1],
            [2, 0, 1, 0, 2],
        ],
        dtype=np.int64,
    )
    bins = np.asarray(
        [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0],
        ],
        dtype=np.int64,
    )
    weights, report = balance_table(labels, bins)
    assert weights.shape == (5, 3, 3)
    assert np.isfinite(weights).all()
    assert report["distance_bin_edges"] == [0.25, 0.75]
    for field in range(5):
        observed = weights[field, labels[:, field], bins[:, field]]
        np.testing.assert_allclose(
            observed.mean(),
            1.0,
            atol=1e-6,
        )


def test_direction_metric_requires_all_five_for_joint_success() -> None:
    target = np.ones((2, 5), dtype=np.int64)
    predicted = target.copy()
    predicted[1, 4] = 2
    metrics = direction_metrics(target, predicted)
    assert metrics["joint_exact_count"] == 1
    assert metrics["joint_exact"] == 0.5
    assert metrics["mean_field_accuracy"] == 0.9


def test_direction_v4_overlay_changes_only_direction_routes() -> None:
    changed = {
        route
        for route in ROUTE_BACKENDS
        if ROUTE_BACKENDS[route] != DIRECTION_V4_ROUTE_BACKENDS[route]
        or ROUTE_ARTIFACTS[route] != DIRECTION_V4_ROUTE_ARTIFACTS[route]
    }
    assert changed == {
        "predict_direction_from_state_v1",
        "predict_direction_from_image_v1",
    }


def test_orchestrated_runtime_uses_optional_direction_v4() -> None:
    class StubDirection:
        def predict_one(self, setup, current, action):
            return {
                "directions": {
                    "centroid_x": "increase",
                    "centroid_y": "no_change",
                    "width_x": "no_change",
                    "width_y": "decrease",
                    "peak_intensity": "increase",
                },
                "model_version": "stub-direction-v4",
                "simulator_at_inference": False,
            }

    runtime = OrchestratedSpecialistRuntimeV4.__new__(
        OrchestratedSpecialistRuntimeV4
    )
    runtime.direction = StubDirection()
    result = runtime.execute_ready(
        "predict_direction_from_state_v1",
        {
            "setup": {},
            "current_beam_state": {},
            "action": {},
        },
        {},
    )
    assert result["model_version"] == "stub-direction-v4"
    assert result["directions"]["centroid_x"] == "increase"
