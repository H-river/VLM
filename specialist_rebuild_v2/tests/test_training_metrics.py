from __future__ import annotations

import numpy as np

from specialist_rebuild_v2.train_models import (
    inverse_metrics,
    measurement_metrics,
)


def test_measurement_metric_uses_declared_tolerances() -> None:
    target = np.asarray([[500.0, 500.0, 100.0, 100.0, 80.0]])
    predicted = np.asarray([[501.0, 499.0, 102.0, 98.0, 84.0]])
    metrics = measurement_metrics(target, predicted)
    assert metrics["strict_all_five_success"] == 1.0
    assert metrics["mae_in_tolerance_units"] == 1.0
    assert all(value == 1.0 for value in metrics["per_field_pass"].values())


def test_inverse_metrics_separate_feasible_and_all_success() -> None:
    scores = np.asarray([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]])
    statuses = np.asarray([0, 2])
    status_logits = np.asarray([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    positives = np.asarray([[False, True, False], [False, False, False]])
    metrics = inverse_metrics(scores, status_logits, positives, statuses)
    assert metrics["target_success_feasible"] == 1.0
    assert metrics["target_success_all"] == 0.5
    assert metrics["status_accuracy"] == 1.0
