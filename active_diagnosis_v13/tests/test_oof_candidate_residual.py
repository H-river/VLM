from __future__ import annotations

import numpy as np

from active_diagnosis_v13.analyze_oof_candidate_residual_calibration import (
    _oof_predictions,
)


def test_residual_predictions_are_group_held_out() -> None:
    records = []
    group_residual = {"a": 1.0, "b": 2.0, "c": 5.0}
    for group, residual in group_residual.items():
        for value in (0.0, 1.0):
            records.append(
                {
                    "group_id": group,
                    "visible_features": [value, value + 0.5],
                    "normalized_residual": [residual] * 5,
                }
            )
    constant, ridge = _oof_predictions(records)
    expected = {"a": 3.5, "b": 3.0, "c": 1.5}
    for index, row in enumerate(records):
        assert np.allclose(constant[index], expected[row["group_id"]])
    assert ridge.shape == (6, 5)
    assert np.isfinite(ridge).all()
