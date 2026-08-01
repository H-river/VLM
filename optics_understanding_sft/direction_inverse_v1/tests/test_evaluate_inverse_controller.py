from __future__ import annotations

import numpy as np

from optics_understanding_sft.direction_inverse_v1.evaluate_inverse_controller import classify, tune_calibration


def test_classify_uses_feasibility_and_ambiguity() -> None:
    assert classify(np.asarray([0.2, 0.9]), 1.0, 0.1) == "unique"
    assert classify(np.asarray([0.2, 0.25]), 1.0, 0.1) == "ambiguous"
    assert classify(np.asarray([1.2, 1.3]), 1.0, 0.1) == "infeasible_within_limits"


def test_calibration_is_validation_only_grid_search() -> None:
    records = [{"target": {"status": status}} for status in
               ("unique", "ambiguous", "infeasible_within_limits")]
    grids = [np.asarray([0.2, 0.8]), np.asarray([0.2, 0.22]), np.asarray([4.0, 5.0])]
    result = tune_calibration(records, grids)
    assert result["validation_status_macro_f1"] == 1.0
