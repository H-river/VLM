import numpy as np

from optics_understanding_sft.direction_inverse_v1.evaluate_inverse_ensemble import blended_index


def test_blended_index_combines_costs() -> None:
    direct = np.asarray([0.0, 1.0])
    residual = np.asarray([10.0, 0.0])
    assert blended_index(direct, residual, 0.0) == 0
    assert blended_index(direct, residual, 2.0) == 1
