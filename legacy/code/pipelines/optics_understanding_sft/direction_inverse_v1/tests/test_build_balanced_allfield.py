import numpy as np

from optics_understanding_sft.direction_inverse_v1.build_balanced_allfield import deterministic_counts


def test_deterministic_counts_preserve_total() -> None:
    counts = deterministic_counts(np.asarray([0.2, 0.3, 0.5]), 11)
    assert counts.sum() == 11
    assert counts.tolist() == [2, 3, 6]
