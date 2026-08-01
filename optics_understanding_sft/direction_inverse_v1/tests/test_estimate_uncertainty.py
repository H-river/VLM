from optics_understanding_sft.direction_inverse_v1.estimate_uncertainty import wilson


def test_wilson_interval_contains_point() -> None:
    low, high = wilson(5, 10)
    assert low < 0.5 < high
    assert wilson(0, 0) == [0.0, 0.0]
