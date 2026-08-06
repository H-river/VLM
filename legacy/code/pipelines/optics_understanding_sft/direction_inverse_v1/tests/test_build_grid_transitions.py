from optics_understanding_sft.direction_inverse_v1.build_grid_transitions import direction


def test_direction_threshold_is_inclusive_no_change() -> None:
    assert direction(1.0, 1.0) == "no_change"
    assert direction(-1.0, 1.0) == "no_change"
    assert direction(1.01, 1.0) == "increase"
    assert direction(-1.01, 1.0) == "decrease"
