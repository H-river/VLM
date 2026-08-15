from __future__ import annotations

from optics_vla.common.config import load_controller_config
from optics_vla.control.continuation import continuation_horizon


def _step(before: float, after: float) -> dict[str, float]:
    return {"before_target_cost": before, "actual_target_cost": after}


def test_frozen_visible_rule_has_four_step_preamble_and_maximum_eight() -> None:
    config = load_controller_config()
    improving = [
        _step(8.0, 7.0),
        _step(7.0, 6.0),
        _step(6.0, 5.0),
        _step(5.0, 4.5),
        _step(4.5, 4.0),
        _step(4.0, 3.5),
        _step(3.5, 3.0),
        _step(3.0, 2.5),
        _step(2.5, 2.0),
    ]
    assert continuation_horizon(improving, config) == 8

    stalled = improving[:4] + [_step(4.5, 4.4)] + improving[5:]
    assert continuation_horizon(stalled, config) == 5

