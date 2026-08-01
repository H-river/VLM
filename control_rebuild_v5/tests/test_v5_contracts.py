from __future__ import annotations

import copy

import numpy as np

from control_rebuild_v5.simulator_grid import simulate_fixed_action_grid
from optical_sim.src.optical_elements import setup_from_dict
from optics_sft.physics.sim_adapter import (
    apply_action_to_setup,
    simulate_and_measure,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    STATE_FIELDS,
    fixed_action_grid,
)


def small_setup():
    return setup_from_dict(
        {
            "source": {
                "type": "gaussian",
                "wavelength": 632.8e-9,
                "beam_waist": 1.0e-3,
                "power": 1.0,
            },
            "lens": {
                "focal_length": 0.1,
                "clear_aperture": 0.025,
                "diameter": 0.0254,
                "x_offset": 0.0001,
                "y_offset": -0.0001,
            },
            "sensor": {
                "resolution": [64, 64],
                "pixel_pitch": 5.5e-6,
            },
            "geometry": {
                "laser_to_lens": 0.2,
                "lens_to_camera": 0.15,
            },
            "camera": {
                "x_offset": 0.00005,
                "y_offset": -0.00005,
            },
            "alignment": {
                "x_offset": 0.0,
                "y_offset": 0.0,
                "tilt_x": 0.0,
                "tilt_y": 0.0,
                "defocus": 0.0,
            },
            "simulation": {
                "grid_size": 64,
                "grid_extent": 0.03,
                "propagation_backend": "fresnel_numpy",
            },
        }
    )


def test_cached_grid_is_identical_to_direct_simulator() -> None:
    setup = small_setup()
    cached = simulate_fixed_action_grid(setup)
    actions = fixed_action_grid()
    assert len(cached) == len(actions) == 81
    for index, action in enumerate(actions):
        assert cached[index]["action"] == action
        direct_setup = apply_action_to_setup(copy.deepcopy(setup), action)
        direct = simulate_and_measure(direct_setup)["state"]
        for field in STATE_FIELDS:
            assert float(cached[index]["state"][field]) == float(direct[field])


def test_action_grid_zero_index_and_order() -> None:
    actions = fixed_action_grid()
    assert len(actions) == 81
    assert all(float(actions[40][field]) == 0.0 for field in ACTION_FIELDS)
    assert len(
        {
            tuple(float(action[field]) for field in ACTION_FIELDS)
            for action in actions
        }
    ) == 81

