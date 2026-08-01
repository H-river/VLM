"""Cached simulator execution for the fixed 81-action control grid."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from optical_sim.src.metrics import compute_metrics
from optical_sim.src.optical_elements import OpticalSetup
from optical_sim.src.simulator import (
    _BACKENDS,
    _extract_sensor_region,
    apply_thin_lens,
    gaussian_source_field,
)
from optics_sft.physics.sim_adapter import metrics_to_state
from specialist_rebuild_v2.common import ACTION_FIELDS, fixed_action_grid


def _state(
    field_at_camera: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    setup: OpticalSetup,
) -> dict[str, float]:
    intensity, sensor_x, sensor_y = _extract_sensor_region(
        field_at_camera,
        grid_x,
        grid_y,
        setup,
    )
    metrics = compute_metrics(intensity, sensor_x, sensor_y)
    return metrics_to_state(metrics, setup)


def simulate_fixed_action_grid(
    setup: OpticalSetup,
) -> list[dict[str, Any]]:
    """Return simulator states in the canonical 81-action order.

    The ordinary simulator recomputes source-to-lens propagation for every
    action. In this fixed grid, that propagation is identical for all actions.
    There are only nine distinct lens positions and nine camera positions.
    This function therefore performs:

    - one source-to-lens propagation;
    - nine lens-to-camera propagations;
    - 81 inexpensive sensor extractions.

    The physical equations, grid, aperture, nearest-neighbour sensor
    extraction, and moment measurement are unchanged.
    """

    propagate = _BACKENDS.get(
        setup.propagation_backend,
        _BACKENDS["fresnel_numpy"],
    )
    wavelength = setup.source.wavelength
    source_field, grid_x, grid_y, spacing = gaussian_source_field(setup)
    at_lens = propagate(
        source_field,
        spacing,
        setup.laser_to_lens,
        wavelength,
    )

    actions = fixed_action_grid()
    results: list[dict[str, Any]] = []
    cached_camera_fields: dict[tuple[float, float], np.ndarray] = {}
    for action in actions:
        lens_key = (
            float(action["lens_x_delta_mm"]),
            float(action["lens_y_delta_mm"]),
        )
        field_at_camera = cached_camera_fields.get(lens_key)
        if field_at_camera is None:
            lens_setup = copy.deepcopy(setup)
            lens_setup.lens.x_offset += lens_key[0] * 1e-3
            lens_setup.lens.y_offset += lens_key[1] * 1e-3
            after_lens = apply_thin_lens(
                at_lens,
                grid_x,
                grid_y,
                lens_setup,
            )
            field_at_camera = propagate(
                after_lens,
                spacing,
                lens_setup.effective_camera_distance,
                wavelength,
            )
            cached_camera_fields[lens_key] = field_at_camera

        candidate_setup = copy.deepcopy(setup)
        candidate_setup.lens.x_offset += lens_key[0] * 1e-3
        candidate_setup.lens.y_offset += lens_key[1] * 1e-3
        candidate_setup.camera.x_offset += (
            float(action["camera_x_delta_mm"]) * 1e-3
        )
        candidate_setup.camera.y_offset += (
            float(action["camera_y_delta_mm"]) * 1e-3
        )
        state = _state(
            field_at_camera,
            grid_x,
            grid_y,
            candidate_setup,
        )
        results.append(
            {
                "action": {
                    field: float(action[field]) for field in ACTION_FIELDS
                },
                "state": state,
            }
        )
    if len(results) != 81:
        raise AssertionError("cached simulator must return exactly 81 actions")
    return results

