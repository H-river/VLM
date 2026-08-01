from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from continuous_control_v12.contracts import (
    Bounds,
    apply_action,
    normalized_distance,
    position_dict,
)
from continuous_control_v12.mpc import (
    CEMMPC,
    run_closed_loop,
    simulator_predictor,
)
from continuous_control_v12.reachability import discrete_81_oracle
from continuous_control_v12.simulator import (
    default_simulator_fixed,
    sample_group_setup,
    simulate_state,
)
from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_sft.physics.sim_adapter import (
    Action,
    apply_action_to_setup,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (REPO_ROOT / "continuous_control_v12/config_v12.json").read_text()
)
SEMANTICS_CONFIG = json.loads(
    (
        REPO_ROOT
        / "continuous_control_v12/config_v12_semantics_v1.json"
    ).read_text()
)
BASE = str((REPO_ROOT / CONFIG["simulator"]["base_config"]).resolve())


def small_setup() -> tuple[Bounds, dict, dict, dict]:
    bounds = Bounds.from_config(CONFIG)
    fixed = default_simulator_fixed(
        BASE, grid_size=128, sensor_resolution=[128, 128]
    )
    context, positions = sample_group_setup(
        "ordinary", "test_group", 9102, fixed, bounds
    )
    return bounds, fixed, context, positions


def test_mm_to_m_adapter_conversion_is_explicit() -> None:
    setup = setup_from_dict(load_yaml(BASE))
    moved = apply_action_to_setup(
        setup,
        Action(
            lens_x_delta_mm=0.05,
            lens_y_delta_mm=0.0,
            camera_x_delta_mm=0.02,
            camera_y_delta_mm=0.0,
        ),
    )
    assert np.isclose(moved.lens.x_offset - setup.lens.x_offset, 0.05e-3)
    assert np.isclose(
        moved.camera.x_offset - setup.camera.x_offset, 0.02e-3
    )


def test_no_op_preserves_positions_and_simulator_metrics() -> None:
    bounds, fixed, context, positions = small_setup()
    action = np.zeros(4)
    next_positions = apply_action(positions, action, bounds)
    assert np.array_equal(
        next_positions,
        np.asarray(
            [
                positions["lens_x_mm"],
                positions["lens_y_mm"],
                positions["camera_x_mm"],
                positions["camera_y_mm"],
            ]
        ),
    )
    before = simulate_state(context, positions, fixed, BASE, bounds)
    after = simulate_state(
        context, position_dict(next_positions), fixed, BASE, bounds
    )
    assert before["metrics"] == after["metrics"]


def test_one_step_target_has_zero_distance_at_generating_q_star() -> None:
    bounds, fixed, context, positions = small_setup()
    current = simulate_state(context, positions, fixed, BASE, bounds)
    action = np.asarray([0.05, 0.0, 0.0, 0.0])
    q_star = apply_action(positions, action, bounds)
    target = simulate_state(
        context, position_dict(q_star), fixed, BASE, bounds
    )
    replay = simulate_state(
        context, position_dict(q_star), fixed, BASE, bounds
    )
    assert (
        normalized_distance(
            replay["metrics"], target["metrics"], current["metrics"]
        )
        <= 1e-12
    )


def test_simulator_oracle_mpc_reduces_distance_and_stays_legal() -> None:
    bounds, fixed, context, positions = small_setup()
    current = simulate_state(context, positions, fixed, BASE, bounds)
    q_star = apply_action(
        positions, np.asarray([0.05, 0.0, 0.0, 0.0]), bounds
    )
    target = simulate_state(
        context, position_dict(q_star), fixed, BASE, bounds
    )
    predictor = simulator_predictor(
        setup_context=context,
        simulator_fixed=fixed,
        base_config_path=BASE,
        bounds=bounds,
    )
    planner = CEMMPC(
        bounds=bounds,
        predictor=predictor,
        config={
            "horizon": 1,
            "population": 12,
            "elites": 3,
            "cem_iterations": 1,
            "mean_error_weight": 0.0,
            "movement_weight": 0.0,
            "limit_penalty": 10.0,
            "boundary_penalty": 0.0,
            "uncertainty_weight": 0.0,
        },
        seed=123,
    )
    episode = run_closed_loop(
        planner=planner,
        setup_context=context,
        simulator_fixed=fixed,
        initial_positions_mm=positions,
        initial_metrics=current["metrics"],
        target_metrics=target["metrics"],
        allowed_dofs=["lens_x", "lens_y", "camera_x", "camera_y"],
        base_config_path=BASE,
        bounds=bounds,
        max_steps=1,
    )
    assert episode["final_normalized_distance"] < episode[
        "initial_normalized_distance"
    ]
    assert episode["success"]
    final = np.asarray(list(episode["final_positions_mm"].values()))
    assert np.all(final <= bounds.position_high + 1e-12)
    assert np.all(final >= bounds.position_low - 1e-12)
    assert episode["illegal_actions"] == 0
    assert episode["planner_exploitation_events"] == 0
    assert episode["trace"][0]["predicted_improvement"] > 0.0
    assert episode["trace"][0]["actual_simulator_improvement"] > 0.0


def test_discrete_oracle_uses_corrected_semantics_for_corrected_target() -> None:
    bounds = Bounds.from_config(SEMANTICS_CONFIG)
    fixed = default_simulator_fixed(
        BASE,
        grid_size=128,
        grid_extent_mm=6.25,
        sensor_resolution=[64, 64],
        semantics=SEMANTICS_CONFIG["simulator"]["semantics"],
    )
    context, positions = sample_group_setup(
        "ordinary", "corrected_oracle_test", 4901, fixed, bounds
    )
    current = simulate_state(context, positions, fixed, BASE, bounds)
    result = discrete_81_oracle(
        setup_context=context,
        simulator_fixed=fixed,
        current_positions_mm=positions,
        current_metrics=current["metrics"],
        target_metrics=current["metrics"],
        base_config_path=BASE,
        bounds=bounds,
    )
    assert result["best_distance"] <= 1e-12
    assert all(
        value == 0.0 for value in result["best_legacy_action"].values()
    )
