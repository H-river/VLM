from __future__ import annotations

import numpy as np
import pytest

from continuous_control_v12.contracts import Bounds, assert_no_q_star
from continuous_control_v12.mpc import CEMMPC
from continuous_control_v12.run_mpc_h1_h3_diagnosis import (
    simulate_counterfactual_sequence,
)


def _bounds() -> Bounds:
    return Bounds(
        action_low=np.asarray([-0.05, -0.05, -0.02, -0.02]),
        action_high=np.asarray([0.05, 0.05, 0.02, 0.02]),
        position_low=np.full(4, -3.0),
        position_high=np.full(4, 3.0),
    )


class LinearPredictor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        positions: np.ndarray,
        metrics: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        del positions
        self.calls += 1
        output = metrics.copy()
        output[:4] += action
        output[4] += action.sum()
        return (
            output,
            np.zeros(5, dtype=np.float64),
            {
                "clipping_probability": np.asarray([0.0]),
                "boundary_probability": np.asarray([0.0]),
            },
        )


def _planner(
    predictor: LinearPredictor,
    *,
    horizon: int,
    audit: bool,
) -> CEMMPC:
    return CEMMPC(
        bounds=_bounds(),
        predictor=predictor,
        config={
            "horizon": horizon,
            "population": 24,
            "elites": 6,
            "cem_iterations": 2,
            "mean_error_weight": 0.15,
            "movement_weight": 0.02,
            "limit_penalty": 10.0,
            "boundary_penalty": 1.0,
            "uncertainty_weight": 0.1,
            "candidate_audit_top_k": 5 if audit else 0,
            "candidate_audit_reference_k": 5 if audit else 0,
        },
        seed=90210,
    )


def _plan(planner: CEMMPC) -> dict:
    return planner.plan(
        positions_mm=np.zeros(4),
        current_metrics=np.zeros(5),
        target_metrics=np.asarray([2.0, -2.0, 3.0, -3.0, 20.0]),
        allowed_dofs=["lens_x", "lens_y", "camera_x", "camera_y"],
        tolerance_reference=np.asarray([0.0, 0.0, 1.0, 1.0, 100.0]),
    )


def test_horizon_controls_exact_rollout_call_count_and_audit_size() -> None:
    h1_predictor = LinearPredictor()
    h3_predictor = LinearPredictor()
    h1 = _plan(_planner(h1_predictor, horizon=1, audit=True))
    h3 = _plan(_planner(h3_predictor, horizon=3, audit=True))
    assert h1["rollout_backend_calls"] == 24 * 2
    assert h3["rollout_backend_calls"] == 24 * 2 * 3
    assert h1_predictor.calls == h1["rollout_backend_calls"]
    assert h3_predictor.calls == h3["rollout_backend_calls"]
    assert len(h1["planned_effective_sequence"]) == 1
    assert len(h3["planned_effective_sequence"]) == 3
    assert len(h1["predicted_rollout_metrics"]) == 1
    assert len(h3["predicted_rollout_metrics"]) == 3
    assert len(h1["candidate_audit"]) == 10
    assert len(h3["candidate_audit"]) == 10


def test_backend_identity_and_candidate_audit_do_not_change_cem_action() -> None:
    first_backend = LinearPredictor()
    second_backend = LinearPredictor()
    audited = _planner(first_backend, horizon=3, audit=True)
    plain = _planner(second_backend, horizon=3, audit=False)
    first_audited = _plan(audited)
    first_plain = _plan(plain)
    assert first_audited["selected_effective_action"] == first_plain[
        "selected_effective_action"
    ]
    assert first_audited["planned_effective_sequence"] == first_plain[
        "planned_effective_sequence"
    ]
    second_audited = _plan(audited)
    second_plain = _plan(plain)
    assert second_audited["planned_effective_sequence"] == second_plain[
        "planned_effective_sequence"
    ]


def test_counterfactual_sequence_clones_start_and_updates_absolute_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    original = start.copy()

    def fake_simulate(
        setup_context,
        positions_mm,
        simulator_fixed,
        base_config_path,
        bounds,
    ):
        del setup_context, simulator_fixed, base_config_path, bounds
        values = np.asarray(list(positions_mm.values()), dtype=np.float64)
        return {
            "metrics": {
                "centroid_x_px": float(values[0]),
                "centroid_y_px": float(values[1]),
                "sigma_x_px": float(values[2]),
                "sigma_y_px": float(values[3]),
                "peak_intensity": float(values.sum()),
            },
            "auxiliary": {
                "captured_power": 1.0,
                "clipping_fraction": 0.0,
                "camera_boundary_indicator": False,
                "actuator_limit_indicator": False,
                "phase_descriptor": None,
                "simulator_valid": True,
            },
        }

    monkeypatch.setattr(
        "continuous_control_v12.run_mpc_h1_h3_diagnosis.simulate_state",
        fake_simulate,
    )
    sequence = [
        [0.05, 0.0, 0.0, 0.0],
        [0.0, -0.05, 0.02, 0.0],
        [0.0, 0.0, 0.0, -0.02],
    ]
    replay = simulate_counterfactual_sequence(
        setup_context={},
        simulator_fixed={},
        base_config_path="unused",
        bounds=_bounds(),
        start_positions=start,
        effective_sequence=sequence,
    )
    assert np.array_equal(start, original)
    assert replay["simulator_calls"] == 3
    assert np.allclose(
        list(replay["final_positions_mm"].values()),
        original + np.sum(sequence, axis=0),
    )


def test_privilege_guard_rejects_q_goal_aliases() -> None:
    for key in (
        "q_goal",
        "q_goal_mm",
        "goal_positions",
        "goal_positions_mm",
    ):
        with pytest.raises(ValueError, match="oracle actuator target leaked"):
            assert_no_q_star({"planner_input": {key: [0.0] * 4}})
