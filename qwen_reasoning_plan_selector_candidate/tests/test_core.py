from __future__ import annotations

import numpy as np
import pytest

from continuous_control_v12.contracts import Bounds
from vlm_optics_benchmark.visual_anomalies import moment_metrics

from qwen_reasoning_plan_selector_candidate.core import (
    ALL_DOFS,
    FIXED_CONTROLLER_CONFIG,
    PLAN_NAMES,
    PLAN_BANK_REVISION,
    PlanCEM,
    build_plan_bank,
    canonical_hash,
    measure_capture,
    _phase_policy,
)


class LinearBatchModel:
    def predict(self, setup_context, positions, current_metrics, actions):
        actions = np.asarray(actions, dtype=np.float64)
        current = np.asarray(current_metrics, dtype=np.float64)
        delta = np.zeros((len(actions), 5), dtype=np.float64)
        delta[:, 0] = 100.0 * actions[:, 0]
        delta[:, 1] = 100.0 * actions[:, 1]
        delta[:, 2] = 100.0 * actions[:, 2]
        delta[:, 3] = 100.0 * actions[:, 3]
        delta[:, 4] = 20.0 * (actions[:, 2] + actions[:, 3])
        return {
            "predicted_next_metrics": current[None, :] + delta,
            "uncertainty": np.zeros((len(actions), 5), dtype=np.float64),
            "auxiliary_predictions": {
                "clipping_fraction": np.zeros(len(actions)),
                "camera_boundary_probability": np.zeros(len(actions)),
                "actuator_limit_probability": np.zeros(len(actions)),
                "captured_power": np.ones(len(actions)),
            },
        }


@pytest.fixture()
def bounds() -> Bounds:
    return Bounds(
        action_low=np.asarray([-0.05, -0.05, -0.02, -0.02]),
        action_high=np.asarray([0.05, 0.05, 0.02, 0.02]),
        position_low=np.asarray([-3.0, -3.0, -3.0, -3.0]),
        position_high=np.asarray([3.0, 3.0, 3.0, 3.0]),
    )


def test_plan_bank_forbids_gain_and_budget_selection() -> None:
    bank = build_plan_bank()
    assert tuple(bank) == PLAN_NAMES
    hashes = {name: canonical_hash(FIXED_CONTROLLER_CONFIG) for name in bank}
    assert len(set(hashes.values())) == 1
    for spec in bank.values():
        payload = spec.to_dict()
        assert not ({"gain", "action_bound", "population", "horizon"} & set(payload))
    assert PLAN_BANK_REVISION == "revision1_mandatory_specialist_first_step"


def test_revised_specialist_plans_execute_mandatory_first_stage() -> None:
    measured = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0])
    target = measured.copy()
    reference = measured.copy()
    bank = build_plan_bank()
    for name in ("centroid_then_full", "shape_then_full", "primary_spot_then_full"):
        phase0 = _phase_policy(bank[name], step=0, measured=measured, target=target, reference=reference, safe_subset=ALL_DOFS)
        phase1 = _phase_policy(bank[name], step=1, measured=measured, target=target, reference=reference, safe_subset=ALL_DOFS)
        assert phase0[3] is False
        assert phase1[3] is True


def test_nonlocked_controller_config_is_rejected(bounds: Bounds) -> None:
    changed = dict(FIXED_CONTROLLER_CONFIG)
    changed["population"] += 1
    with pytest.raises(ValueError, match="locked fixed-gain"):
        PlanCEM(
            model=LinearBatchModel(),
            setup_context={},
            bounds=bounds,
            config=changed,
            seed=1,
        )


def test_metric_weights_and_actuator_mask_change_executed_search(bounds: Bounds) -> None:
    kwargs = {
        "positions": np.zeros(4),
        "current_metrics": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]),
        "target_metrics": np.asarray([5.0, 0.0, 2.0, 0.0, 1.0]),
        "tolerance_reference": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]),
    }
    centroid = PlanCEM(
        model=LinearBatchModel(), setup_context={}, bounds=bounds,
        config=FIXED_CONTROLLER_CONFIG, seed=90210,
    ).plan(
        **kwargs,
        metric_weights=(1.0, 1.0, 0.05, 0.05, 0.05),
        active_dofs=ALL_DOFS,
    )
    shape = PlanCEM(
        model=LinearBatchModel(), setup_context={}, bounds=bounds,
        config=FIXED_CONTROLLER_CONFIG, seed=90210,
    ).plan(
        **kwargs,
        metric_weights=(0.05, 0.05, 1.0, 1.0, 1.0),
        active_dofs=ALL_DOFS,
    )
    assert centroid["selected_action"] != shape["selected_action"]
    lens_only = PlanCEM(
        model=LinearBatchModel(), setup_context={}, bounds=bounds,
        config=FIXED_CONTROLLER_CONFIG, seed=90210,
    ).plan(
        **kwargs,
        metric_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        active_dofs=("lens_x", "lens_y"),
    )
    assert lens_only["selected_action"]["camera_x_delta_mm"] == 0.0
    assert lens_only["selected_action"]["camera_y_delta_mm"] == 0.0
    assert lens_only["active_dofs"] == ["lens_x", "lens_y"]


def test_primary_spot_measurement_recomputes_from_observed_reflection() -> None:
    y, x = np.indices((128, 128), dtype=np.float64)
    clean = 100.0 * np.exp(-0.5 * (((x - 52.0) / 8.0) ** 2 + ((y - 60.0) / 10.0) ** 2))
    clean_moments = moment_metrics(clean)
    capture = {
        "intensity": clean.astype(np.float32),
        "metrics": dict(
            zip(
                ("centroid_x_px", "centroid_y_px", "sigma_x_px", "sigma_y_px", "peak_intensity"),
                clean_moments,
                strict=True,
            )
        ),
    }
    anomaly = {"k": 2.75, "amplitude": 0.4, "width_ratio": 1.0, "angle_radians": 0.0}
    standard, _, standard_evidence = measure_capture(
        capture, family="width_relative_reflection", anomaly=anomaly, source="standard"
    )
    primary, _, primary_evidence = measure_capture(
        capture, family="width_relative_reflection", anomaly=anomaly, source="primary_spot"
    )
    standard_error = float(np.linalg.norm(standard[:4] - clean_moments[:4]))
    primary_error = float(np.linalg.norm(primary[:4] - clean_moments[:4]))
    assert primary_error < 0.10 * standard_error
    assert standard_evidence["specialist"]["method"] == "standard_full_image_moments"
    assert primary_evidence["specialist"]["components"] == 2
    assert primary_evidence["specialist"]["source"] == "observed_image_only"
