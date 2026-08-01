from __future__ import annotations

from active_diagnosis_v13.analyze_control_step_budget import (
    _assert_replay,
    _switched_row,
)
from active_diagnosis_v13.analyze_control_horizon_curve import _cap
from active_diagnosis_v13.analyze_temporal_seed_statistics import _two_way_bootstrap
from active_diagnosis_v13.audit_boundary_candidate_coverage import _group_bootstrap
from active_diagnosis_v13.analyze_boundary_recovery_overlap import _partition
from active_diagnosis_v13.analyze_sequential_horizon_rule import _sequential_cap
from active_diagnosis_v13.run_boundary_candidate_coverage import _condition_uniform
from continuous_control_v12.contracts import Bounds
import numpy as np


def _step(before: float, after: float, command: float = 0.01) -> dict:
    return {
        "command_mm": {"lens_x_delta_mm": command},
        "predicted_next_metrics": {"centroid_x_px": 1.0},
        "observed_next_metrics": {"centroid_x_px": 2.0},
        "before_target_cost": before,
        "predicted_target_cost": after - 0.1,
        "actual_target_cost": after,
        "actual_step_audit": {
            "step_saturated": False,
            "absolute_position_saturated": False,
        },
    }


def _row(*, success: bool, distance: float, steps: list[dict]) -> dict:
    return {
        "strict_success": success,
        "final_normalized_distance": distance,
        "control_steps": len(steps),
        "trace": steps,
        "saturation_count": 0,
    }


def test_visible_continuation_rule_selects_only_productive_nearby_failure() -> None:
    baseline = _row(
        success=False,
        distance=3.0,
        steps=[_step(8.0, 7.0), _step(7.0, 6.0), _step(6.0, 5.0), _step(5.0, 3.0)],
    )
    treatment = _row(
        success=True,
        distance=0.8,
        steps=[*baseline["trace"], _step(3.0, 1.5), _step(1.5, 0.8)],
    )
    selected = _switched_row(
        baseline,
        treatment,
        minimum_last_improvement=1.0,
        maximum_final_distance=4.0,
    )
    assert selected["continuation_selected"] is True
    assert selected["strict_success"] is True
    rejected = _switched_row(
        baseline,
        treatment,
        minimum_last_improvement=2.5,
        maximum_final_distance=4.0,
    )
    assert rejected["continuation_selected"] is False
    assert rejected["strict_success"] is False


def test_replay_audit_compares_only_the_baseline_prefix() -> None:
    key = ("case", 0.5)
    baseline = _row(
        success=False,
        distance=3.0,
        steps=[_step(5.0, 4.0), _step(4.0, 3.0)],
    )
    treatment = _row(
        success=True,
        distance=0.8,
        steps=[*baseline["trace"], _step(3.0, 0.8)],
    )
    assert _assert_replay({key: baseline}, {key: treatment}) == []
    changed = _row(
        success=True,
        distance=0.8,
        steps=[_step(5.0, 4.1), *treatment["trace"][1:]],
    )
    assert _assert_replay({key: baseline}, {key: changed}) == ["case__g0.5"]


def test_horizon_cap_reconstructs_outcome_steps_and_saturation() -> None:
    steps = [
        _step(4.0, 2.0),
        _step(2.0, 1.2),
        _step(1.2, 0.8),
    ]
    steps[1]["actual_step_audit"]["absolute_position_saturated"] = True
    row = {
        **_row(success=True, distance=0.8, steps=steps),
        "case_id": "case",
        "group_id": "group",
        "stratum": "reachable_boundary_or_clipping",
        "evaluator_only_true_gain": 0.5,
        "initial_normalized_distance": 4.0,
        "probe_record": {"saturation_count": 1},
    }
    capped_one = _cap(row, 1)
    assert capped_one["strict_success"] is False
    assert capped_one["final_normalized_distance"] == 2.0
    assert capped_one["control_steps"] == 1
    assert capped_one["saturation_count"] == 1
    capped_two = _cap(row, 2)
    assert capped_two["strict_success"] is False
    assert capped_two["saturation_count"] == 2
    capped_three = _cap(row, 3)
    assert capped_three["strict_success"] is True
    assert capped_three["final_normalized_distance"] == 0.8
    assert capped_three["control_steps"] == 3
    assert capped_three["saturation_count"] == 2


def test_two_way_seed_group_bootstrap_is_deterministic_and_paired() -> None:
    values = np.asarray([[0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
    first = _two_way_bootstrap(values, seed=7, samples=200)
    second = _two_way_bootstrap(values, seed=7, samples=200)
    assert first == second
    assert first["estimate"] == 0.5
    assert first["planner_seeds"] == 2
    assert first["setup_groups"] == 2


def test_candidate_coverage_bootstrap_clusters_repeated_faults_by_group() -> None:
    values = [("a", 1.0), ("a", 1.0), ("b", -1.0)]
    first = _group_bootstrap(values, seed=11, samples=200)
    second = _group_bootstrap(values, seed=11, samples=200)
    assert first == second
    assert np.isclose(first["episode_estimate"], 1.0 / 3.0)
    assert first["group_mean_estimate"] == 0.0
    assert first["independent_groups"] == 2


def test_boundary_recovery_partition_is_exhaustive_and_disjoint() -> None:
    partition = _partition({"a", "b", "c", "d"}, {"a", "b"}, {"b", "c"})
    assert partition == {
        "both": ["b"],
        "temporal_only": ["a"],
        "candidate_only": ["c"],
        "neither": ["d"],
    }
    flattened = [value for values in partition.values() for value in values]
    assert sorted(flattened) == ["a", "b", "c", "d"]


def test_sequential_horizon_rule_rechecks_visible_improvement_each_step() -> None:
    steps = [
        _step(5.0, 4.0),
        _step(4.0, 3.0),
        _step(3.0, 2.5),
        _step(2.5, 2.0),
        _step(2.0, 1.5),
        _step(1.5, 1.6),
        _step(1.6, 0.8),
    ]
    row = {
        **_row(success=True, distance=0.8, steps=steps),
        "case_id": "case",
        "group_id": "group",
        "stratum": "multi_step_reachable_interior",
        "evaluator_only_true_gain": 0.5,
        "initial_normalized_distance": 5.0,
        "probe_record": None,
    }
    stopped = _sequential_cap(row, minimum_improvement=0.0, maximum_distance=float("inf"))
    assert stopped["control_steps"] == 6
    assert stopped["strict_success"] is False
    assert stopped["stopped_by_visible_rule"] is True
    permissive = _sequential_cap(
        row, minimum_improvement=-1.0, maximum_distance=float("inf")
    )
    assert permissive["control_steps"] == 7
    assert permissive["strict_success"] is True


def test_boundary_conditioning_biases_only_near_bound_axes_toward_interior() -> None:
    bounds = Bounds(
        action_low=np.full(4, -0.1),
        action_high=np.full(4, 0.1),
        position_low=np.full(4, -1.0),
        position_high=np.full(4, 1.0),
    )
    uniform = np.full(4, 0.25)
    position = np.asarray([-0.9, 0.9, 0.0, 0.0])
    transformed, axes = _condition_uniform(uniform, position, bounds)
    assert transformed[0] > uniform[0]
    assert transformed[1] < uniform[1]
    assert transformed[2] == uniform[2]
    assert transformed[3] == uniform[3]
    assert axes == [
        "lens_x_delta_mm:away_from_lower",
        "lens_y_delta_mm:away_from_upper",
    ]
