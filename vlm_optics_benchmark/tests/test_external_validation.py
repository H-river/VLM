from __future__ import annotations

from vlm_optics_benchmark.external_validation import derive_arms, selected_gain


def _case(stratum: str, index: int) -> dict:
    return {"stratum": stratum, "case_id": f"suite_00_{index:04d}"}


def test_preregistered_gain_assignment_is_deterministic() -> None:
    assert selected_gain(_case("one_step_reachable_interior", 3)) == 1.0
    expected = [0.5, 0.75, 1.25, 1.5, 0.5]
    assert [selected_gain(_case("multi_step_reachable_interior", i)) for i in range(5)] == expected
    assert [selected_gain(_case("reachable_boundary_or_clipping", i)) for i in range(5)] == expected


def test_fixed4_and_sequential_rule_are_exact_trace_prefixes() -> None:
    trace = []
    costs = [4.0, 3.0, 2.0, 1.5, 1.2, 0.9]
    before = 5.0
    for index, cost in enumerate(costs, start=1):
        trace.append(
            {
                "actual_target_cost": cost,
                "before_target_cost": before,
                "actual_step_audit": {
                    "step_saturated": False,
                    "absolute_position_saturated": False,
                },
            }
        )
        before = cost
    row = {
        "case_id": "c",
        "group_id": "g",
        "stratum": "multi_step_reachable_interior",
        "evaluator_only_true_gain": 0.75,
        "initial_normalized_distance": 5.0,
        "constraint_violation_count": 0,
        "probe_record": {"saturation_count": 0},
        "trace": trace,
    }
    fixed, sequential = derive_arms(
        row,
        {"minimum_last_step_improvement": 0.25, "maximum_final_distance": "infinity"},
    )
    assert fixed["control_steps"] == 4
    assert fixed["final_normalized_distance"] == 1.5
    assert sequential["control_steps"] == 6
    assert sequential["strict_success"] is True

