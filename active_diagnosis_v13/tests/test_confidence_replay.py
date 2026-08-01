from active_diagnosis_v13.validate_confidence_replay import (
    _key,
    _trace_projection,
    _values_close,
)


def test_confidence_replay_key_accepts_prediction_and_control_schema() -> None:
    assert _key({"case_id": "case", "true_gain_evaluator_only": 0.75}) == (
        "case",
        0.75,
    )
    assert _key({"case_id": "case", "evaluator_only_true_gain": 1.25}) == (
        "case",
        1.25,
    )


def test_trace_projection_ignores_policy_identity_but_keeps_outcomes() -> None:
    row = {
        "trace": [
            {
                "command_mm": {"lens_x": 0.1},
                "predicted_next_metrics": {"centroid_x": 1.0},
                "observed_next_metrics": {"centroid_x": 1.1},
                "actual_target_cost": 0.5,
                "visible_plan_input": {"gain_belief": None},
            }
        ]
    }
    assert _trace_projection(row) == [
        {
            "command_mm": {"lens_x": 0.1},
            "predicted_next_metrics": {"centroid_x": 1.0},
            "observed_next_metrics": {"centroid_x": 1.1},
            "actual_target_cost": 0.5,
        }
    ]


def test_numeric_projection_tolerates_machine_roundoff_only() -> None:
    assert _values_close({"value": 1.0}, {"value": 1.0 + 5e-13})
    assert not _values_close({"value": 1.0}, {"value": 1.0 + 2e-12})
