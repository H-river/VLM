from __future__ import annotations

import unittest

from optics_understanding_sft.build_intermediate_evidence_v6 import derive_records, tool_arguments
from optics_understanding_sft.decision_tools import (
    CONTROL_TOOL,
    SUFFICIENCY_TOOL,
    run_tool,
    select_minimum_motion_action,
    threshold_completion_directions,
)
from optics_understanding_sft.evaluate_intermediate_evidence_v6 import evaluate


def source_record(task: str) -> dict:
    common = {
        "group_id": "group",
        "split": "train",
        "provenance": {"match_group_id": f"pair-{task}", "scenario_seed": 7},
    }
    if task == "constrained_intervention":
        return {
            **common,
            "example_id": "control",
            "task_type": task,
            "prompt_inputs": {
                "actuator_constraints": {
                    "active_actuator": "lens_x_delta_mm",
                    "allowed_values_mm": [-1.0, 0.0, 1.0],
                    "success_tolerance_px": 0.2,
                },
                "candidate_action_trials": [
                    {"action": {"lens_x_delta_mm": 1.0}, "measured_residual_px": 0.1},
                    {"action": {"lens_x_delta_mm": -1.0}, "measured_residual_px": 0.1},
                    {"action": {"lens_x_delta_mm": 0.0}, "measured_residual_px": 1.0},
                ],
            },
            "target": {
                "status": "feasible",
                "answer": {
                    "control_plan": {"lens_x_delta_mm": -1.0},
                    "expected_residual_px": 0.1,
                    "best_achievable_residual_px": None,
                },
            },
        }
    return {
        **common,
        "example_id": "sufficiency",
        "task_type": task,
        "prompt_inputs": {
            "hidden_action_field": "lens_x_delta_mm",
            "questioned_output": "centroid_x_direction",
            "direction_threshold_px": 1.0,
            "compatible_completion_trials": [
                {"hidden_value_mm": -0.1, "measured_delta_px": -1.2},
                {"hidden_value_mm": 0.1, "measured_delta_px": 0.5},
            ],
        },
        "target": {
            "status": "insufficient_information",
            "answer": {
                "centroid_x_direction": None,
                "missing_fields": ["lens_x_delta_mm"],
                "nonidentifiable_output": "centroid_x_direction",
                "visible_conflicting_witness": [-0.1, 0.1],
            },
        },
    }


class DecisionToolTests(unittest.TestCase):
    def test_direction_threshold_is_inclusive_no_change(self) -> None:
        result = threshold_completion_directions([-1.0001, -1.0, 1.0, 1.0001], 1.0)
        self.assertEqual(
            result["per_trial_directions"],
            ["decrease", "no_change", "no_change", "increase"],
        )
        self.assertEqual(result["observed_direction_set"], ["decrease", "no_change", "increase"])
        self.assertEqual(result["conflicting_pair_indices"], [0, 1])

    def test_control_selection_uses_motion_residual_then_allowed_order(self) -> None:
        result = select_minimum_motion_action(
            [0.1, 0.1, 1.0], [1.0, -1.0, 0.0], 0.2, [2, 0, 1]
        )
        self.assertEqual(result["successful_action_indices"], [0, 1])
        self.assertEqual(result["selected_index"], 1)
        self.assertEqual(result["best_residual_index"], 1)

    def test_control_rejects_invalid_allowed_order(self) -> None:
        with self.assertRaises(ValueError):
            select_minimum_motion_action([0.1, 0.2], [0.0, 1.0], 0.2, [0, 0])

    def test_tool_arguments_preserve_displayed_order(self) -> None:
        arguments = tool_arguments(source_record("constrained_intervention"))
        self.assertEqual(arguments["candidate_residuals_px"], [0.1, 0.1, 1.0])
        self.assertEqual(arguments["allowed_order_indices"], [2, 0, 1])

    def test_derived_records_supervise_all_three_stages(self) -> None:
        records = derive_records(source_record("information_sufficiency"), "v6")
        self.assertEqual([record["stage"] for record in records], [
            "tool_choice", "tool_call_construction", "tool_result_interpretation"
        ])
        self.assertEqual(records[0]["target"]["tool_name"], SUFFICIENCY_TOOL)
        evidence = records[2]["target"]["intermediate_evidence"]
        self.assertEqual(evidence["observed_direction_set"], ["decrease", "no_change"])
        self.assertEqual(evidence["conflicting_pair_indices"], [0, 1])

    def test_registered_calls_execute(self) -> None:
        self.assertTrue(run_tool(CONTROL_TOOL, tool_arguments(source_record("constrained_intervention")))["feasible_within_tolerance"])
        self.assertFalse(run_tool(SUFFICIENCY_TOOL, tool_arguments(source_record("information_sufficiency")))["all_directions_agree"])

    def test_oracle_predictions_pass_every_evaluation_gate(self) -> None:
        records = derive_records(source_record("constrained_intervention"), "v6") + derive_records(source_record("information_sufficiency"), "v6")
        predictions = [{"example_id": record["example_id"], "parsed_json": record["target"]} for record in records]
        config = {"promotion_gates": {
            "schema_valid_rate": 1.0,
            "tool_choice_accuracy": 1.0,
            "tool_arguments_exact_match": 1.0,
            "tool_execution_result_match": 1.0,
            "intermediate_evidence_exact_match": 1.0,
            "interpretation_status_macro_f1": 1.0,
            "interpretation_answer_exact_match": 1.0,
            "pair_joint_status_accuracy": 0.0,
            "end_to_end_exact_match": 1.0,
        }}
        _, summary, gates = evaluate(config, records, predictions)
        self.assertEqual(summary["end_to_end_exact_match"], 1.0)
        self.assertTrue(gates["passed"])


if __name__ == "__main__":
    unittest.main()
