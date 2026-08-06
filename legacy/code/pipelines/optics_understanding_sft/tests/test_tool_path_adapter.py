from __future__ import annotations

import unittest

from optics_understanding_sft.decision_tools import CONTROL_TOOL, SUFFICIENCY_TOOL
from optics_understanding_sft.tool_path_adapter import (
    CONTROL_SOURCE_MAP,
    SUFFICIENCY_SOURCE_MAP,
    build_tool_arguments,
    compact_decision,
    materialize_final_answer,
    resolve_path,
    run_mapped_tool,
)


class ToolPathAdapterTests(unittest.TestCase):
    def test_resolve_path_rejects_missing_keys(self) -> None:
        self.assertEqual(resolve_path({"a": {"b": 2}}, "a.b"), 2)
        with self.assertRaises(ValueError):
            resolve_path({"a": {}}, "a.b")

    def test_control_mapping_builds_ordered_literal_arguments(self) -> None:
        evidence = {
            "actuator_constraints": {
                "active_actuator": "lens_x_delta_mm",
                "allowed_values_mm": [-1.0, 0.0, 1.0],
                "success_tolerance_px": 0.2,
            },
            "candidate_action_trials": [
                {"action": {"lens_x_delta_mm": 1.0}, "measured_residual_px": 0.1},
                {"action": {"lens_x_delta_mm": -1.0}, "measured_residual_px": 0.3},
                {"action": {"lens_x_delta_mm": 0.0}, "measured_residual_px": 0.4},
            ],
        }
        arguments = build_tool_arguments(CONTROL_TOOL, evidence, CONTROL_SOURCE_MAP)
        self.assertEqual(arguments["candidate_residuals_px"], [0.1, 0.3, 0.4])
        self.assertEqual(arguments["active_actuator_motions_mm"], [1.0, -1.0, 0.0])
        self.assertEqual(arguments["allowed_order_indices"], [2, 0, 1])
        self.assertEqual(
            run_mapped_tool(CONTROL_TOOL, evidence, CONTROL_SOURCE_MAP)["selected_index"], 0
        )

    def test_sufficiency_mapping_thresholds_without_llm_arithmetic(self) -> None:
        evidence = {
            "direction_threshold_px": 1.0,
            "compatible_completion_trials": [
                {"measured_delta_px": -1.2},
                {"measured_delta_px": 0.2},
            ],
        }
        arguments = build_tool_arguments(
            SUFFICIENCY_TOOL, evidence, SUFFICIENCY_SOURCE_MAP
        )
        self.assertEqual(arguments["measured_deltas_px"], [-1.2, 0.2])
        result = run_mapped_tool(SUFFICIENCY_TOOL, evidence, SUFFICIENCY_SOURCE_MAP)
        self.assertEqual(result["observed_direction_set"], ["decrease", "no_change"])

    def test_mapping_rejects_invented_shortcuts(self) -> None:
        with self.assertRaises(ValueError):
            build_tool_arguments(
                CONTROL_TOOL,
                {},
                {"action_trial_index": "candidate_action_trials.0"},
            )

    def test_compact_control_decision_and_materialization(self) -> None:
        evidence = {
            "candidate_action_trials": [
                {"action": {"lens_x_delta_mm": 0.0}, "measured_residual_px": 0.4},
                {"action": {"lens_x_delta_mm": 0.1}, "measured_residual_px": 0.1},
            ]
        }
        result = {
            "successful_action_indices": [1],
            "selected_index": 1,
            "best_residual_index": 1,
            "feasible_within_tolerance": True,
        }
        self.assertEqual(compact_decision(CONTROL_TOOL, result)["status"], "feasible")
        final = materialize_final_answer(CONTROL_TOOL, result, evidence)
        self.assertEqual(final["answer"]["control_plan"], {"lens_x_delta_mm": 0.1})
        self.assertEqual(final["answer"]["expected_residual_px"], 0.1)

    def test_compact_sufficiency_decision_and_materialization(self) -> None:
        evidence = {
            "hidden_action_field": "lens_x_delta_mm",
            "questioned_output": "centroid_x_direction",
            "compatible_completion_trials": [
                {"hidden_value_mm": -0.1},
                {"hidden_value_mm": 0.2},
            ],
        }
        result = {
            "per_trial_directions": ["decrease", "increase"],
            "observed_direction_set": ["decrease", "increase"],
            "conflicting_pair_indices": [0, 1],
            "all_directions_agree": False,
        }
        decision = compact_decision(SUFFICIENCY_TOOL, result)
        self.assertEqual(decision["status"], "insufficient_information")
        final = materialize_final_answer(SUFFICIENCY_TOOL, result, evidence)
        self.assertEqual(final["answer"]["visible_conflicting_witness"], [-0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
