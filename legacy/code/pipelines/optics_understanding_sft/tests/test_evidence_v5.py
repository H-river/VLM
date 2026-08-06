from __future__ import annotations

import unittest

from optics_understanding_sft.audit_evidence_v5_probe import (
    control_solution,
    direction,
    pair_invariant,
)
from optics_understanding_sft.build_evidence_v5_probe import conflicting_witness, sufficiency_trials
from optics_understanding_sft.build_evidence_v5a import split_groups, target_free
from optics_understanding_sft.evaluate_evidence_v5a import pair_metrics, witness_valid


class EvidenceV5Tests(unittest.TestCase):
    def test_fresh_split_is_deterministic_and_group_disjoint(self) -> None:
        group_ids = [f"group_{index}" for index in range(6)]
        counts = {"train": 3, "dev": 2, "confirmation": 1}
        assignments_a, groups_a = split_groups(group_ids, counts, 42)
        assignments_b, groups_b = split_groups(group_ids, counts, 42)
        self.assertEqual(assignments_a, assignments_b)
        self.assertEqual(groups_a, groups_b)
        self.assertEqual(set().union(*(set(groups) for groups in groups_a.values())), set(group_ids))
        self.assertEqual(sum(len(groups) for groups in groups_a.values()), len(group_ids))

    def test_confirmation_projection_removes_target(self) -> None:
        record = {"example_id": "x", "target": {"status": "answerable"}}
        projected = target_free(record)
        self.assertNotIn("target", projected)
        self.assertIn("target", record)

    def test_conflicting_witness_uses_visible_values(self) -> None:
        trials = [
            {"hidden_value_mm": -0.1, "measured_direction": "increase"},
            {"hidden_value_mm": 0.0, "measured_direction": "increase"},
            {"hidden_value_mm": 0.1, "measured_direction": "decrease"},
        ]
        self.assertEqual(conflicting_witness(trials), [-0.1, 0.1])
        self.assertIsNone(conflicting_witness(trials[:2]))

    def test_gate_witness_must_reference_different_visible_directions(self) -> None:
        record = {
            "prompt_inputs": {
                "compatible_completion_trials": [
                    {"hidden_value_mm": -0.1, "measured_direction": "increase"},
                    {"hidden_value_mm": 0.0, "measured_direction": "increase"},
                    {"hidden_value_mm": 0.1, "measured_direction": "decrease"},
                ]
            }
        }
        self.assertTrue(
            witness_valid(
                record, {"answer": {"visible_conflicting_witness": [-0.1, 0.1]}}
            )
        )
        self.assertFalse(
            witness_valid(
                record, {"answer": {"visible_conflicting_witness": [-0.1, 0.0]}}
            )
        )

    def test_pair_gate_requires_both_members_correct(self) -> None:
        records = [
            {
                "example_id": "a",
                "task_type": "information_sufficiency",
                "target": {"status": "answerable"},
                "provenance": {"match_group_id": "pair"},
            },
            {
                "example_id": "b",
                "task_type": "information_sufficiency",
                "target": {"status": "insufficient_information"},
                "provenance": {"match_group_id": "pair"},
            },
        ]
        metrics = pair_metrics(
            records,
            {
                "a": {"status": "answerable"},
                "b": {"status": "answerable"},
            },
        )["information_sufficiency"]
        self.assertEqual(metrics["both_statuses_correct_rate"], 0.0)
        self.assertEqual(metrics["prediction_changes_with_pair_rate"], 0.0)

    def test_direction_threshold_is_inclusive_no_change(self) -> None:
        self.assertEqual(direction(1.0001, 1.0), "increase")
        self.assertEqual(direction(1.0, 1.0), "no_change")
        self.assertEqual(direction(-1.0, 1.0), "no_change")
        self.assertEqual(direction(-1.0001, 1.0), "decrease")

    def test_sufficiency_scaffold_is_derived_from_raw_centroids(self) -> None:
        record = {
            "prompt_inputs": {
                "hidden_action_field": "lens_x_delta_mm",
                "compatible_hidden_values_mm": [-0.1, 0.0, 0.1],
                "current_observation": {"centroid_x_px": 10.0},
            }
        }
        private = {
            "replay_specs": [
                {
                    "name": "completion_0",
                    "action": {"lens_x_delta_mm": -0.1},
                    "expected_state": {"centroid_x_px": 8.0},
                },
                {
                    "name": "completion_1",
                    "action": {"lens_x_delta_mm": 0.0},
                    "expected_state": {"centroid_x_px": 10.5},
                },
                {
                    "name": "completion_2",
                    "action": {"lens_x_delta_mm": 0.1},
                    "expected_state": {"centroid_x_px": 12.0},
                },
            ]
        }
        trials = sufficiency_trials(record, private)
        self.assertEqual([trial["measured_delta_px"] for trial in trials], [-2.0, 0.5, 2.0])
        self.assertEqual(
            [trial["measured_direction"] for trial in trials],
            ["decrease", "no_change", "increase"],
        )

    def test_control_solver_uses_shuffled_visible_evidence(self) -> None:
        record = {
            "example_id": "control",
            "prompt_inputs": {
                "actuator_constraints": {
                    "active_actuator": "lens_x_delta_mm",
                    "allowed_values_mm": [-1.0, 0.0, 1.0],
                    "success_tolerance_px": 0.2,
                },
                "target_observation": {"centroid_x_px": 5.0, "centroid_y_px": 5.0},
                "candidate_action_trials": [
                    {
                        "action": {"lens_x_delta_mm": 1.0},
                        "measured_centroid_x_px": 5.1,
                        "measured_centroid_y_px": 5.0,
                    },
                    {
                        "action": {"lens_x_delta_mm": -1.0},
                        "measured_centroid_x_px": 8.0,
                        "measured_centroid_y_px": 5.0,
                    },
                    {
                        "action": {"lens_x_delta_mm": 0.0},
                        "measured_centroid_x_px": 6.0,
                        "measured_centroid_y_px": 5.0,
                    },
                ],
            },
        }
        solution = control_solution(record)
        self.assertEqual(solution["status"], "feasible")
        self.assertEqual(solution["control_plan"], {"lens_x_delta_mm": 1.0})
        self.assertAlmostEqual(solution["expected_residual_px"], 0.1)

    def test_control_pair_invariant_removes_target_dependent_residuals(self) -> None:
        base = {
            "task_type": "constrained_intervention",
            "prompt_inputs": {
                "target_observation": {"centroid_x_px": 1.0},
                "candidate_action_trials": [
                    {
                        "action": {"lens_x_delta_mm": 0.0},
                        "measured_centroid_x_px": 2.0,
                        "measured_residual_px": 1.0,
                    }
                ],
            },
        }
        paired = {
            **base,
            "prompt_inputs": {
                "target_observation": {"centroid_x_px": 4.0},
                "candidate_action_trials": [
                    {
                        "action": {"lens_x_delta_mm": 0.0},
                        "measured_centroid_x_px": 2.0,
                        "measured_residual_px": 2.0,
                    }
                ],
            },
        }
        self.assertEqual(pair_invariant(base), pair_invariant(paired))


if __name__ == "__main__":
    unittest.main()
