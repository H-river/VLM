from __future__ import annotations

import copy
import unittest

from optics_understanding_sft.check_promotion import ANCHOR_TASKS, STATUS_TASKS, candidate_gates


class PromotionGateTests(unittest.TestCase):
    def fixture(self) -> tuple[dict, dict, list[dict]]:
        baseline = {"per_task": {task: {"task_score": 0.5} for task in ANCHOR_TASKS}}
        summary = {
            "macro_task_score": 0.7,
            "schema_valid_rate": 0.95,
            "per_task": {
                **{task: {"task_score": 0.55} for task in ANCHOR_TASKS},
                **{task: {"status_macro_f1": 0.8} for task in STATUS_TASKS},
            },
        }
        details = []
        for label in ("feasible", "infeasible_within_limits"):
            details += [
                {
                    "task_type": "constrained_intervention",
                    "target_status": label,
                    "predicted_status": label,
                    "simulator_outcome_success": 1.0 if label == "feasible" else 0.0,
                }
                for _ in range(5)
            ]
        return baseline, summary, details

    def test_all_frozen_gates_pass_for_balanced_candidate(self) -> None:
        baseline, summary, details = self.fixture()
        self.assertTrue(candidate_gates(baseline, summary, details)["passed"])

    def test_constant_control_prediction_fails_promotion(self) -> None:
        baseline, summary, details = self.fixture()
        collapsed = copy.deepcopy(details)
        for row in collapsed:
            row["predicted_status"] = "feasible"
        result = candidate_gates(baseline, summary, collapsed)
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["control_infeasible_recall_at_least_0_60"])


if __name__ == "__main__":
    unittest.main()
