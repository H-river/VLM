from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import torch

from optics_sft.scripts.train_qwen25vl_qlora import (
    decision_weighted_loss,
    token_sequence_mask,
    token_sequence_weights,
)
from optics_understanding_sft.build_hard_pairs_v4 import compact_target, replace_contract
from optics_understanding_sft.check_hard_pair_diagnostic import check
from optics_understanding_sft.analyze_hard_pair_v4_failure import (
    local_turn_count,
    replacement_deltas,
)
from optics_understanding_sft.core import write_jsonl
from optics_understanding_sft.evaluate import schema_valid


class HardPairsV4Tests(unittest.TestCase):
    def test_failure_analysis_counts_nonmonotonic_response_turns(self) -> None:
        self.assertEqual(local_turn_count([3.0, 1.0, 2.0, 0.0, 4.0]), 3)
        self.assertEqual(local_turn_count([4.0, 3.0, 2.0, 1.0]), 0)

    def test_failure_analysis_handles_multiple_counterbalanced_replacements(self) -> None:
        self.assertEqual(
            replacement_deltas([-0.01, 0.01, 0.02, 0.04, 0.06], [-0.06, 0.01, 0.02, 0.04, 0.02]),
            [0.049999999999999996, 0.039999999999999994],
        )

    def test_control_target_is_compacted_to_dev_schema(self) -> None:
        record = {
            "task_type": "constrained_intervention",
            "prompt": "prefix Task output contract (all fields shown; use null when not applicable):\n{}",
            "target": {
                "action": {"actuator": "lens_x_delta_mm"},
                "status": "feasible",
                "answer": {
                    "control_plan": {
                        "lens_x_delta_mm": 0.01,
                        "lens_y_delta_mm": 0.0,
                        "camera_x_delta_mm": 0.0,
                        "camera_y_delta_mm": 0.0,
                    },
                    "expected_residual_px": 1.0,
                },
            },
        }
        compact_target(record)
        self.assertEqual(list(record["target"]), ["status", "answer"])
        self.assertNotIn("action", record["target"])
        self.assertTrue(schema_valid(record["task_type"], record["target"])[0])

    def test_sufficiency_target_is_compacted_without_evidence_arrays(self) -> None:
        record = {
            "task_type": "information_sufficiency",
            "prompt": "prefix Task output contract (all fields shown; use null when not applicable):\n{}",
            "target": {
                "status": "insufficient_information",
                "answer": {
                    "missing_fields": ["lens_x_delta_mm"],
                    "nonidentifiable_output": "centroid_x_direction",
                    "compatible_completions": [{"hidden_value_mm": 0.0}],
                    "answer_changing_completions": [{"hidden_value_mm": 0.0}],
                },
            },
        }
        compact_target(record)
        self.assertNotIn("compatible_completions", record["target"]["answer"])
        self.assertTrue(schema_valid(record["task_type"], record["target"])[0])

    def test_contract_replacement_removes_old_tail(self) -> None:
        prompt = "question\nTask output contract (all fields shown; use null when not applicable):\nold"
        replaced = replace_contract(prompt, {"status": "a | b", "answer": {}})
        self.assertNotIn("old", replaced)
        self.assertIn('"status": "a | b"', replaced)

    def test_decision_token_mask_marks_exact_subsequences(self) -> None:
        labels = torch.tensor(
            [
                [-100, 10, 20, 21, 30, 40, 41, 42],
                [-100, 20, 99, 21, 40, 41, 42, 50],
            ]
        )
        mask = token_sequence_mask(torch, labels, [[20, 21], [40, 41, 42]])
        expected = torch.tensor(
            [
                [False, False, True, True, False, True, True, True],
                [False, False, False, False, True, True, True, False],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_weighted_loss_emphasizes_bad_decision_tokens(self) -> None:
        labels = torch.tensor([[-100, 0, 1, 2, 0]])
        logits = torch.zeros((1, 5, 3), dtype=torch.float32)
        logits[0, 0, 0] = 5.0
        logits[0, 1, 0] = 5.0  # deliberately wrong for label 1
        logits[0, 2, 0] = 5.0  # deliberately wrong for label 2
        logits[0, 3, 0] = 5.0
        ordinary, matched = decision_weighted_loss(torch, logits, labels, [[1, 2]], 2.0)
        weighted, _ = decision_weighted_loss(torch, logits, labels, [[1, 2]], 8.0)
        self.assertEqual(matched, 2)
        self.assertGreater(float(weighted), float(ordinary))

    def test_overlapping_label_sequences_receive_equal_span_mass(self) -> None:
        labels = torch.tensor([[10, 11, -100, -100, -100], [20, 10, 11, 21, 22]])
        weights, covered = token_sequence_weights(
            torch,
            labels,
            [[10, 11], [20, 10, 11, 21, 22]],
            12.0,
        )
        self.assertAlmostEqual(float(weights[0][covered[0]].sum()), 12.0)
        self.assertAlmostEqual(float(weights[1][covered[1]].sum()), 12.0)

    def test_hard_pair_gate_requires_complete_paired_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_dir = root / "result"
            result_dir.mkdir()
            summary = {
                "schema_valid_rate": 1.0,
                "per_task": {
                    "information_sufficiency": {"status_macro_f1": 1.0},
                    "constrained_intervention": {"status_macro_f1": 1.0},
                },
            }
            (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            records = []
            details = []
            specifications = (
                ("control_pair", "constrained_intervention", "feasible", 1.0),
                (
                    "control_pair",
                    "constrained_intervention",
                    "infeasible_within_limits",
                    0.0,
                ),
                ("suff_pair", "information_sufficiency", "answerable", 0.0),
                (
                    "suff_pair",
                    "information_sufficiency",
                    "insufficient_information",
                    0.0,
                ),
            )
            for index, (pair, task, status, simulator_success) in enumerate(specifications):
                example_id = f"example_{index}"
                records.append(
                    {
                        "example_id": example_id,
                        "provenance": {"match_group_id": pair},
                    }
                )
                details.append(
                    {
                        "example_id": example_id,
                        "task_type": task,
                        "target_status": status,
                        "predicted_status": status,
                        "simulator_outcome_success": simulator_success,
                    }
                )
            records_path = root / "records.jsonl"
            write_jsonl(records_path, records)
            write_jsonl(result_dir / "details.jsonl", details)
            self.assertTrue(check(result_dir, records_path)["passed"])


if __name__ == "__main__":
    unittest.main()
