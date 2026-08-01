from __future__ import annotations

import json
import unittest

from optics_understanding_sft.build_schema_repair_curriculum import (
    CONTRACT_MARKER,
    transform_focused,
)


def qwen_row(task: str, target: dict) -> dict:
    return {
        "example_id": f"example_{task}",
        "group_id": "group",
        "task_type": task,
        "match_group_id": "match",
        "images": [],
        "prompt": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Prompt\n{CONTRACT_MARKER}\n{{\"action\": {{}}, \"answer\": {{}}, \"status\": \"x\"}}",
                    }
                ],
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(target)}],
            }
        ],
    }


class SchemaRepairCurriculumTests(unittest.TestCase):
    def test_control_evidence_is_nested_and_serialized_before_status(self) -> None:
        row = qwen_row(
            "constrained_intervention",
            {
                "action": {
                    "actuator": "lens_x_delta_mm",
                    "signed_movement_mm": -0.01,
                    "predicted_residual_px": 1.2,
                    "executable_valid": True,
                },
                "status": "feasible",
                "answer": {
                    "control_plan": {
                        "lens_x_delta_mm": -0.01,
                        "lens_y_delta_mm": 0.0,
                        "camera_x_delta_mm": 0.0,
                        "camera_y_delta_mm": 0.0,
                    },
                    "expected_residual_px": 1.2,
                },
            },
        )
        transformed = transform_focused(row)
        text = transformed["completion"][0]["content"][0]["text"]
        target = json.loads(text)
        self.assertEqual(list(target), ["answer", "status"])
        self.assertEqual(
            list(target["answer"])[:4],
            ["actuator", "signed_movement_mm", "predicted_residual_px", "executable_valid"],
        )
        prompt = transformed["prompt"][0]["content"][-1]["text"]
        self.assertNotIn('\n  "action": {', prompt.split(CONTRACT_MARKER, 1)[-1])

    def test_sufficiency_answer_object_is_serialized_before_status(self) -> None:
        row = qwen_row(
            "information_sufficiency",
            {
                "status": "insufficient_information",
                "answer": {
                    "missing_fields": ["lens_x_delta_mm"],
                    "compatible_completions": [],
                    "answer_changing_completions": [],
                },
            },
        )
        transformed = transform_focused(row)
        target = json.loads(transformed["completion"][0]["content"][0]["text"])
        self.assertEqual(list(target), ["answer", "status"])


if __name__ == "__main__":
    unittest.main()
