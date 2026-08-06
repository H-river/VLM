"""Regression and contract tests for the frozen specialist runtime."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from typing import Any

from Qwen_orchestration.runtime.dispatcher import OrchestrationRuntime, validate_decision
from Qwen_orchestration.runtime.errors import ContractError
from Qwen_orchestration.runtime.numerics import SETUP_FIELDS, fixed_action_grid
from Qwen_orchestration.runtime.prompt_contract import (
    DECISION_SYSTEM_CONTRACT,
    apply_decision_contract,
)
from Qwen_orchestration.runtime.specialists import run_specialist


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "optics_understanding_sft/direction_inverse_v1/data/v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def setup_only(value: dict[str, Any]) -> dict[str, Any]:
    return {field: value[field] for field in SETUP_FIELDS}


class PromptContractTest(unittest.TestCase):
    def test_contract_is_prepended_once_without_mutating_input(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "route this"}],
            }
        ]
        contracted = apply_decision_contract(messages)
        repeated = apply_decision_contract(contracted)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(contracted[0]["role"], "system")
        self.assertEqual(contracted[0]["content"][0]["text"], DECISION_SYSTEM_CONTRACT)
        self.assertEqual(repeated, contracted)


class FrozenSpecialistRegressionTest(unittest.TestCase):
    def test_direction_reproduces_recorded_prediction(self) -> None:
        row = read_jsonl(DATA / "direction/all_fields/val.jsonl")[0]
        expected = read_jsonl(
            REPO
            / "optics_understanding_sft/direction_inverse_v1/results/"
            "direction_small_v1/details.jsonl"
        )[0]
        inputs = row["prompt_inputs"]
        result = run_specialist(
            "predict_direction_from_state_v1",
            {
                "setup": setup_only(inputs["setup"]),
                "current_beam_state": inputs["current_beam_state"],
                "action": inputs["action"],
            },
            {},
        )
        self.assertEqual(result["directions"], expected["prediction"])
        for field, classes in expected["probabilities"].items():
            for class_name, probability in classes.items():
                self.assertAlmostEqual(
                    result["probabilities"][field][class_name], probability, places=6
                )

    def test_forward_reproduces_recorded_prediction(self) -> None:
        row = read_jsonl(
            REPO / "optics_understanding_sft/data/native_hybrid_v1/val.jsonl"
        )[0]
        expected = read_jsonl(
            REPO
            / "optics_understanding_sft/direction_inverse_v1/results/"
            "forward_hybrid_v1/details.jsonl"
        )[0]
        inputs = row["inputs"]
        result = run_specialist(
            "predict_forward_from_state_v1",
            {
                "setup": setup_only(inputs["setup"]),
                "current_beam_state": inputs["current_beam_state"],
                "action": inputs["action"],
            },
            {},
        )
        self.assertEqual(result["directions"], expected["predicted_directions"])
        for field, value in expected["predicted_change"].items():
            self.assertAlmostEqual(result["change"][field], value, places=5)

    def test_numeric_inverse_reproduces_recorded_selection(self) -> None:
        example_id = "invv1_case_000231_infeasible_within_limits_numeric"
        row = next(
            item
            for item in read_jsonl(DATA / "inverse/canonical/val.jsonl")
            if item["example_id"] == example_id
        )
        expected = next(
            item
            for item in read_jsonl(
                REPO
                / "optics_understanding_sft/direction_inverse_v1/results/"
                "inverse_ensemble_v1/details.jsonl"
            )
            if item["example_id"] == example_id
        )
        inputs = row["prompt_inputs"]
        result = run_specialist(
            "select_inverse_action_from_states_v1",
            {
                "setup": setup_only(inputs["setup"]),
                "current_beam_state": inputs["current_beam_state_A"],
                "desired_beam_state": inputs["desired_beam_state_B"],
            },
            {},
        )
        self.assertEqual(result["predicted_status"], expected["predicted_status"])
        self.assertEqual(result["selected_index"], expected["selected_index"])
        self.assertEqual(result["selected_action"], expected["selected_action"])
        self.assertEqual(result["action_grid_size"], 81)

    def test_visual_inverse_reproduces_meter_and_selection(self) -> None:
        row = next(
            item
            for item in read_jsonl(DATA / "inverse/canonical/val.jsonl")
            if item["task_type"] == "inverse_action_visual"
        )
        expected = next(
            item
            for item in read_jsonl(
                REPO
                / "optics_understanding_sft/direction_inverse_v1/results/"
                "visual_pipeline_sensor_v1/details.jsonl"
            )
            if item["example_id"] == row["example_id"]
        )
        inputs = row["prompt_inputs"]
        calibration = {
            key: inputs["image_calibration"][key]
            for key in (
                "linear_intensity_low",
                "linear_intensity_high",
                "gamma",
                "source_sensor_resolution_px",
            )
        }
        result = run_specialist(
            "select_inverse_action_from_images_v1",
            {
                "setup": setup_only(inputs["setup"]),
                "image_calibration": calibration,
            },
            {
                "current_beam": DATA / inputs["images"][0],
                "desired_beam": DATA / inputs["images"][1],
            },
        )
        self.assertEqual(result["predicted_status"], expected["predicted_status"])
        self.assertEqual(result["selected_index"], expected["selected_index"])
        self.assertEqual(result["selected_action"], expected["selected_action"])
        for field, value in expected["measured_A"].items():
            self.assertAlmostEqual(
                result["measured_current_beam_state"][field], value, places=9
            )
        for field, value in expected["measured_B"].items():
            self.assertAlmostEqual(
                result["measured_desired_beam_state"][field], value, places=9
            )

    def test_wrappers_do_not_import_optical_simulator(self) -> None:
        import sys

        self.assertFalse(any(name == "optical_sim" or name.startswith("optical_sim.")
                             for name in sys.modules))

    def test_action_grid_order_is_frozen(self) -> None:
        grid = fixed_action_grid()
        self.assertEqual(len(grid), 81)
        self.assertEqual(
            grid[0],
            {
                "lens_x_delta_mm": -0.05,
                "lens_y_delta_mm": -0.05,
                "camera_x_delta_mm": -0.02,
                "camera_y_delta_mm": -0.02,
            },
        )
        self.assertTrue(all(value == 0.0 for value in grid[40].values()))


class DecisionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        row = read_jsonl(
            REPO / "optics_understanding_sft/data/native_hybrid_v1/val.jsonl"
        )[0]["inputs"]
        self.decision = {
            "schema_version": "qwen_orchestration_decision_v1",
            "status": "ready",
            "task_type": "forward_prediction",
            "route_name": "predict_forward_from_state_v1",
            "arguments": {
                "setup": setup_only(row["setup"]),
                "current_beam_state": row["current_beam_state"],
                "action": row["action"],
            },
            "image_roles": {},
            "missing_fields": [],
            "clarification_question": None,
        }

    def test_valid_decision_passes_registry_contract(self) -> None:
        self.assertEqual(validate_decision(self.decision, {}), {})

    def test_valid_decision_executes_exactly_one_specialist(self) -> None:
        result = OrchestrationRuntime().dispatch(self.decision)
        self.assertTrue(result["executed"])
        self.assertEqual(result["route_name"], "predict_forward_from_state_v1")
        self.assertEqual(set(result["result"]["change"]), {
            "centroid_x_px", "centroid_y_px", "sigma_x_px", "sigma_y_px",
            "peak_intensity",
        })

    def test_task_route_mismatch_is_rejected(self) -> None:
        self.decision["task_type"] = "direction_prediction"
        with self.assertRaisesRegex(ContractError, "does not match"):
            validate_decision(self.decision, {}, require_enabled=False)

    def test_missing_argument_group_is_rejected(self) -> None:
        del self.decision["arguments"]["action"]
        with self.assertRaisesRegex(ContractError, "argument groups differ"):
            validate_decision(self.decision, {}, require_enabled=False)

    def test_extra_argument_group_is_rejected(self) -> None:
        self.decision["arguments"]["desired_beam_state"] = dict(
            self.decision["arguments"]["current_beam_state"]
        )
        with self.assertRaisesRegex(ContractError, "argument groups differ"):
            validate_decision(self.decision, {}, require_enabled=False)

    def test_non_finite_number_is_rejected(self) -> None:
        self.decision["arguments"]["action"]["lens_x_delta_mm"] = math.nan
        with self.assertRaisesRegex(ContractError, "non-finite"):
            validate_decision(self.decision, {}, require_enabled=False)

    def test_specialist_rejects_extra_numeric_field(self) -> None:
        arguments = self.decision["arguments"]
        arguments["setup"]["sensor_resolution_px"] = [1024, 1024]
        with self.assertRaisesRegex(ContractError, "fields differ"):
            run_specialist("predict_forward_from_state_v1", arguments, {})

    def test_clarification_never_executes(self) -> None:
        decision = {
            "schema_version": "qwen_orchestration_decision_v1",
            "status": "needs_clarification",
            "task_type": "forward_prediction",
            "route_name": None,
            "arguments": {},
            "image_roles": {},
            "missing_fields": ["action"],
            "clarification_question": "What action should be evaluated?",
        }
        self.assertEqual(validate_decision(decision, {}), {})


if __name__ == "__main__":
    unittest.main()
