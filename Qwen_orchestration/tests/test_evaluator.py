"""Metric tests for generated orchestration decisions."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from Qwen_orchestration.runtime.constrained_decoding import build_constraint
from Qwen_orchestration.runtime.formatting import format_outcome
from Qwen_orchestration.runtime.normalization import normalize_nonexecuting_decision
from Qwen_orchestration.scripts.evaluate_qwen import normalize_predictions, score
from Qwen_orchestration.scripts.select_checkpoint import select


def ready_decision() -> dict:
    return {
        "schema_version": "qwen_orchestration_decision_v1",
        "status": "ready",
        "task_type": "forward_prediction",
        "route_name": "predict_forward_from_state_v1",
        "arguments": {
            "setup": {
                "wavelength_nm": 532.0,
                "beam_waist_mm": 1.0,
                "power_w": 0.1,
                "lens_focal_length_mm": 50.0,
                "lens_aperture_mm": 25.0,
                "source_to_lens_mm": 100.0,
                "lens_to_camera_mm": 75.0,
                "lens_x_offset_mm": 0.0,
                "lens_y_offset_mm": 0.0,
                "camera_x_offset_mm": 0.0,
                "camera_y_offset_mm": 0.0,
                "pixel_size_um": 5.5,
            },
            "current_beam_state": {
                "centroid_x_px": 64.0,
                "centroid_y_px": 64.0,
                "sigma_x_px": 8.0,
                "sigma_y_px": 9.0,
                "peak_intensity": 0.8,
            },
            "action": {
                "lens_x_delta_mm": 0.1,
                "lens_y_delta_mm": -0.1,
                "camera_x_delta_mm": 0.0,
                "camera_y_delta_mm": 0.1,
            },
        },
        "image_roles": {},
        "missing_fields": [],
        "clarification_question": None,
    }


class Stage2MetricTest(unittest.TestCase):
    def test_ready_metrics_use_group_and_numeric_denominators(self) -> None:
        target = ready_decision()
        canonical = {
            "case": {
                "example_id": "case",
                "target_decision": target,
            }
        }
        prediction = {
            "example_id": "case",
            "images": [],
            "parsed_json": copy.deepcopy(target),
        }
        metrics = score([prediction], canonical, "stage2", Path("."))
        self.assertEqual(metrics["registry_valid_ready_call_rate"], 1.0)
        self.assertEqual(metrics["required_argument_group_exact_accuracy"], 1.0)
        self.assertEqual(metrics["numeric_value_unit_exact_accuracy"], 1.0)
        self.assertEqual(
            metrics["stage2_denominators"]["required_argument_groups"], 3
        )
        self.assertEqual(
            metrics["stage2_denominators"]["numeric_values_with_unit_paths"], 21
        )

        altered = copy.deepcopy(prediction)
        altered["parsed_json"]["arguments"]["setup"]["wavelength_nm"] = 633.0
        metrics = score([altered], canonical, "stage2", Path("."))
        self.assertEqual(metrics["registry_valid_ready_call_rate"], 1.0)
        self.assertEqual(metrics["ready_arguments_exact_accuracy"], 0.0)
        self.assertAlmostEqual(
            metrics["required_argument_group_exact_accuracy"], 2 / 3
        )
        self.assertAlmostEqual(metrics["numeric_value_unit_exact_accuracy"], 20 / 21)

    def test_unparseable_outputs_remain_in_target_denominators(self) -> None:
        ready = ready_decision()
        clarification = {
            "schema_version": "qwen_orchestration_decision_v1",
            "status": "needs_clarification",
            "task_type": "forward_prediction",
            "route_name": None,
            "arguments": {},
            "image_roles": {},
            "missing_fields": ["arguments.setup"],
            "clarification_question": "What are the optical setup values?",
        }
        unsupported = {
            "schema_version": "qwen_orchestration_decision_v1",
            "status": "unsupported",
            "task_type": None,
            "route_name": None,
            "arguments": {},
            "image_roles": {},
            "missing_fields": [],
            "clarification_question": None,
        }
        targets = {
            "ready": ready,
            "clarification": clarification,
            "unsupported": unsupported,
        }
        canonical = {
            example_id: {
                "example_id": example_id,
                "target_decision": target,
            }
            for example_id, target in targets.items()
        }
        predictions = [
            {
                "example_id": example_id,
                "images": [],
                "parsed_json": None,
            }
            for example_id in targets
        ]

        metrics = score(predictions, canonical, "stage2", Path("."))

        self.assertEqual(metrics["schema_valid_rate"], 0.0)
        self.assertEqual(metrics["ready_route_exact_accuracy"], 0.0)
        self.assertEqual(metrics["registry_valid_ready_call_rate"], 0.0)
        self.assertEqual(metrics["required_argument_group_exact_accuracy"], 0.0)
        self.assertEqual(metrics["numeric_value_unit_exact_accuracy"], 0.0)
        self.assertEqual(metrics["clarification_recall"], 0.0)
        self.assertEqual(metrics["unsupported_recall"], 0.0)
        self.assertEqual(metrics["ready_prediction_on_unsupported"], 0.0)
        self.assertEqual(metrics["stage2_denominators"]["ready_records"], 1)
        self.assertEqual(
            metrics["stage2_denominators"]["required_argument_groups"], 3
        )
        self.assertEqual(
            metrics["stage2_denominators"]["numeric_values_with_unit_paths"], 21
        )
        self.assertEqual(
            metrics["stage2_denominators"]["clarification_records"], 1
        )


class CheckpointSelectionTest(unittest.TestCase):
    def test_stage1_rejects_failed_gate_and_prefers_earlier_tie(self) -> None:
        passing_metrics = {
            "schema_valid_rate": 1.0,
            "status_exact_accuracy": 0.99,
            "ready_route_exact_accuracy": 0.98,
            "clarification_recall": 0.97,
            "unsupported_recall": 0.96,
            "ready_prediction_on_unsupported": 0.0,
        }
        reports = [
            {
                "adapter_path": "/run/checkpoint-250",
                "metrics": dict(passing_metrics),
            },
            {
                "adapter_path": "/run/checkpoint-500",
                "metrics": dict(passing_metrics),
            },
            {
                "adapter_path": "/run/checkpoint-750",
                "metrics": {
                    **passing_metrics,
                    "ready_route_exact_accuracy": 1.0,
                    "unsupported_recall": 0.5,
                },
            },
        ]
        assessments, selected = select(reports, "stage1")
        self.assertEqual([item["passed"] for item in assessments], [True, True, False])
        self.assertIsNotNone(selected)
        self.assertEqual(selected["checkpoint_step"], 250)


class ConstrainedDecodingConfigurationTest(unittest.TestCase):
    def test_constraint_is_disabled_when_not_configured(self) -> None:
        callback, metadata = build_constraint({"generation": {}}, object())
        self.assertIsNone(callback)
        self.assertIsNone(metadata)


class NonexecutingNormalizationTest(unittest.TestCase):
    def test_ready_decision_is_never_changed(self) -> None:
        decision = ready_decision()
        normalized, changed = normalize_nonexecuting_decision(decision)
        self.assertIs(normalized, decision)
        self.assertFalse(changed)

    def test_clarification_synonym_and_extra_field_are_canonicalized(self) -> None:
        decision = {
            "schema_version": "qwen_orchestration_decision_v1",
            "status": "needs_clarification",
            "task_type": "beam_measurement",
            "route_name": "invented_route",
            "arguments": {},
            "image_roles": ["current"],
            "missing_fields": ["image_calibration"],
            "clarification_question": "What is the calibration?",
            "orchestration_role": "strategy_selector",
        }
        normalized, changed = normalize_nonexecuting_decision(decision)
        self.assertTrue(changed)
        self.assertIsNone(normalized["task_type"])
        self.assertIsNone(normalized["route_name"])
        self.assertEqual(normalized["image_roles"], {})
        self.assertNotIn("orchestration_role", normalized)

    def test_resumed_predictions_retain_normalization_count(self) -> None:
        prediction = {
            "example_id": "clarification",
            "parsed_json": {
                "schema_version": "qwen_orchestration_decision_v1",
                "status": "needs_clarification",
                "task_type": None,
                "route_name": None,
                "arguments": {},
                "image_roles": {},
                "missing_fields": ["arguments.setup"],
                "clarification_question": "What setup values are available?",
            },
            "normalization_applied": True,
        }
        normalized, count = normalize_predictions(
            [prediction],
            {"orchestration": {"normalize_nonexecuting_decisions": True}},
        )
        self.assertEqual(count, 1)
        self.assertEqual(normalized, [prediction])


class DeterministicFormattingTest(unittest.TestCase):
    def test_ready_result_preserves_numeric_json(self) -> None:
        outcome = {
            "status": "ready",
            "route_name": "measure_beam_profile_v1",
            "result": {"beam_state": {"centroid_x_px": 12.3456789012345}},
        }
        rendered = format_outcome(outcome)
        self.assertIn("12.3456789012345", rendered)
        self.assertIn("measure_beam_profile_v1", rendered)

    def test_clarification_is_returned_verbatim(self) -> None:
        question = "What is the wavelength in nanometres?"
        self.assertEqual(
            format_outcome(
                {
                    "status": "needs_clarification",
                    "clarification_question": question,
                }
            ),
            question,
        )


if __name__ == "__main__":
    unittest.main()
