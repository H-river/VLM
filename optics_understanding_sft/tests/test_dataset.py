from __future__ import annotations

import copy
import random
import unittest
from pathlib import Path

from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_sft.physics.sim_adapter import simulate_and_measure
from optics_sft.scripts.train_qwen25vl_qlora import (
    prebuilt_chat_row_to_sft_example,
    prebuilt_smoke_rows,
)

from optics_understanding_sft.core import (
    TASK_TYPES,
    apply_action_dict,
    assign_tasks_to_groups,
    axis_action,
    centroid_distance,
    classify_effects,
    load_yaml,
    modified_setup_config,
    prompt_key_hits,
    qwen_completion,
    sample_setup_config,
)
from optics_understanding_sft.build_dataset import (
    assign_corrective_match_groups,
    constrained_intervention,
    controlled_status_schedule,
    diagnosis,
    information_sufficiency,
    prompt_text,
)
from optics_understanding_sft.core import read_jsonl
from optics_understanding_sft.evaluate import (
    TASKS,
    evaluate_rows,
    score_counterfactual_v2,
    score_setup_v2,
    set_f1,
)
from optics_understanding_sft.run_inference import extract_json_object


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_yaml(ROOT / "optics_understanding_sft/configs/pilot_v1.yaml")
BASE = load_sim_yaml(str(ROOT / "optical_sim/configs/base_config.yaml"))


class DatasetCoreTests(unittest.TestCase):
    def test_corrective_status_schedules_are_exact_and_actuator_balanced(self) -> None:
        control = controlled_status_schedule(
            "constrained_intervention", {"feasible": 8, "infeasible_within_limits": 8}
        )
        self.assertEqual(control.count("feasible"), 8)
        self.assertEqual(control.count("infeasible_within_limits"), 8)
        for actuator_index in range(4):
            labels = control[actuator_index::4]
            self.assertEqual(labels.count("feasible"), labels.count("infeasible_within_limits"))
        diagnosis_schedule = controlled_status_schedule(
            "diagnosis", {"unique": 4, "ambiguous": 4, "unsupported": 4}
        )
        for axis_index in range(2):
            self.assertEqual(
                {label: diagnosis_schedule[axis_index::2].count(label) for label in set(diagnosis_schedule)},
                {"unique": 2, "ambiguous": 2, "unsupported": 2},
            )

    def test_action_first_status_schedule_oversamples_feasible_per_actuator(self) -> None:
        control = controlled_status_schedule(
            "constrained_intervention", {"feasible": 24, "infeasible_within_limits": 8}
        )
        for actuator_index in range(4):
            labels = control[actuator_index::4]
            self.assertEqual(labels.count("feasible"), 6)
            self.assertEqual(labels.count("infeasible_within_limits"), 2)

    def test_qwen_completion_preserves_action_first_target_order(self) -> None:
        target = {
            "action": {"actuator": "lens_x_delta_mm"},
            "status": "feasible",
            "answer": {"control_plan": {}},
        }
        text = qwen_completion(target)[0]["content"][0]["text"]
        self.assertLess(text.index('"action"'), text.index('"status"'))
        self.assertLess(text.index('"status"'), text.index('"answer"'))

    def test_corrective_builders_hit_requested_statuses_and_form_match_groups(self) -> None:
        sampled, _ = sample_setup_config(BASE, CONFIG["simulation"], random.Random(91))
        base = simulate_and_measure(setup_from_dict(copy.deepcopy(sampled)))
        templates = load_yaml(ROOT / "optics_understanding_sft/prompts/templates.yaml")
        common = {
            "split": "train",
            "scenario_seed": 91,
            "config": sampled,
            "base": base,
            "templates": templates,
            "rng": random.Random(91),
            "label_cfg": CONFIG["labels"],
            "visual": False,
            "output_dir": ROOT / "optics_understanding_sft/data/smoke_fixture",
            "render_cfg": CONFIG["rendering"],
        }
        requests = [
            (information_sufficiency, 0, "answerable"),
            (information_sufficiency, 1, "insufficient_information"),
            (diagnosis, 0, "unique"),
            (diagnosis, 2, "ambiguous"),
            (diagnosis, 4, "unsupported"),
            (constrained_intervention, 0, "feasible"),
            (constrained_intervention, 4, "infeasible_within_limits"),
        ]
        master_records = []
        for builder, task_index, status in requests:
            record, private = builder(
                group_id=f"corrective_{builder.__name__}_{task_index}",
                task_index=task_index,
                desired_status=status,
                **common,
            )
            self.assertEqual(record["target"]["status"], status)
            record["provenance"]["status_schedule_index"] = task_index
            master_records.append({"record": record, "private_eval": private})
        counts = assign_corrective_match_groups(
            [{"group_id": "combined", "split": "train", "records": master_records}], "test_v2"
        )
        self.assertEqual(counts["train:information_sufficiency"], 1)
        self.assertEqual(counts["train:diagnosis"], 1)
        self.assertEqual(counts["train:constrained_intervention"], 1)
        self.assertTrue(all("match_group_id" in item["record"]["provenance"] for item in master_records))
        control_rows = [
            item["record"]
            for item in master_records
            if item["record"]["task_type"] == "constrained_intervention"
        ]
        baseline_errors = [
            centroid_distance(
                row["prompt_inputs"]["current_observation"],
                row["prompt_inputs"]["target_observation"],
            )
            for row in control_rows
        ]
        self.assertLessEqual(max(baseline_errors) - min(baseline_errors), 5.0)

    def test_task_assignment_is_exact_unique_and_deterministic(self) -> None:
        groups = [f"g{i}" for i in range(5)]
        counts = {
            "setup_interpretation": 3,
            "information_sufficiency": 3,
            "causal_effects": 3,
            "forward_prediction": 3,
            "diagnosis": 3,
            "constrained_intervention": 3,
            "counterfactual_reasoning": 2,
        }
        first = assign_tasks_to_groups(groups, counts, random.Random(7), 4)
        second = assign_tasks_to_groups(groups, counts, random.Random(7), 4)
        self.assertEqual(first, second)
        self.assertTrue(all(len(tasks) == len(set(tasks)) == 4 for tasks in first.values()))
        observed = {task: 0 for task in TASK_TYPES}
        for tasks in first.values():
            for task in tasks:
                observed[task] += 1
        self.assertEqual(observed, counts)

    def test_ood_sampling_moves_exactly_requested_parameter(self) -> None:
        iid_cfg, _ = sample_setup_config(BASE, CONFIG["simulation"], random.Random(10))
        ood_cfg, _ = sample_setup_config(
            BASE,
            CONFIG["simulation"],
            random.Random(10),
            ood_parameter="lens_focal_length_mm",
            ood_band=1,
        )
        nominal = float(BASE["lens"]["focal_length"])
        self.assertGreaterEqual(float(ood_cfg["lens"]["focal_length"]), nominal * 1.2)
        self.assertLessEqual(float(iid_cfg["lens"]["focal_length"]), nominal * 1.2)
        self.assertEqual(iid_cfg["source"]["wavelength"], ood_cfg["source"]["wavelength"])

    def test_setup_sampling_is_seed_reproducible(self) -> None:
        first, first_values = sample_setup_config(BASE, CONFIG["simulation"], random.Random(42))
        second, second_values = sample_setup_config(BASE, CONFIG["simulation"], random.Random(42))
        self.assertEqual(first, second)
        self.assertEqual(first_values, second_values)

    def test_causal_thresholds_cover_all_labels(self) -> None:
        before = {
            "centroid_x_px": 10.0,
            "centroid_y_px": 10.0,
            "sigma_x_px": 5.0,
            "sigma_y_px": 5.0,
            "peak_intensity": 1.0,
        }
        after = {
            "centroid_x_px": 12.0,
            "centroid_y_px": 8.0,
            "sigma_x_px": 6.0,
            "sigma_y_px": 5.0,
            "peak_intensity": 1.01,
        }
        effects = classify_effects(before, after, CONFIG["labels"])
        self.assertEqual(effects["centroid_x"], "increase")
        self.assertEqual(effects["centroid_y"], "decrease")
        self.assertEqual(effects["sigma_x"], "no_change")
        self.assertEqual(effects["peak_intensity"], "no_change")

    def test_prompt_policy_allows_observations_but_blocks_answers(self) -> None:
        safe = {"current_observation": {"centroid_x_px": 4.0}, "target_observation": {"centroid_x_px": 5.0}}
        self.assertEqual(prompt_key_hits(safe), [])
        self.assertTrue(prompt_key_hits({"after_state": {"centroid_x_px": 6.0}}))

    def test_prompt_contract_does_not_embed_target_status(self) -> None:
        prompt = prompt_text(
            "Solve the bounded control problem.",
            "constrained_intervention",
            {"actuator_constraints": {"allowed_values_mm": [-0.1, 0.0, 0.1]}},
        )
        self.assertIn("feasible | infeasible_within_limits", prompt)
        self.assertNotIn('"status": "feasible"', prompt)
        self.assertNotIn('"status": "infeasible_within_limits"', prompt)

    def test_counterfactual_changes_only_selected_parameter(self) -> None:
        changed = modified_setup_config(BASE, "lens_focal_length_mm", 1.1)
        self.assertAlmostEqual(changed["lens"]["focal_length"], BASE["lens"]["focal_length"] * 1.1)
        unchanged = copy.deepcopy(BASE)
        unchanged["lens"]["focal_length"] = changed["lens"]["focal_length"]
        self.assertEqual(changed, unchanged)

    def test_discrete_feasible_action_replays_to_target(self) -> None:
        setup = setup_from_dict(copy.deepcopy(BASE))
        action = axis_action("lens_x_delta_mm", 0.03)
        target = simulate_and_measure(apply_action_dict(setup, action))["state"]
        replayed = simulate_and_measure(apply_action_dict(setup, action))["state"]
        self.assertAlmostEqual(target["centroid_x_px"], replayed["centroid_x_px"], places=8)


class PrebuiltChatTests(unittest.TestCase):
    def test_text_only_prebuilt_record(self) -> None:
        row = {
            "example_id": "text",
            "images": [],
            "prompt": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": "{}"}]}],
        }
        example = prebuilt_chat_row_to_sft_example(row, ROOT)
        self.assertEqual(example["images"], [])

    def test_placeholder_mismatch_is_rejected(self) -> None:
        row = {
            "example_id": "bad",
            "images": [],
            "prompt": [{"role": "user", "content": [{"type": "image"}]}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": "{}"}]}],
        }
        with self.assertRaises(ValueError):
            prebuilt_chat_row_to_sft_example(row, ROOT)

    def test_smoke_selection_includes_text_and_visual(self) -> None:
        rows = [{"example_id": "a", "images": []}, {"example_id": "b", "images": ["x.png"]}]
        selected = prebuilt_smoke_rows(rows, 2)
        self.assertEqual({bool(row["images"]) for row in selected}, {False, True})


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = ROOT / "optics_understanding_sft/data/smoke_fixture"
        cls.records = read_jsonl(fixture / "canonical/train.jsonl")
        cls.master = read_jsonl(fixture / "master/cases.jsonl")

    def test_json_extraction_accepts_fenced_or_prefixed_object(self) -> None:
        parsed, error = extract_json_object('result:\n```json\n{"status":"answerable","answer":{}}\n```')
        self.assertIsNone(error)
        self.assertEqual(parsed["status"], "answerable")

    def test_set_f1_is_order_independent(self) -> None:
        self.assertEqual(set_f1(["a", "b"], ["b", "a"]), 1.0)
        self.assertAlmostEqual(set_f1(["a"], ["a", "b"]), 2 / 3)

    def test_oracle_predictions_score_one_for_every_task(self) -> None:
        predictions = [
            {
                "example_id": row["example_id"],
                "parsed_json": copy.deepcopy(row["target"]),
                "latency_seconds": 0.1,
                "input_tokens": 10,
                "output_tokens": 5,
            }
            for row in self.records
        ]
        details, summary = evaluate_rows(self.records, predictions, self.master)
        self.assertEqual(len(details), len(self.records))
        self.assertEqual(summary["macro_task_score"], 1.0)
        for task in TASKS:
            self.assertEqual(summary["per_task"][task]["task_score"], 1.0)

        _, summary_v2 = evaluate_rows(self.records, predictions, self.master, rubric_version="v2")
        self.assertEqual(summary_v2["macro_task_score"], 1.0)

    def test_v2_control_gates_wrong_feasibility_branch(self) -> None:
        record = next(
            row
            for row in self.records
            if row["task_type"] == "constrained_intervention" and row["target"]["status"] == "feasible"
        )
        prediction = {
            "example_id": record["example_id"],
            "parsed_json": {
                "status": "infeasible_within_limits",
                "answer": {
                    "control_plan": None,
                    "expected_residual_px": None,
                    "best_achievable_residual_px": 999.0,
                },
            },
        }
        details, _ = evaluate_rows([record], [prediction], self.master, rubric_version="v2")
        self.assertEqual(details[0]["task_score"], 0.0)

    def test_v2_setup_accepts_semantic_alias_and_excludes_hidden_adjustability(self) -> None:
        target = {
            "answer": {
                "component_order": ["gaussian_source", "thin_lens", "camera_sensor"],
                "adjustable_parameters": ["lens_x"],
                "total_source_to_sensor_mm": 500.0,
                "lens_focal_length_m": 0.05,
            }
        }
        prediction = {
            "answer": {
                "component_order": ["laser", "lens", "camera"],
                "adjustable_parameters": [],
                "total_source_to_sensor_mm": 500.0,
                "lens_focal_length_m": 0.05,
            }
        }
        metrics = score_setup_v2(prediction, target, {})
        self.assertEqual(metrics["component_order_semantic_exact"], 1.0)
        self.assertEqual(metrics["adjustable_parameters_f1"], 0.0)
        self.assertEqual(metrics["task_score"], 1.0)

    def test_v2_counterfactual_weights_numeric_prediction_at_sixty_percent(self) -> None:
        state = {
            "centroid_x_px": 0.0,
            "centroid_y_px": 0.0,
            "peak_intensity": 1.0,
            "sigma_x_px": 1.0,
            "sigma_y_px": 1.0,
        }
        target = {
            "answer": {
                "changed_parameter": "focal_length",
                "centroid_direction_preserved": True,
                "response_a": state,
                "response_b": state,
                "response_difference": state,
            }
        }
        wrong_state = {key: 100.0 for key in state}
        prediction = {
            "answer": {
                "changed_parameter": "focal_length",
                "centroid_direction_preserved": True,
                "response_a": wrong_state,
                "response_b": wrong_state,
                "response_difference": wrong_state,
            }
        }
        metrics = score_counterfactual_v2(prediction, target, {})
        self.assertEqual(metrics["numeric_difference_score"], 0.0)
        self.assertAlmostEqual(metrics["task_score"], 0.4)

    def test_malformed_prediction_scores_zero_without_crashing(self) -> None:
        record = self.records[0]
        predictions = [{"example_id": record["example_id"], "parsed_json": {"status": "wrong", "answer": {}}}]
        details, summary = evaluate_rows([record], predictions, self.master)
        self.assertEqual(details[0]["task_score"], 0.0)
        self.assertFalse(details[0]["schema_valid"])
        self.assertEqual(summary["evaluated_records"], 1)


if __name__ == "__main__":
    unittest.main()
