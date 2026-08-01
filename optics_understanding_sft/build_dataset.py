#!/usr/bin/env python3
"""Build the simulator-grounded optics-understanding SFT pilot."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_sft.physics.sim_adapter import simulate_and_measure

from .core import (
    ACTION_KEYS,
    DATASET_VERSION,
    TASK_TYPES,
    action_dict,
    apply_action_dict,
    assign_tasks_to_groups,
    axis_action,
    centroid_distance,
    classify_effects,
    discrete_grid,
    file_sha256,
    load_yaml,
    make_messages,
    make_qwen_record,
    modified_setup_config,
    prompt_key_hits,
    render_image,
    round_tree,
    rounded_state,
    safe_metadata,
    sample_setup_config,
    select_visual_examples,
    setup_snapshot,
    stable_json_hash,
    state_change,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "templates.yaml"
PARAMETERS = (
    "lens_focal_length_mm",
    "lens_to_camera_mm",
    "source_to_lens_mm",
    "beam_waist_mm",
    "lens_aperture_mm",
    "wavelength_nm",
)
SHORT_NAMES = {
    "setup_interpretation": "setup",
    "information_sufficiency": "suff",
    "causal_effects": "causal",
    "forward_prediction": "forward",
    "diagnosis": "diagnosis",
    "constrained_intervention": "control",
    "counterfactual_reasoning": "cf",
}
PROMPT_CONTRACT_VERSION = 2
OUTPUT_CONTRACTS: dict[str, dict[str, Any]] = {
    "setup_interpretation": {
        "status": "answerable",
        "answer": {
            "component_order": ["component_name"],
            "adjustable_parameters": ["parameter_name"],
            "total_source_to_sensor_mm": "number",
            "lens_focal_length_m": "number",
        },
    },
    "information_sufficiency": {
        "status": "answerable | insufficient_information",
        "answer": {
            "centroid_x_direction": "increase | decrease | no_change | null",
            "missing_fields": ["field_name"],
            "nonidentifiable_output": "field_name | null",
            "compatible_completions": ["object with hidden_value_mm and centroid_x_direction"],
            "answer_changing_completions": ["object with hidden_value_mm and centroid_x_direction"],
        },
    },
    "causal_effects": {
        "status": "answerable",
        "answer": {
            "effects": {
                "centroid_x": "increase | decrease | no_change",
                "centroid_y": "increase | decrease | no_change",
                "sigma_x": "increase | decrease | no_change",
                "sigma_y": "increase | decrease | no_change",
                "peak_intensity": "increase | decrease | no_change",
            }
        },
    },
    "forward_prediction": {
        "status": "answerable",
        "answer": {
            "after_state": {
                "centroid_x_px": "number",
                "centroid_y_px": "number",
                "sigma_x_px": "number",
                "sigma_y_px": "number",
                "peak_intensity": "number",
            },
            "change": {
                "centroid_x_px": "number",
                "centroid_y_px": "number",
                "sigma_x_px": "number",
                "sigma_y_px": "number",
                "peak_intensity": "number",
            },
        },
    },
    "diagnosis": {
        "status": "unique | ambiguous | unsupported",
        "answer": {"plausible_causes": ["candidate_id"]},
    },
    "constrained_intervention": {
        "action": {
            "actuator": "actuator_name",
            "signed_movement_mm": "number",
            "predicted_residual_px": "number",
            "executable_valid": "boolean",
        },
        "status": "feasible | infeasible_within_limits",
        "answer": {
            "control_plan": "object with numeric lens_x_delta_mm, lens_y_delta_mm, camera_x_delta_mm, camera_y_delta_mm | null",
            "expected_residual_px": "number | null",
            "best_achievable_residual_px": "number | null",
        },
    },
    "counterfactual_reasoning": {
        "status": "answerable",
        "answer": {
            "changed_parameter": "parameter_name",
            "response_a": "state_change_object",
            "response_b": "state_change_object",
            "response_difference": "state_change_object",
            "centroid_direction_preserved": "boolean",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("optics_understanding_sft/configs/pilot_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optics_understanding_sft/data/pilot_v1"),
    )
    parser.add_argument("--smoke", action="store_true", help="Build the 28-record smoke fixture.")
    parser.add_argument("--force", action="store_true", help="Replace an existing generated output directory.")
    parser.add_argument("--skip-audit", action="store_true")
    return parser.parse_args()


def smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(config)
    dataset = copied["dataset"]
    dataset["name"] = "optics_understanding_sft_smoke_fixture"
    dataset["scenarios"] = {"train": 5, "val": 1, "test": 1}
    dataset["questions_per_scenario"] = 4
    dataset["visual_records"] = {"train": 2, "val": 1, "test": 1}
    dataset["ood_test_scenarios"] = 0
    dataset["task_counts"] = {
        "train": {
            "setup_interpretation": 3,
            "information_sufficiency": 3,
            "causal_effects": 3,
            "forward_prediction": 3,
            "diagnosis": 3,
            "constrained_intervention": 3,
            "counterfactual_reasoning": 2,
        },
        "val": {
            "setup_interpretation": 0,
            "information_sufficiency": 0,
            "causal_effects": 0,
            "forward_prediction": 1,
            "diagnosis": 1,
            "constrained_intervention": 1,
            "counterfactual_reasoning": 1,
        },
        "test": {
            "setup_interpretation": 1,
            "information_sufficiency": 1,
            "causal_effects": 1,
            "forward_prediction": 0,
            "diagnosis": 0,
            "constrained_intervention": 0,
            "counterfactual_reasoning": 1,
        },
    }
    return copied


def controlled_status_schedule(task_type: str, expected: Mapping[str, Any]) -> list[str]:
    """Create an exact, nuisance-counterbalanced status schedule for corrective datasets."""
    counts = {str(label): int(count) for label, count in expected.items()}
    if any(count < 0 for count in counts.values()):
        raise ValueError(f"negative status count for {task_type}: {counts}")
    if task_type == "constrained_intervention":
        labels = ("feasible", "infeasible_within_limits")
        cycle_width = len(ACTION_KEYS)
    elif task_type == "information_sufficiency":
        labels = ("answerable", "insufficient_information")
        cycle_width = 1
    elif task_type == "diagnosis":
        labels = ("unique", "ambiguous", "unsupported")
        cycle_width = 2  # one x-axis and one y-axis example per status
    else:
        raise ValueError(f"status schedule is not supported for {task_type}")
    if set(counts) != set(labels):
        raise ValueError(f"unexpected status labels for {task_type}: {counts}")
    if task_type != "constrained_intervention" and len(set(counts.values())) != 1:
        raise ValueError(f"corrective status counts must be equal for {task_type}: {counts}")
    if any(count % cycle_width for count in counts.values()):
        raise ValueError(
            f"status counts for {task_type} must be divisible by nuisance cycle {cycle_width}: {counts}"
        )
    schedule: list[str] = []
    if len(set(counts.values())) == 1:
        # Preserve the frozen v2 ordering for all previously supported balanced
        # schedules.
        per_label = next(iter(counts.values()), 0)
        for _ in range(per_label // cycle_width):
            for label in labels:
                schedule.extend([label] * cycle_width)
    else:
        # The focused action-first round deliberately has three feasible
        # actuator cycles per infeasible cycle.  Keeping each cycle contiguous
        # still counterbalances every actuator exactly.
        for label in labels:
            schedule.extend([label] * counts[label])
    if Counter(schedule) != Counter(counts):
        raise AssertionError(f"internal status schedule mismatch for {task_type}")
    return schedule


def configured_status_schedules(dataset_cfg: Mapping[str, Any]) -> dict[str, dict[str, list[str]]]:
    schedules: dict[str, dict[str, list[str]]] = {}
    for split, task_counts in dataset_cfg.get("status_counts", {}).items():
        schedules[str(split)] = {
            str(task): controlled_status_schedule(str(task), statuses)
            for task, statuses in task_counts.items()
        }
    return schedules


def simulator_result(config: Mapping[str, Any], action: Mapping[str, Any] | None = None) -> dict[str, Any]:
    setup = setup_from_dict(copy.deepcopy(dict(config)))
    if action is not None:
        setup = apply_action_dict(setup, action)
    return simulate_and_measure(setup)


def replay_spec(
    name: str,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    action: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = {
        "name": name,
        "setup_config": copy.deepcopy(dict(config)),
        "expected_state": rounded_state(result["state"]),
    }
    if action is not None:
        spec["action"] = round_tree(dict(action), 6)
    return spec


def prompt_for(
    templates: Mapping[str, Any],
    task_type: str,
    split: str,
    rng: random.Random,
    prompt_inputs: Mapping[str, Any],
) -> tuple[str, str]:
    template_group = "test" if split == "test" else "train"
    options = list(templates[task_type][template_group])
    index = rng.randrange(len(options))
    template_id = f"{task_type}:{template_group}:{index}"
    prompt = prompt_text(options[index], task_type, prompt_inputs)
    return prompt, template_id


def prompt_text(template: str, task_type: str, prompt_inputs: Mapping[str, Any]) -> str:
    """Build a label-independent prompt with a task-wide output contract."""
    visible = {key: value for key, value in prompt_inputs.items() if key != "images"}
    return (
        f"{template}\n\n"
        f"Input data:\n{json.dumps(visible, indent=2, sort_keys=True)}\n\n"
        "Return only strict JSON and no additional prose. Literal alternatives separated by | are allowed values. "
        "Use null for fields that do not apply.\n"
        "Task output contract (all fields shown; use null when not applicable):\n"
        f"{json.dumps(OUTPUT_CONTRACTS[task_type], indent=2, sort_keys=True)}"
    )


def save_visual(
    output_dir: Path,
    split: str,
    example_id: str,
    named_results: list[tuple[str, Mapping[str, Any]]],
    rng: random.Random,
    render_cfg: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    paths: list[str] = []
    options: dict[str, Any] = {}
    for name, result in named_results:
        relative = Path("images") / split / f"{example_id}_{name}.png"
        options[name] = render_image(
            result["intensity"],
            output_dir / relative,
            rng,
            size_px=int(render_cfg["size_px"]),
            difficulty=str(render_cfg["difficulty"]),
        )
        paths.append(relative.as_posix())
    return paths, options


def base_record(
    *,
    group_id: str,
    split: str,
    task_type: str,
    task_index: int,
    prompt_inputs: dict[str, Any],
    target: dict[str, Any],
    scenario_seed: int,
    templates: Mapping[str, Any],
    rng: random.Random,
    visual: bool,
) -> dict[str, Any]:
    hits = prompt_key_hits(prompt_inputs)
    if hits:
        raise ValueError(f"Unsafe prompt fields for {group_id}/{task_type}: {hits}")
    example_id = f"{group_id}_{SHORT_NAMES[task_type]}_{task_index:03d}"
    prompt, template_id = prompt_for(templates, task_type, split, rng, prompt_inputs)
    return {
        "example_id": example_id,
        "group_id": group_id,
        "split": split,
        "task_type": task_type,
        "modality": "visual" if visual else "text",
        "prompt": prompt,
        "prompt_inputs": prompt_inputs,
        "target": target,
        "provenance": {
            "dataset_version": DATASET_VERSION,
            "scenario_seed": scenario_seed,
            "template_id": template_id,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "label_source": "optical_sim",
        },
    }


def setup_interpretation(
    group_id: str,
    split: str,
    task_index: int,
    scenario_seed: int,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    templates: Mapping[str, Any],
    rng: random.Random,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    setup = setup_from_dict(copy.deepcopy(dict(config)))
    metadata = safe_metadata(setup)
    prompt_inputs = {
        "setup": metadata,
        "requested_units": "metres for focal length",
        "actuator_interface": {
            "adjustable_parameters": ["lens_x", "lens_y", "camera_x", "camera_y"],
            "fixed_during_alignment": [
                "lens_focal_length_mm",
                "source_to_lens_mm",
                "lens_to_camera_mm",
                "beam_waist_mm",
                "wavelength_nm",
            ],
        },
    }
    target = {
        "status": "answerable",
        "answer": {
            "component_order": ["gaussian_source", "thin_lens", "camera_sensor"],
            "adjustable_parameters": ["lens_x", "lens_y", "camera_x", "camera_y"],
            "total_source_to_sensor_mm": round(
                float(metadata["source_to_lens_mm"]) + float(metadata["lens_to_camera_mm"]), 4
            ),
            "lens_focal_length_m": round(float(metadata["lens_focal_length_mm"]) / 1000.0, 6),
        },
    }
    record = base_record(
        group_id=group_id,
        split=split,
        task_type="setup_interpretation",
        task_index=task_index,
        prompt_inputs=prompt_inputs,
        target=target,
        scenario_seed=scenario_seed,
        templates=templates,
        rng=rng,
        visual=False,
    )
    return record, {"replay_specs": [replay_spec("baseline", config, base)]}


def information_sufficiency(
    group_id: str,
    split: str,
    task_index: int,
    scenario_seed: int,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    templates: Mapping[str, Any],
    rng: random.Random,
    label_cfg: Mapping[str, Any],
    desired_status: str | None = None,
    precomputed_sufficiency_grid: Mapping[str, Any] | None = None,
    forced_sufficiency_indices: list[int] | None = None,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if desired_status is not None:
        if desired_status not in {"answerable", "insufficient_information"}:
            raise ValueError(f"invalid sufficiency desired_status: {desired_status}")
        hidden_field = "lens_x_delta_mm"
        bound = 0.06
        if precomputed_sufficiency_grid is None:
            grid_values = tuple(
                round(-bound + (2.0 * bound / 12.0) * index, 6) for index in range(13)
            )
            grid_actions = [action_dict(**{hidden_field: value}) for value in grid_values]
            grid_results = [simulator_result(config, action) for action in grid_actions]
            grid_directions: list[str] = []
            for result in grid_results:
                delta = float(result["state"]["centroid_x_px"]) - float(base["state"]["centroid_x_px"])
                grid_directions.append(
                    "increase" if delta > 1.0 else "decrease" if delta < -1.0 else "no_change"
                )
        else:
            if precomputed_sufficiency_grid.get("hidden_field") != hidden_field:
                raise ValueError("precomputed sufficiency grid uses a different hidden field")
            grid_values = tuple(float(value) for value in precomputed_sufficiency_grid["grid_values"])
            grid_actions = list(precomputed_sufficiency_grid["grid_actions"])
            grid_results = list(precomputed_sufficiency_grid["grid_results"])
            grid_directions = list(precomputed_sufficiency_grid["grid_directions"])
            if not (len(grid_values) == len(grid_actions) == len(grid_results) == len(grid_directions) == 13):
                raise ValueError("precomputed sufficiency grid must contain thirteen aligned results")
        if forced_sufficiency_indices is not None:
            selected_indices = [int(index) for index in forced_sufficiency_indices]
            if len(selected_indices) != 5 or len(set(selected_indices)) != 5 or any(
                index < 0 or index >= len(grid_values) for index in selected_indices
            ):
                raise ValueError("forced sufficiency indices must contain five unique grid indices")
            selected_directions = {grid_directions[index] for index in selected_indices}
            expected_invariant = desired_status == "answerable"
            if (len(selected_directions) == 1) != expected_invariant:
                raise ValueError("forced sufficiency indices disagree with the requested status")
        elif desired_status == "answerable":
            direction_groups = {
                direction: [index for index, value in enumerate(grid_directions) if value == direction]
                for direction in ("increase", "decrease", "no_change")
            }
            selected_indices = max(direction_groups.values(), key=lambda values: (len(values), values))[:5]
            if len(selected_indices) < 5:
                raise RuntimeError(f"could not find five identifiable completions for {group_id}")
        else:
            first_by_direction = [
                next((index for index, value in enumerate(grid_directions) if value == direction), None)
                for direction in ("increase", "decrease", "no_change")
            ]
            selected_indices = [index for index in first_by_direction if index is not None]
            if len(selected_indices) < 2:
                raise RuntimeError(f"could not find nonidentifiable completions for {group_id}")
            selected_indices += [index for index in range(len(grid_values)) if index not in selected_indices][
                : 5 - len(selected_indices)
            ]
        values = tuple(grid_values[index] for index in selected_indices)
        actions = [grid_actions[index] for index in selected_indices]
        completions = [grid_results[index] for index in selected_indices]
        directions = [grid_directions[index] for index in selected_indices]
        visible_action = dict(actions[len(actions) // 2])
        visible_action.pop(hidden_field)
        completion_evidence = [
            {
                "hidden_value_mm": values[index],
                "centroid_x_direction": directions[index],
            }
            for index in range(len(values))
        ]
        first_by_output: dict[str, dict[str, Any]] = {}
        for item in completion_evidence:
            first_by_output.setdefault(str(item["centroid_x_direction"]), item)
        answer_changing = list(first_by_output.values())
        evidence_v3 = label_cfg.get("sufficiency_supervision") == "completion_evidence_v3"
        if evidence_v3 and desired_status == "insufficient_information" and len(answer_changing) < 2:
            raise RuntimeError(f"insufficient completion evidence did not change the answer for {group_id}")
        if desired_status == "answerable":
            answer = {"centroid_x_direction": directions[0], "missing_fields": []}
            if evidence_v3:
                answer["compatible_completions"] = completion_evidence
                answer["answer_changing_completions"] = []
        else:
            answer = {"missing_fields": [hidden_field], "nonidentifiable_output": "centroid_x_direction"}
            if evidence_v3:
                answer["compatible_completions"] = completion_evidence
                answer["answer_changing_completions"] = answer_changing
        target = {
            "status": desired_status,
            "answer": answer,
        }
        prompt_inputs = {
            "setup": safe_metadata(setup_from_dict(copy.deepcopy(dict(config)))),
            "current_observation": rounded_state(base["state"]),
            "candidate_action_with_hidden_field": round_tree(visible_action, 6),
            "hidden_action_field": hidden_field,
            "compatible_hidden_values_mm": list(values),
            "questioned_output": "centroid_x_direction",
        }
        record = base_record(
            group_id=group_id,
            split=split,
            task_type="information_sufficiency",
            task_index=task_index,
            prompt_inputs=prompt_inputs,
            target=target,
            scenario_seed=scenario_seed,
            templates=templates,
            rng=rng,
            visual=False,
        )
        if evidence_v3:
            record["provenance"]["sufficiency_supervision"] = "completion_evidence_v3"
            record["provenance"]["completion_sensitivity"] = (
                "invariant" if desired_status == "answerable" else "answer_changing"
            )
        specs = [
            replay_spec(f"completion_{index}", config, result, action)
            for index, (action, result) in enumerate(zip(actions, completions))
        ]
        return record, {
            "hidden_field": hidden_field,
            "compatible_values_mm": list(values),
            "completion_directions": directions,
            "completion_evidence": completion_evidence,
            "replay_specs": specs,
        }

    insufficient = task_index % 2 == 0
    hidden_field = "lens_focal_length_mm" if task_index % 4 < 2 else "lens_to_camera_mm"
    action = axis_action("lens_x_delta_mm", 0.06 if task_index % 3 else -0.06)
    metadata = safe_metadata(setup_from_dict(copy.deepcopy(dict(config))))
    if insufficient:
        factors = (0.60, 0.80, 1.0, 1.20, 1.40)
        candidate_fields = (
            hidden_field,
            "source_to_lens_mm",
            "beam_waist_mm",
            "lens_focal_length_mm",
            "lens_to_camera_mm",
        )
        best: tuple[float, str, list[dict[str, Any]], list[dict[str, Any]]] | None = None
        for candidate_field in dict.fromkeys(candidate_fields):
            candidate_variants = [modified_setup_config(config, candidate_field, factor) for factor in factors]
            candidate_results = [simulator_result(variant, action) for variant in candidate_variants]
            candidate_values = [float(result["state"]["centroid_x_px"]) for result in candidate_results]
            candidate_spread = max(candidate_values) - min(candidate_values)
            if best is None or candidate_spread > best[0]:
                best = (candidate_spread, candidate_field, candidate_variants, candidate_results)
            if candidate_spread > float(label_cfg["centroid_effect_threshold_px"]):
                break
        assert best is not None
        _, hidden_field, variants, precomputed_completions = best
    else:
        hidden_field = "power_w"
        factors = (0.5, 0.75, 1.0, 1.25, 1.5)
        variants = []
        for factor in factors:
            changed = copy.deepcopy(dict(config))
            changed["source"]["power"] = float(config["source"].get("power", 1.0)) * factor
            variants.append(changed)
    metadata.pop(hidden_field, None)
    completions = (
        precomputed_completions
        if insufficient
        else [simulator_result(variant, action) for variant in variants]
    )
    x_values = [float(result["state"]["centroid_x_px"]) for result in completions]
    spread = max(x_values) - min(x_values)
    is_identifiable = spread <= float(label_cfg["centroid_effect_threshold_px"])
    direction_delta = x_values[len(x_values) // 2] - float(base["state"]["centroid_x_px"])
    direction = "increase" if direction_delta > 1.0 else "decrease" if direction_delta < -1.0 else "no_change"
    target = {
        "status": "answerable" if is_identifiable else "insufficient_information",
        "answer": (
            {"centroid_x_direction": direction, "missing_fields": []}
            if is_identifiable
            else {"missing_fields": [hidden_field], "nonidentifiable_output": "centroid_x_direction"}
        ),
    }
    prompt_inputs = {
        "setup_with_hidden_field": metadata,
        "current_observation": rounded_state(base["state"]),
        "candidate_action": action,
        "questioned_output": "centroid_x_direction",
    }
    record = base_record(
        group_id=group_id,
        split=split,
        task_type="information_sufficiency",
        task_index=task_index,
        prompt_inputs=prompt_inputs,
        target=target,
        scenario_seed=scenario_seed,
        templates=templates,
        rng=rng,
        visual=False,
    )
    specs = [replay_spec(f"completion_{i}", variant, result, action) for i, (variant, result) in enumerate(zip(variants, completions))]
    return record, {"hidden_field": hidden_field, "completion_spread_px": spread, "replay_specs": specs}


def causal_effects(
    group_id: str,
    split: str,
    task_index: int,
    scenario_seed: int,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    templates: Mapping[str, Any],
    rng: random.Random,
    label_cfg: Mapping[str, Any],
    visual: bool,
    output_dir: Path,
    render_cfg: Mapping[str, Any],
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    axes = ACTION_KEYS
    if task_index % 7 == 0:
        action = action_dict()
    else:
        axis = axes[task_index % len(axes)]
        scale = 0.035 if axis.startswith("lens") else 0.018
        action = axis_action(axis, scale if task_index % 2 else -scale)
    after = simulator_result(config, action)
    target = {"status": "answerable", "answer": {"effects": classify_effects(base["state"], after["state"], label_cfg)}}
    prompt_inputs: dict[str, Any] = {
        "setup": safe_metadata(setup_from_dict(copy.deepcopy(dict(config)))),
        "intervention": action,
        "images": [],
    }
    render_options: dict[str, Any] = {}
    example_stub = f"{group_id}_{SHORT_NAMES['causal_effects']}_{task_index:03d}"
    if visual:
        prompt_inputs["images"], render_options = save_visual(
            output_dir, split, example_stub, [("before", base), ("after", after)], rng, render_cfg
        )
        prompt_inputs["observation_format"] = "before image followed by after image"
    else:
        prompt_inputs["current_observation"] = rounded_state(base["state"])
        prompt_inputs["observed_post_intervention"] = rounded_state(after["state"])
    record = base_record(
        group_id=group_id,
        split=split,
        task_type="causal_effects",
        task_index=task_index,
        prompt_inputs=prompt_inputs,
        target=target,
        scenario_seed=scenario_seed,
        templates=templates,
        rng=rng,
        visual=visual,
    )
    return record, {
        "render_options": render_options,
        "replay_specs": [replay_spec("before", config, base), replay_spec("after", config, after, action)],
    }


def forward_prediction(
    group_id: str,
    split: str,
    task_index: int,
    scenario_seed: int,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    templates: Mapping[str, Any],
    rng: random.Random,
    visual: bool,
    output_dir: Path,
    render_cfg: Mapping[str, Any],
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    action = action_dict(
        lens_x_delta_mm=rng.uniform(-0.05, 0.05),
        lens_y_delta_mm=rng.uniform(-0.05, 0.05),
        camera_x_delta_mm=rng.uniform(-0.02, 0.02),
        camera_y_delta_mm=rng.uniform(-0.02, 0.02),
    )
    action = round_tree(action, 6)
    after = simulator_result(config, action)
    target = {
        "status": "answerable",
        "answer": {
            "after_state": rounded_state(after["state"]),
            "change": state_change(base["state"], after["state"]),
        },
    }
    prompt_inputs: dict[str, Any] = {
        "setup": safe_metadata(setup_from_dict(copy.deepcopy(dict(config)))),
        "action": action,
        "images": [],
    }
    render_options: dict[str, Any] = {}
    example_stub = f"{group_id}_{SHORT_NAMES['forward_prediction']}_{task_index:03d}"
    if visual:
        prompt_inputs["images"], render_options = save_visual(
            output_dir, split, example_stub, [("before", base)], rng, render_cfg
        )
        prompt_inputs["observation_format"] = "initial beam image"
    else:
        prompt_inputs["current_observation"] = rounded_state(base["state"])
    record = base_record(
        group_id=group_id,
        split=split,
        task_type="forward_prediction",
        task_index=task_index,
        prompt_inputs=prompt_inputs,
        target=target,
        scenario_seed=scenario_seed,
        templates=templates,
        rng=rng,
        visual=visual,
    )
    return record, {
        "render_options": render_options,
        "replay_specs": [replay_spec("before", config, base), replay_spec("after", config, after, action)],
    }


def _candidate_match(a: Mapping[str, Any], b: Mapping[str, Any], labels: Mapping[str, Any]) -> bool:
    return (
        centroid_distance(a, b) <= float(labels["diagnosis_centroid_tolerance_px"])
        and abs(float(a["sigma_x_px"]) - float(b["sigma_x_px"])) <= float(labels["diagnosis_sigma_tolerance_px"])
        and abs(float(a["sigma_y_px"]) - float(b["sigma_y_px"])) <= float(labels["diagnosis_sigma_tolerance_px"])
    )


def _matching_camera_action(
    config: Mapping[str, Any], base: Mapping[str, Any], observed: Mapping[str, Any], axis: str
) -> dict[str, float]:
    key = f"camera_{axis}_delta_mm"
    probe_action = axis_action(key, 0.01)
    probe = simulator_result(config, probe_action)
    state_key = f"centroid_{axis}_px"
    slope = (float(probe["state"][state_key]) - float(base["state"][state_key])) / 0.01
    desired = float(observed["state"][state_key]) - float(base["state"][state_key])
    delta = 0.0 if abs(slope) < 1e-9 else max(-0.06, min(0.06, desired / slope))
    return axis_action(key, delta)


def diagnosis(
    group_id: str,
    split: str,
    task_index: int,
    scenario_seed: int,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    templates: Mapping[str, Any],
    rng: random.Random,
    label_cfg: Mapping[str, Any],
    visual: bool,
    output_dir: Path,
    render_cfg: Mapping[str, Any],
    desired_status: str | None = None,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    axis = "x" if task_index % 2 == 0 else "y"
    mode = task_index % 10
    magnitude = (0.20 if desired_status is not None else 0.035) * (-1.0 if task_index % 3 == 0 else 1.0)
    true_action = axis_action(f"lens_{axis}_delta_mm", magnitude)
    observed_config = dict(config)
    observed_action: dict[str, float] | None = true_action
    if desired_status is None and mode == 0:
        observed_config = modified_setup_config(config, "lens_focal_length_mm", 1.10)
        observed_action = None
    if desired_status == "unsupported":
        sign = -1.0 if task_index % 2 else 1.0
        candidates = [
            {"candidate_id": "lens_same_axis", "action": axis_action(f"lens_{axis}_delta_mm", 0.03 * sign)},
            {"candidate_id": "lens_opposite", "action": axis_action(f"lens_{axis}_delta_mm", -0.03 * sign)},
            {
                "candidate_id": "lens_other_axis",
                "action": axis_action(f"lens_{'y' if axis == 'x' else 'x'}_delta_mm", 0.03 * sign),
            },
            {"candidate_id": "camera_same_axis", "action": axis_action(f"camera_{axis}_delta_mm", 0.02 * sign)},
        ]
        candidate_results = [(candidate, simulator_result(config, candidate["action"])) for candidate in candidates]
        observed_options: list[tuple[dict[str, Any], dict[str, float] | None]] = [
            (modified_setup_config(config, "lens_focal_length_mm", factor), None)
            for factor in (0.50, 0.70, 1.30, 1.60)
        ]
        observed_options += [
            (modified_setup_config(config, "beam_waist_mm", factor), None) for factor in (0.50, 1.80, 2.20)
        ]
        observed_options += [
            (modified_setup_config(config, "lens_to_camera_mm", factor), None)
            for factor in (0.60, 0.75, 1.25, 1.50)
        ]
        observed_options += [
            (
                dict(config),
                action_dict(lens_x_delta_mm=value * sign, lens_y_delta_mm=-value * sign),
            )
            for value in (0.10, 0.20, 0.50, 1.0)
        ]
        observed = None
        for option_config, option_action in observed_options:
            option_result = simulator_result(option_config, option_action)
            if not any(
                _candidate_match(result["state"], option_result["state"], label_cfg)
                for _, result in candidate_results
            ):
                observed_config, observed_action, observed = option_config, option_action, option_result
                break
        if observed is None:
            raise RuntimeError(f"could not construct unsupported diagnosis observation for {group_id}")
    elif desired_status in {"unique", "ambiguous"}:
        lens_values = (-1.0, -0.30, -0.10, -0.03, 0.03, 0.10, 0.30, 1.0)
        camera_values = (-1.0, -0.10, -0.03, 0.03, 0.10, 1.0)
        library_actions = [axis_action(key, value) for key in ACTION_KEYS[:2] for value in lens_values]
        library_actions += [axis_action(key, value) for key in ACTION_KEYS[2:] for value in camera_values]
        library_actions += [
            action_dict(lens_x_delta_mm=value, lens_y_delta_mm=-value) for value in (-1.0, -0.2, 0.2, 1.0)
        ]
        primary_actions = [
            action
            for action in library_actions
            if abs(float(action[f"lens_{axis}_delta_mm"])) > 0.0
            and abs(float(action[f"lens_{'y' if axis == 'x' else 'x'}_delta_mm"])) == 0.0
        ]
        selected = None
        for primary_action in primary_actions:
            primary_result = simulator_result(config, primary_action)
            alternatives = []
            for alternative_action in library_actions:
                if alternative_action == primary_action:
                    continue
                alternative_result = simulator_result(config, alternative_action)
                if not _candidate_match(alternative_result["state"], primary_result["state"], label_cfg):
                    alternatives.append((alternative_action, alternative_result))
                if len(alternatives) == 3:
                    break
            if len(alternatives) >= 3:
                selected = (primary_action, primary_result, alternatives)
                break
        if selected is None:
            raise RuntimeError(f"could not construct separable diagnosis candidates for {group_id}")
        true_action, observed, alternatives = selected
        observed_action = true_action
        observed_config = dict(config)
        candidates = [{"candidate_id": "primary_lens_intervention", "action": true_action}]
        candidates += [
            {"candidate_id": f"alternative_intervention_{index}", "action": action}
            for index, (action, _) in enumerate(alternatives[:2], start=1)
        ]
        if desired_status == "ambiguous":
            combination = dict(true_action)
            camera_key = f"camera_{axis}_delta_mm"
            for camera_delta in (0.005, 0.002, 0.001, 0.0005, 0.0001):
                combination[camera_key] = camera_delta
                combined_result = simulator_result(config, combination)
                if _candidate_match(combined_result["state"], observed["state"], label_cfg):
                    break
            else:
                raise RuntimeError(f"could not construct ambiguous diagnosis alternative for {group_id}")
            candidates.append({"candidate_id": "lens_camera_combination", "action": round_tree(combination, 6)})
        else:
            candidates.append({"candidate_id": "alternative_intervention_3", "action": alternatives[2][0]})
        candidate_results = [(candidate, simulator_result(config, candidate["action"])) for candidate in candidates]
    else:
        observed = simulator_result(observed_config, observed_action)
        candidates = [
            {"candidate_id": "lens_same_axis", "action": true_action},
            {"candidate_id": "lens_opposite", "action": axis_action(f"lens_{axis}_delta_mm", -magnitude)},
            {
                "candidate_id": "lens_other_axis",
                "action": axis_action(f"lens_{'y' if axis == 'x' else 'x'}_delta_mm", magnitude),
            },
        ]
        if mode in (1, 2, 3):
            camera_action = _matching_camera_action(config, base, observed, axis)
        else:
            camera_action = axis_action(f"camera_{axis}_delta_mm", -0.02 if magnitude > 0 else 0.02)
        candidates.append({"candidate_id": "camera_same_axis", "action": round_tree(camera_action, 6)})
        candidate_results = [(candidate, simulator_result(config, candidate["action"])) for candidate in candidates]
    plausible = [
        candidate["candidate_id"]
        for candidate, result in candidate_results
        if _candidate_match(result["state"], observed["state"], label_cfg)
    ]
    status = "unsupported" if not plausible else "unique" if len(plausible) == 1 else "ambiguous"
    if desired_status is not None and status != desired_status:
        raise RuntimeError(
            f"diagnosis status construction failed for {group_id}: wanted {desired_status}, got {status} ({plausible})"
        )
    target = {"status": status, "answer": {"plausible_causes": plausible}}
    prompt_inputs: dict[str, Any] = {
        "setup": safe_metadata(setup_from_dict(copy.deepcopy(dict(config)))),
        "candidate_interventions": candidates,
        "matching_tolerance": {
            "centroid_px": float(label_cfg["diagnosis_centroid_tolerance_px"]),
            "sigma_px": float(label_cfg["diagnosis_sigma_tolerance_px"]),
        },
        "images": [],
    }
    render_options: dict[str, Any] = {}
    example_stub = f"{group_id}_{SHORT_NAMES['diagnosis']}_{task_index:03d}"
    if visual:
        prompt_inputs["images"], render_options = save_visual(
            output_dir, split, example_stub, [("before", base), ("observed", observed)], rng, render_cfg
        )
        prompt_inputs["observation_format"] = "before image followed by observed image"
    else:
        prompt_inputs["current_observation"] = rounded_state(base["state"])
        prompt_inputs["observed_transition_state"] = rounded_state(observed["state"])
    record = base_record(
        group_id=group_id,
        split=split,
        task_type="diagnosis",
        task_index=task_index,
        prompt_inputs=prompt_inputs,
        target=target,
        scenario_seed=scenario_seed,
        templates=templates,
        rng=rng,
        visual=visual,
    )
    specs = [replay_spec("baseline", config, base), replay_spec("observed", observed_config, observed, observed_action)]
    specs += [replay_spec(candidate["candidate_id"], config, result, candidate["action"]) for candidate, result in candidate_results]
    return record, {"render_options": render_options, "replay_specs": specs}


def constrained_intervention(
    group_id: str,
    split: str,
    task_index: int,
    scenario_seed: int,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    templates: Mapping[str, Any],
    rng: random.Random,
    label_cfg: Mapping[str, Any],
    desired_status: str | None = None,
    forced_control_grid: list[float] | None = None,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    action_first_v3 = label_cfg.get("control_supervision") == "action_first_v3"
    axis_key = (
        ("lens_x_delta_mm", "lens_y_delta_mm")[task_index % 2]
        if action_first_v3
        else ACTION_KEYS[task_index % len(ACTION_KEYS)]
    )
    is_lens = axis_key.startswith("lens")
    max_bound = 0.06 if is_lens else 0.03
    min_bound = 0.025 if is_lens else 0.012
    bound = round(rng.uniform(min_bound, max_bound), 5)
    grid_count = int(label_cfg["constrained_grid_values"])
    control_variant: str | None = None
    desired_action_sign: int | None = None

    def make_grid(candidate_bound: float) -> tuple[list[float], list[tuple[float, dict[str, float], dict[str, Any]]]]:
        if forced_control_grid is not None:
            candidate_grid = [round(float(value), 6) for value in forced_control_grid]
            if len(candidate_grid) != grid_count or 0.0 not in candidate_grid:
                raise ValueError("forced control grid must contain the configured number of values and zero")
        elif action_first_v3 and control_variant is not None and grid_count == 9:
            if control_variant == "near_boundary" and desired_action_sign == -1:
                ratios = (-1.0, -0.9375, -0.875, -0.8125, -0.75, 0.0, 0.75, 0.875, 1.0)
            elif control_variant == "near_boundary":
                ratios = (-1.0, -0.875, -0.75, 0.0, 0.75, 0.8125, 0.875, 0.9375, 1.0)
            elif desired_action_sign == -1:
                ratios = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.75, 0.875, 0.9375, 1.0)
            else:
                ratios = (-1.0, -0.9375, -0.875, -0.75, 0.0, 0.25, 0.5, 0.75, 1.0)
            candidate_grid = [round(candidate_bound * ratio, 6) for ratio in ratios]
        else:
            candidate_grid = discrete_grid(candidate_bound, grid_count)
        candidate_results = []
        for candidate_value in candidate_grid:
            candidate_action = axis_action(axis_key, candidate_value)
            candidate_results.append(
                (candidate_value, candidate_action, simulator_result(config, candidate_action))
            )
        return candidate_grid, candidate_results

    grid, grid_results = make_grid(bound)

    if desired_status not in {None, "feasible", "infeasible_within_limits"}:
        raise ValueError(f"invalid control desired_status: {desired_status}")
    tolerance = float(label_cfg["control_success_tolerance_px"])
    # Keep the prompt-visible baseline error distribution comparable across the
    # two labels.  The zero action is in every grid, so an infeasible target is
    # necessarily more than ``tolerance`` from the baseline.  Choosing the
    # farthest grid point for feasible rows made camera examples much easier
    # than infeasible rows and left a status shortcut in the baseline error.
    target_error_goal = tolerance + 0.5
    desired_feasible = desired_status == "feasible" if desired_status is not None else task_index % 2 == 0
    if desired_feasible:
        if action_first_v3:
            # Cycle through both signs and both movement regimes.  Consecutive
            # regimes are paired after generation, so the prompt-visible error
            # can be matched across independent physical scenarios.
            feasible_cycle = (task_index // len(ACTION_KEYS)) % 4
            control_variant, desired_action_sign = (
                ("minimum_motion", 1),
                ("near_boundary", -1),
                ("minimum_motion", -1),
                ("near_boundary", 1),
            )[feasible_cycle]
            candidates: list[
                tuple[
                    tuple[float, float, float],
                    float,
                    list[float],
                    list[tuple[float, dict[str, float], dict[str, Any]]],
                    float,
                    dict[str, float],
                    dict[str, Any],
                ]
            ] = []
            # Three grids are sufficient to expose the centre-adjacent and
            # boundary regimes.  A wider sweep produced the same canonical
            # modes but multiplied simulator cost without adding supervision.
            def collect_candidates(candidate_bounds: list[float]) -> None:
                for candidate_bound in sorted(set(candidate_bounds)):
                    candidate_grid, candidate_results = make_grid(candidate_bound)
                    for target_value, target_action, candidate_target in candidate_results:
                        baseline_error = centroid_distance(base["state"], candidate_target["state"])
                        if abs(target_value) <= 1e-9 or baseline_error <= tolerance:
                            continue
                        target_scores = [
                            (
                                centroid_distance(result["state"], candidate_target["state"]),
                                abs(value),
                                value,
                                action,
                            )
                            for value, action, result in candidate_results
                        ]
                        successful_target = [item for item in target_scores if item[0] <= tolerance]
                        if not successful_target:
                            continue
                        canonical = min(successful_target, key=lambda item: (item[1], item[0]))
                        canonical_value = float(canonical[2])
                        if canonical_value * desired_action_sign <= 1e-9:
                            continue
                        movement_ratio = abs(canonical_value) / candidate_bound
                        mode_ok = (
                            movement_ratio <= 0.35
                            if control_variant == "minimum_motion"
                            else movement_ratio >= 0.50
                        )
                        if not mode_ok:
                            continue
                        mode_score = movement_ratio if control_variant == "minimum_motion" else -movement_ratio
                        candidates.append(
                            (
                                (abs(baseline_error - target_error_goal), mode_score, abs(target_value)),
                                candidate_bound,
                                candidate_grid,
                                candidate_results,
                                target_value,
                                target_action,
                                candidate_target,
                            )
                        )

            collect_candidates([round(min_bound, 6), bound, round(max_bound, 6)])
            if not candidates and control_variant == "minimum_motion":
                collect_candidates(
                    [
                        round(min_bound + (max_bound - min_bound) * fraction, 6)
                        for fraction in (0.25, 0.5, 0.75)
                    ]
                )
            if not candidates and control_variant == "minimum_motion":
                # Some sampled setups are locally insensitive inside the
                # nominal alignment range.  Evaluate outer grids only for
                # those actual failures.
                collect_candidates([round(max_bound * 2.0, 6), round(max_bound * 5.0, 6)])
            if not candidates and control_variant == "minimum_motion":
                collect_candidates(
                    [
                        round(max_bound * 10.0, 6),
                        round(max_bound * 20.0, 6),
                        round(max_bound * 50.0, 6),
                    ]
                )
            if not candidates:
                raise RuntimeError(
                    f"could not construct {control_variant} feasible control with sign {desired_action_sign} for {group_id}"
                )
            _, bound, grid, grid_results, hidden_value, hidden_action, target_result = min(
                candidates, key=lambda item: item[0]
            )
        else:
            hidden_value, hidden_action, target_result = min(
                grid_results,
                key=lambda item: (
                    abs(centroid_distance(base["state"], item[2]["state"]) - target_error_goal),
                    -abs(item[0]),
                ),
            )
    else:
        hidden_candidates: list[tuple[dict[str, float], dict[str, Any], float]] = []
        if desired_status is not None:
            desired_error = target_error_goal
            # Any actuator except the declared active one is fixed from the
            # model's point of view and may ground an infeasible target.  Keep
            # the inactive actuator on the same axis in the search: excluding
            # it created rare diffraction/aliasing fallbacks with enormous
            # prompt-visible errors.
            other_axes = [key for key in ACTION_KEYS if key != axis_key]
            active_values = [0.0] if action_first_v3 else [grid[index] for index in (0, 4, 8)]
            # Fine factors find the first simulator-grounded target just beyond
            # the declared grid tolerance instead of jumping to a very distant
            # fallback target.  This is still a bounded deterministic search.
            factors = (
                (-0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5)
                if action_first_v3
                else (-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, -0.02,
                      0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
            )
            for active_value in active_values:
                for other_axis in other_axes:
                    other_bound = 0.06 if other_axis.startswith("lens") else 0.03
                    for factor in factors:
                        candidate_action = action_dict(
                            **{axis_key: active_value, other_axis: other_bound * factor}
                        )
                        candidate_result = simulator_result(config, candidate_action)
                        nearest_grid_residual = min(
                            centroid_distance(candidate_result["state"], item[2]["state"])
                            for item in grid_results
                        )
                        if nearest_grid_residual > tolerance:
                            hidden_candidates.append((candidate_action, candidate_result, nearest_grid_residual))
            if not hidden_candidates:
                # Rare near-Nyquist configurations can be locally insensitive
                # over the normal nuisance range.  Expand one inactive axis at
                # a time before trying combined or legacy extreme fallbacks;
                # this finds the first nearby simulator state beyond tolerance
                # and avoids discontinuous, very distant targets.
                for other_axis in other_axes:
                    other_bound = 0.06 if other_axis.startswith("lens") else 0.03
                    for factor in (-50.0, -30.0, -20.0, -10.0, -5.0, -3.0,
                                   3.0, 5.0, 10.0, 20.0, 30.0, 50.0):
                        candidate_action = axis_action(other_axis, other_bound * factor)
                        candidate_result = simulator_result(config, candidate_action)
                        nearest_grid_residual = min(
                            centroid_distance(candidate_result["state"], item[2]["state"])
                            for item in grid_results
                        )
                        if nearest_grid_residual > tolerance:
                            hidden_candidates.append((candidate_action, candidate_result, nearest_grid_residual))
            if not hidden_candidates:
                for active_value in active_values:
                    for factor in (3.0, 5.0, 10.0):
                        overrides = {axis_key: active_value}
                        for index, other_axis in enumerate(other_axes):
                            other_bound = 0.06 if other_axis.startswith("lens") else 0.03
                            overrides[other_axis] = other_bound * factor * (-1.0 if index else 1.0)
                        candidate_action = action_dict(**overrides)
                        candidate_result = simulator_result(config, candidate_action)
                        nearest_grid_residual = min(
                            centroid_distance(candidate_result["state"], item[2]["state"])
                            for item in grid_results
                        )
                        if nearest_grid_residual > tolerance:
                            hidden_candidates.append((candidate_action, candidate_result, nearest_grid_residual))
            if not hidden_candidates:
                for other_axis in ACTION_KEYS:
                    if other_axis == axis_key:
                        continue
                    magnitude = 0.50 if other_axis.startswith("lens") else 2.00
                    for value in (-magnitude, magnitude):
                        candidate_action = axis_action(other_axis, value)
                        candidate_result = simulator_result(config, candidate_action)
                        nearest_grid_residual = min(
                            centroid_distance(candidate_result["state"], item[2]["state"])
                            for item in grid_results
                        )
                        if nearest_grid_residual > tolerance:
                            hidden_candidates.append((candidate_action, candidate_result, nearest_grid_residual))
            if not hidden_candidates:
                raise RuntimeError(f"could not construct any infeasible control for {group_id}")
            hidden_action, target_result, _ = min(
                hidden_candidates,
                key=lambda item: (
                    abs(centroid_distance(base["state"], item[1]["state"]) - desired_error),
                    item[2],
                ),
            )
        else:
            for other_axis in ACTION_KEYS:
                if other_axis == axis_key:
                    continue
                magnitude = 0.50 if other_axis.startswith("lens") else 2.00
                for value in (-magnitude, magnitude):
                    candidate_action = axis_action(other_axis, value)
                    candidate_result = simulator_result(config, candidate_action)
                    nearest_grid_residual = min(
                        centroid_distance(candidate_result["state"], item[2]["state"])
                        for item in grid_results
                    )
                    hidden_candidates.append((candidate_action, candidate_result, nearest_grid_residual))
            hidden_action, target_result, _ = max(hidden_candidates, key=lambda item: item[2])
    scored: list[tuple[float, float, dict[str, float], dict[str, Any]]] = []
    for value, action, result in grid_results:
        scored.append((centroid_distance(result["state"], target_result["state"]), abs(value), action, result))
    successful = [item for item in scored if item[0] <= tolerance]
    if successful:
        best = min(successful, key=lambda item: (item[1], item[0]))
        status = "feasible"
        answer = {
            "control_plan": round_tree(best[2], 6),
            "expected_residual_px": round(best[0], 4),
        }
    else:
        best = min(scored, key=lambda item: item[0])
        status = "infeasible_within_limits"
        answer = {
            "control_plan": None,
            "best_achievable_residual_px": round(best[0], 4),
        }
    prompt_inputs = {
        "setup": safe_metadata(setup_from_dict(copy.deepcopy(dict(config)))),
        "current_observation": rounded_state(base["state"]),
        "target_observation": rounded_state(target_result["state"]),
        "actuator_constraints": {
            "active_actuator": axis_key,
            "allowed_values_mm": grid,
            "success_tolerance_px": tolerance,
            "all_other_actuators_fixed": True,
        },
    }
    if action_first_v3:
        # This insertion order is preserved in the exported completion text.
        # The status is intentionally derived only after the executable action
        # evidence has been stated.
        target = {
            "action": {
                "actuator": axis_key,
                "signed_movement_mm": round(float(best[2][axis_key]), 6),
                "predicted_residual_px": round(float(best[0]), 4),
                "executable_valid": status == "feasible",
            },
            "status": status,
            "answer": answer,
        }
    else:
        target = {"status": status, "answer": answer}
    if desired_status is not None and status != desired_status:
        raise RuntimeError(f"control status construction failed for {group_id}: wanted {desired_status}, got {status}")
    record = base_record(
        group_id=group_id,
        split=split,
        task_type="constrained_intervention",
        task_index=task_index,
        prompt_inputs=prompt_inputs,
        target=target,
        scenario_seed=scenario_seed,
        templates=templates,
        rng=rng,
        visual=False,
    )
    if action_first_v3:
        record["provenance"]["control_supervision"] = "action_first_v3"
        record["provenance"]["control_action_mode"] = control_variant or "best_attempt"
        movement = float(best[2][axis_key])
        record["provenance"]["control_action_sign"] = (
            "positive" if movement > 0 else "negative" if movement < 0 else "zero"
        )
    specs = [replay_spec("baseline", config, base), replay_spec("target", config, target_result, hidden_action)]
    specs += [replay_spec(f"grid_{index}", config, item[3], item[2]) for index, item in enumerate(scored)]
    return record, {
        "hidden_target_action": round_tree(hidden_action, 6),
        "active_actuator": axis_key,
        "control_action_mode": control_variant or "best_attempt",
        "canonical_action": round_tree(best[2], 6),
        "grid_scores": [round(item[0], 6) for item in scored],
        "replay_specs": specs,
    }


def counterfactual_reasoning(
    group_id: str,
    split: str,
    task_index: int,
    scenario_seed: int,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    templates: Mapping[str, Any],
    rng: random.Random,
    label_cfg: Mapping[str, Any],
    visual: bool,
    output_dir: Path,
    render_cfg: Mapping[str, Any],
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parameter = PARAMETERS[task_index % len(PARAMETERS)]
    factor = 0.85 if task_index % 2 == 0 else 1.15
    changed_config = modified_setup_config(config, parameter, factor)
    changed_before = simulator_result(changed_config)
    action = axis_action("lens_x_delta_mm" if task_index % 2 == 0 else "lens_y_delta_mm", 0.04)
    original_after = simulator_result(config, action)
    changed_after = simulator_result(changed_config, action)
    response_a = state_change(base["state"], original_after["state"])
    response_b = state_change(changed_before["state"], changed_after["state"])
    state_key = "centroid_x_px" if action["lens_x_delta_mm"] else "centroid_y_px"
    direction_a = math.copysign(1.0, response_a[state_key]) if abs(response_a[state_key]) > 1.0 else 0.0
    direction_b = math.copysign(1.0, response_b[state_key]) if abs(response_b[state_key]) > 1.0 else 0.0
    target = {
        "status": "answerable",
        "answer": {
            "changed_parameter": parameter,
            "response_a": response_a,
            "response_b": response_b,
            "response_difference": {
                key: round(float(response_b[key]) - float(response_a[key]), 4) for key in response_a
            },
            "centroid_direction_preserved": direction_a == direction_b,
        },
    }
    prompt_inputs: dict[str, Any] = {
        "scenario_a_setup": safe_metadata(setup_from_dict(copy.deepcopy(dict(config)))),
        "scenario_b_setup": safe_metadata(setup_from_dict(copy.deepcopy(changed_config))),
        "changed_parameter": parameter,
        "shared_action": action,
        "images": [],
    }
    render_options: dict[str, Any] = {}
    example_stub = f"{group_id}_{SHORT_NAMES['counterfactual_reasoning']}_{task_index:03d}"
    if visual:
        prompt_inputs["images"], render_options = save_visual(
            output_dir,
            split,
            example_stub,
            [("scenario_a_after", original_after), ("scenario_b_after", changed_after)],
            rng,
            render_cfg,
        )
        prompt_inputs["observation_format"] = "scenario A after image followed by scenario B after image"
    else:
        prompt_inputs["scenario_a_before"] = rounded_state(base["state"])
        prompt_inputs["scenario_b_before"] = rounded_state(changed_before["state"])
    record = base_record(
        group_id=group_id,
        split=split,
        task_type="counterfactual_reasoning",
        task_index=task_index,
        prompt_inputs=prompt_inputs,
        target=target,
        scenario_seed=scenario_seed,
        templates=templates,
        rng=rng,
        visual=visual,
    )
    specs = [
        replay_spec("scenario_a_before", config, base),
        replay_spec("scenario_a_after", config, original_after, action),
        replay_spec("scenario_b_before", changed_config, changed_before),
        replay_spec("scenario_b_after", changed_config, changed_after, action),
    ]
    return record, {"render_options": render_options, "counterfactual_factor": factor, "replay_specs": specs}


BUILDERS = {
    "setup_interpretation": setup_interpretation,
    "information_sufficiency": information_sufficiency,
    "causal_effects": causal_effects,
    "forward_prediction": forward_prediction,
    "diagnosis": diagnosis,
    "constrained_intervention": constrained_intervention,
    "counterfactual_reasoning": counterfactual_reasoning,
}


def canonical_public(record: Mapping[str, Any], include_target: bool) -> dict[str, Any]:
    keys = ("example_id", "group_id", "split", "task_type", "modality", "prompt", "prompt_inputs", "provenance")
    result = {key: copy.deepcopy(record[key]) for key in keys}
    if include_target:
        result["target"] = copy.deepcopy(record["target"])
    return result


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty; use --force: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _state_from_spec(private_eval: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for spec in private_eval.get("replay_specs", []):
        if spec.get("name") == name:
            return spec["expected_state"]
    raise KeyError(f"missing replay spec {name}")


def _matching_feature(record: Mapping[str, Any], private_eval: Mapping[str, Any]) -> tuple[float, float]:
    setup = record["prompt_inputs"].get("setup", {})
    focal = float(setup.get("lens_focal_length_mm", 0.0))
    if record["task_type"] == "constrained_intervention":
        inputs = record["prompt_inputs"]
        return centroid_distance(inputs["current_observation"], inputs["target_observation"]), focal
    baseline = _state_from_spec(private_eval, "baseline") if record["task_type"] == "diagnosis" else None
    if baseline is not None:
        observed = _state_from_spec(private_eval, "observed")
        return centroid_distance(baseline, observed), focal
    current = record["prompt_inputs"]["current_observation"]
    return math.hypot(float(current["centroid_x_px"]), float(current["centroid_y_px"])), focal


def _greedy_nearest(
    anchors: list[tuple[dict[str, Any], dict[str, Any]]],
    candidates: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[tuple[dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]]:
    remaining = list(candidates)
    result = []
    for anchor in sorted(anchors, key=lambda item: _matching_feature(*item)):
        feature = _matching_feature(*anchor)
        index = min(
            range(len(remaining)),
            key=lambda idx: abs(feature[0] - _matching_feature(*remaining[idx])[0])
            + 0.05 * abs(feature[1] - _matching_feature(*remaining[idx])[1]),
        )
        result.append((anchor, remaining.pop(index)))
    if remaining:
        raise ValueError("unmatched corrective candidates remain")
    return result


def _set_match_metadata(
    members: tuple[tuple[dict[str, Any], dict[str, Any]], ...],
    match_id: str,
    kind: str,
) -> None:
    for record, private in members:
        record["provenance"]["match_group_id"] = match_id
        record["provenance"]["match_group_kind"] = kind
        private["match_group_id"] = match_id
        private["match_group_kind"] = kind


def assign_action_first_control_groups(
    controls: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    split: str,
    dataset_version: str,
) -> int:
    """Build feasible action pairs, attaching every infeasible row to one pair."""
    group_count = 0
    for actuator in ACTION_KEYS:
        actuator_rows = [
            entry
            for entry in controls
            if entry[0]["prompt_inputs"]["actuator_constraints"]["active_actuator"] == actuator
        ]
        feasible = [entry for entry in actuator_rows if entry[0]["target"]["status"] == "feasible"]
        infeasible = [
            entry
            for entry in actuator_rows
            if entry[0]["target"]["status"] == "infeasible_within_limits"
        ]

        def category(mode: str, sign: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
            return [
                entry
                for entry in feasible
                if entry[0]["provenance"].get("control_action_mode") == mode
                and entry[0]["provenance"].get("control_action_sign") == sign
            ]

        pairings = [
            (category("minimum_motion", "positive"), category("near_boundary", "negative")),
            (category("minimum_motion", "negative"), category("near_boundary", "positive")),
        ]
        action_pairs: list[
            tuple[tuple[dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]
        ] = []
        for left, right in pairings:
            if len(left) != len(right):
                raise ValueError(
                    f"action-pair imbalance for {split}/{actuator}: {len(left)} vs {len(right)}"
                )
            action_pairs.extend(_greedy_nearest(left, right))
        if len(action_pairs) * 2 != len(feasible):
            raise ValueError(f"unpaired feasible controls for {split}/{actuator}")
        if len(infeasible) > len(action_pairs):
            raise ValueError(f"more infeasible rows than action pairs for {split}/{actuator}")

        remaining_pair_indices = set(range(len(action_pairs)))
        attached: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        for entry in sorted(infeasible, key=lambda item: _matching_feature(*item)[0]):
            error = _matching_feature(*entry)[0]
            selected = min(
                remaining_pair_indices,
                key=lambda index: abs(
                    error
                    - sum(_matching_feature(*member)[0] for member in action_pairs[index]) / 2.0
                ),
            )
            attached[selected] = entry
            remaining_pair_indices.remove(selected)

        ordered_pairs = sorted(
            enumerate(action_pairs),
            key=lambda item: (
                sum(_matching_feature(*member)[0] for member in item[1]) / 2.0,
                item[1][0][0]["example_id"],
            ),
        )
        for local_index, (original_index, pair) in enumerate(ordered_pairs):
            third = attached.get(original_index)
            members = pair if third is None else (pair[0], pair[1], third)
            kind = "action_diversity_pair" if third is None else "action_diversity_triplet"
            match_id = f"{dataset_version}:{split}:control:{actuator}:{local_index:04d}"
            _set_match_metadata(tuple(members), match_id, kind)
            group_count += 1
    return group_count


def assign_corrective_match_groups(masters: list[dict[str, Any]], dataset_version: str) -> dict[str, int]:
    """Pair or triple opposite labels without exposing match IDs in prompts."""
    entries = [
        (item["record"], item["private_eval"])
        for master in masters
        for item in master["records"]
        if item["record"]["task_type"]
        in {"information_sufficiency", "diagnosis", "constrained_intervention"}
        and "status_schedule_index" in item["record"]["provenance"]
    ]
    counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        split_entries = [entry for entry in entries if entry[0]["split"] == split]

        controls = [entry for entry in split_entries if entry[0]["task_type"] == "constrained_intervention"]
        if dataset_version == "action_first_v3":
            counts[f"{split}:constrained_intervention"] += assign_action_first_control_groups(
                controls, split=split, dataset_version=dataset_version
            )
        else:
            for actuator in ACTION_KEYS:
                feasible = [
                    entry
                    for entry in controls
                    if entry[0]["prompt_inputs"]["actuator_constraints"]["active_actuator"] == actuator
                    and entry[0]["target"]["status"] == "feasible"
                ]
                infeasible = [
                    entry
                    for entry in controls
                    if entry[0]["prompt_inputs"]["actuator_constraints"]["active_actuator"] == actuator
                    and entry[0]["target"]["status"] == "infeasible_within_limits"
                ]
                if len(feasible) != len(infeasible):
                    raise ValueError(f"control match imbalance for {split}/{actuator}: {len(feasible)} vs {len(infeasible)}")
                # Baseline error is the prompt-visible nuisance covered by the
                # frozen control-pair gate.  Rank matching is the optimal
                # one-dimensional assignment for absolute error and avoids focal
                # length perturbing otherwise close pairs.
                control_pairs = zip(
                    sorted(feasible, key=lambda item: _matching_feature(*item)[0]),
                    sorted(infeasible, key=lambda item: _matching_feature(*item)[0]),
                )
                for index, pair in enumerate(control_pairs):
                    match_id = f"{dataset_version}:{split}:control:{actuator}:{index:04d}"
                    _set_match_metadata(tuple(pair), match_id, "status_contrast")
                    counts[f"{split}:constrained_intervention"] += 1

        sufficiency = [entry for entry in split_entries if entry[0]["task_type"] == "information_sufficiency"]
        answerable = [entry for entry in sufficiency if entry[0]["target"]["status"] == "answerable"]
        insufficient = [
            entry for entry in sufficiency if entry[0]["target"]["status"] == "insufficient_information"
        ]
        if len(answerable) != len(insufficient):
            raise ValueError(f"sufficiency match imbalance for {split}")
        for index, pair in enumerate(_greedy_nearest(answerable, insufficient)):
            match_id = f"{dataset_version}:{split}:sufficiency:{index:04d}"
            _set_match_metadata(tuple(pair), match_id, "status_contrast")
            counts[f"{split}:information_sufficiency"] += 1

        diagnosis = [entry for entry in split_entries if entry[0]["task_type"] == "diagnosis"]
        by_status = {
            status: [entry for entry in diagnosis if entry[0]["target"]["status"] == status]
            for status in ("unique", "ambiguous", "unsupported")
        }
        sizes = {len(values) for values in by_status.values()}
        if len(sizes) > 1:
            raise ValueError(f"diagnosis match imbalance for {split}: { {key: len(value) for key, value in by_status.items()} }")
        ambiguous_by_unique = {
            unique[0]["example_id"]: ambiguous
            for unique, ambiguous in _greedy_nearest(by_status["unique"], by_status["ambiguous"])
        }
        unsupported_by_unique = {
            unique[0]["example_id"]: unsupported
            for unique, unsupported in _greedy_nearest(by_status["unique"], by_status["unsupported"])
        }
        for index, unique in enumerate(sorted(by_status["unique"], key=lambda item: _matching_feature(*item))):
            match_id = f"{dataset_version}:{split}:diagnosis:{index:04d}"
            example_id = unique[0]["example_id"]
            _set_match_metadata(
                (unique, ambiguous_by_unique[example_id], unsupported_by_unique[example_id]),
                match_id,
                "status_contrast",
            )
            counts[f"{split}:diagnosis"] += 1
    return dict(sorted(counts.items()))


def build(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    simulation_cfg = config["simulation"]
    label_cfg = config["labels"]
    render_cfg = config["rendering"]
    seed = int(dataset_cfg["seed"])
    dataset_version = str(dataset_cfg.get("version", DATASET_VERSION))
    group_prefix = str(dataset_cfg.get("group_prefix", "case"))
    scenario_start_index = int(dataset_cfg.get("scenario_start_index", 0))
    status_schedules = configured_status_schedules(dataset_cfg)
    for split, task_schedules in status_schedules.items():
        for task_type, schedule in task_schedules.items():
            expected_task_count = int(dataset_cfg["task_counts"][split][task_type])
            if len(schedule) != expected_task_count:
                raise ValueError(
                    f"status schedule length mismatch for {split}/{task_type}: {len(schedule)} vs {expected_task_count}"
                )
    templates = load_yaml(TEMPLATE_PATH)
    base_config_path = ROOT / str(simulation_cfg["base_config"])
    base_cfg = load_sim_yaml(str(base_config_path))

    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "val", "test")}
    masters: list[dict[str, Any]] = []
    task_indices = Counter()
    split_task_indices: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    scenario_index = scenario_start_index
    generated_count = 0
    ood_parameters = list(simulation_cfg["ood_factors"])

    for split in ("train", "val", "test"):
        split_count = int(dataset_cfg["scenarios"][split])
        group_ids = [f"{group_prefix}_{scenario_index + index:06d}" for index in range(split_count)]
        assignments = assign_tasks_to_groups(
            group_ids,
            dataset_cfg["task_counts"][split],
            random.Random(seed + {"train": 100, "val": 200, "test": 300}[split]),
            int(dataset_cfg["questions_per_scenario"]),
        )
        visual_keys = select_visual_examples(
            assignments,
            int(dataset_cfg["visual_records"][split]),
            random.Random(seed + {"train": 101, "val": 201, "test": 301}[split]),
        )
        for local_index, group_id in enumerate(group_ids):
            ood = split == "test" and local_index >= split_count - int(dataset_cfg["ood_test_scenarios"])
            ood_parameter = ood_parameters[(local_index - (split_count - int(dataset_cfg["ood_test_scenarios"]))) % len(ood_parameters)] if ood else None
            ood_band = local_index % 2
            max_attempts = 64 if dataset_version == "action_first_v3" else 1
            last_error: RuntimeError | None = None
            for sampling_attempt in range(max_attempts):
                scenario_seed = (
                    seed * 1_000_000
                    + scenario_index
                    + sampling_attempt * 1_000_000_000_000
                )
                scenario_rng = random.Random(scenario_seed)
                sampled_cfg, sampled_values = sample_setup_config(
                    base_cfg,
                    simulation_cfg,
                    scenario_rng,
                    ood_parameter=ood_parameter,
                    ood_band=ood_band,
                )
                setup = setup_from_dict(copy.deepcopy(sampled_cfg))
                base = simulate_and_measure(setup)
                master_records = []
                try:
                    for task_type in assignments[group_id]:
                        task_index = task_indices[task_type]
                        split_task_index = split_task_indices[split][task_type]
                        task_status_schedule = status_schedules.get(split, {}).get(task_type)
                        desired_status = (
                            task_status_schedule[split_task_index]
                            if task_status_schedule is not None
                            else None
                        )
                        visual = (group_id, task_type) in visual_keys
                        record, private_eval = BUILDERS[task_type](
                            group_id=group_id,
                            split=split,
                            task_index=task_index,
                            scenario_seed=scenario_seed,
                            config=sampled_cfg,
                            base=base,
                            templates=templates,
                            rng=scenario_rng,
                            label_cfg=label_cfg,
                            visual=visual,
                            output_dir=output_dir,
                            render_cfg=render_cfg,
                            desired_status=desired_status,
                        )
                        record["provenance"]["dataset_version"] = dataset_version
                        if dataset_version == "action_first_v3":
                            record["provenance"]["scenario_sampling_attempt"] = sampling_attempt
                        if desired_status is not None:
                            record["provenance"]["status_schedule_index"] = split_task_index
                            record["provenance"]["desired_status"] = desired_status
                        master_records.append({"record": record, "private_eval": private_eval})
                    break
                except RuntimeError as error:
                    last_error = error
                    if dataset_version != "action_first_v3":
                        raise
            else:
                raise RuntimeError(
                    f"could not sample a physically supported scenario for {group_id} after {max_attempts} attempts"
                ) from last_error

            for task_type in assignments[group_id]:
                task_indices[task_type] += 1
                split_task_indices[split][task_type] += 1
            rows_by_split[split].extend(item["record"] for item in master_records)
            masters.append(
                {
                    "group_id": group_id,
                    "split": split,
                    "scenario_seed": scenario_seed,
                    "distribution": "ood" if ood else "iid",
                    "ood_parameter": ood_parameter,
                    "sampled_parameters": round_tree(sampled_values, 8),
                    "setup_config": setup_snapshot(setup),
                    "baseline_state": rounded_state(base["state"]),
                    "records": master_records,
                }
            )
            scenario_index += 1
            generated_count += 1
            if generated_count % 10 == 0 or generated_count == sum(int(v) for v in dataset_cfg["scenarios"].values()):
                print(f"[{scenario_index}] generated {group_id} ({split}, {'ood' if ood else 'iid'})", flush=True)

    matched_group_counts = assign_corrective_match_groups(masters, dataset_version)
    for split in rows_by_split:
        rows_by_split[split].sort(key=lambda row: row["example_id"])
    write_jsonl(output_dir / "master" / "cases.jsonl", masters)
    write_jsonl(output_dir / "canonical" / "train.jsonl", (canonical_public(row, True) for row in rows_by_split["train"]))
    write_jsonl(output_dir / "canonical" / "val.jsonl", (canonical_public(row, True) for row in rows_by_split["val"]))
    write_jsonl(output_dir / "canonical" / "test_prompts.jsonl", (canonical_public(row, False) for row in rows_by_split["test"]))
    write_jsonl(
        output_dir / "private" / "test_labels.jsonl",
        (
            {
                "example_id": item["record"]["example_id"],
                "group_id": master["group_id"],
                "target": item["record"]["target"],
                "private_eval": item["private_eval"],
            }
            for master in masters
            if master["split"] == "test"
            for item in master["records"]
        ),
    )
    for split, rows in rows_by_split.items():
        include_target = split != "test"
        write_jsonl(output_dir / "exports" / "messages" / f"{split}.jsonl", (make_messages(row, include_target) for row in rows))
        write_jsonl(output_dir / "exports" / "qwen" / f"{split}.jsonl", (make_qwen_record(row, include_target) for row in rows))

    counts = {split: len(rows) for split, rows in rows_by_split.items()}
    task_counts = {split: dict(Counter(row["task_type"] for row in rows)) for split, rows in rows_by_split.items()}
    visual_counts = {split: sum(row["modality"] == "visual" for row in rows) for split, rows in rows_by_split.items()}
    status_counts = {
        split: {
            task: dict(sorted(Counter(row["target"]["status"] for row in rows if row["task_type"] == task).items()))
            for task in ("information_sufficiency", "diagnosis", "constrained_intervention")
            if any(row["task_type"] == task for row in rows)
        }
        for split, rows in rows_by_split.items()
    }
    manifest = {
        "dataset": dataset_cfg["name"],
        "version": dataset_version,
        "seed": seed,
        "simulator": "optical_sim:fresnel_numpy",
        "base_config": str(base_config_path.relative_to(ROOT)),
        "scenario_counts": {key: int(value) for key, value in dataset_cfg["scenarios"].items()},
        "group_prefix": group_prefix,
        "scenario_start_index": scenario_start_index,
        "record_counts": counts,
        "task_counts": task_counts,
        "status_counts": status_counts,
        "configured_status_counts": copy.deepcopy(dataset_cfg.get("status_counts", {})),
        "matched_group_counts": matched_group_counts,
        "visual_counts": visual_counts,
        "visual_fraction": sum(visual_counts.values()) / sum(counts.values()),
        "ood_test_scenarios": int(dataset_cfg["ood_test_scenarios"]),
        "audit_requirements": {
            "counterbalanced_pair_minimum": int(
                dataset_cfg.get("audit_requirements", {}).get(
                    "counterbalanced_pair_minimum", 100 if counts["test"] else 0
                )
            )
        },
        "ground_truth_scope": "synthetic simulator truth; not laboratory validation",
        "config_hash": stable_json_hash(config),
        "files": {
            "master": "master/cases.jsonl",
            "train": "canonical/train.jsonl",
            "val": "canonical/val.jsonl",
            "test_prompts": "canonical/test_prompts.jsonl",
            "test_labels": "private/test_labels.jsonl",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_paths = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name not in {"checksums.sha256", "audit_report.json"}
    )
    checksum_lines = [f"{file_sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in checksum_paths]
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.smoke:
        config = smoke_overrides(config)
    prepare_output_dir(args.output_dir, args.force)
    manifest = build(config, args.output_dir)
    print(json.dumps(manifest["record_counts"], sort_keys=True))
    if not args.skip_audit:
        from .audit_dataset import audit

        report = audit(args.output_dir, replay=False, strict=not args.smoke)
        (args.output_dir / "audit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not report["passed"]:
            raise RuntimeError(f"Dataset audit failed: {report['failures']}")


if __name__ == "__main__":
    main()
