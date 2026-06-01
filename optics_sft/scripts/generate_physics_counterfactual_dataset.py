#!/usr/bin/env python3
"""Generate counterfactual physics SFT rows from paired optical scenarios."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.optical_elements import OpticalSetup, setup_from_dict
from optics_sft.physics.control_search import (
    ActionBounds,
    choose_control_action,
    clamp_action,
    estimate_local_jacobian,
)
from optics_sft.physics.metadata_policy import find_leakage_fields
from optics_sft.physics.rendering import random_render_params, save_intensity_png
from optics_sft.physics.sim_adapter import (
    Action,
    apply_action_to_setup,
    residual_error_px,
    setup_to_safe_metadata,
    simulate_and_measure,
)


M_TO_MM = 1e3
MM_TO_M = 1e-3
M_TO_NM = 1e9
NM_TO_M = 1e-9
M_TO_UM = 1e6
UM_TO_M = 1e-6

DATASET_NAME = "physics_sft_counterfactual_v1"
COUNTERFACTUAL_PARAMETERS = (
    "lens_focal_length_mm",
    "lens_to_camera_mm",
    "source_to_lens_mm",
    "beam_waist_mm",
    "pixel_size_um",
)
DEFAULT_LENS_ACTION_BOUND_MM = 0.06
DEFAULT_VISUAL_ANCHOR_BOUND_MM = 0.035
ACTION_CHANGE_THRESHOLD_MM = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate counterfactual physics SFT rows.")
    parser.add_argument("--output-dir", type=Path, default=Path("../VLM_data/physics_sft_counterfactual_v1"))
    parser.add_argument("--num-pairs", "--num-samples", dest="num_pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("optical_sim/configs/base_config.yaml"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--render-difficulty",
        choices=("clean", "medium", "hard"),
        default="medium",
    )
    return parser.parse_args()


def positive_range(center: float, low_factor: float, high_factor: float) -> tuple[float, float]:
    return max(center * low_factor, 1e-12), max(center * high_factor, 1e-12)


def parameter_ranges_from_config(cfg: dict[str, Any]) -> dict[str, list[float]]:
    source = cfg["source"]
    lens = cfg["lens"]
    geometry = cfg["geometry"]
    sensor = cfg["sensor"]
    wavelength_nm = float(source["wavelength"]) * M_TO_NM
    beam_waist_mm = float(source["beam_waist"]) * M_TO_MM
    focal_mm = float(lens["focal_length"]) * M_TO_MM
    aperture_mm = float(lens["clear_aperture"]) * M_TO_MM
    source_to_lens_mm = float(geometry["laser_to_lens"]) * M_TO_MM
    lens_to_camera_mm = float(geometry["lens_to_camera"]) * M_TO_MM
    pixel_size_um = float(sensor["pixel_pitch"]) * M_TO_UM

    ranges = {
        "wavelength_nm": positive_range(wavelength_nm, 0.98, 1.02),
        "beam_waist_mm": positive_range(beam_waist_mm, 0.8, 1.25),
        "lens_focal_length_mm": positive_range(focal_mm, 0.85, 1.15),
        "lens_aperture_mm": positive_range(aperture_mm, 0.9, 1.1),
        "source_to_lens_mm": positive_range(source_to_lens_mm, 0.85, 1.15),
        "lens_to_camera_mm": positive_range(lens_to_camera_mm, 0.85, 1.15),
        "pixel_size_um": positive_range(pixel_size_um, 0.9, 1.1),
        "lens_x_offset_mm": (-0.18, 0.18),
        "lens_y_offset_mm": (-0.18, 0.18),
        "camera_x_offset_mm": (-0.1, 0.1),
        "camera_y_offset_mm": (-0.1, 0.1),
    }
    return {key: [float(value[0]), float(value[1])] for key, value in ranges.items()}


def sample_uniform(rng: random.Random, ranges: dict[str, list[float]], key: str) -> float:
    low, high = ranges[key]
    return rng.uniform(low, high)


def varied_setup_from_base(
    base_cfg: dict[str, Any],
    ranges: dict[str, list[float]],
    rng: random.Random,
) -> tuple[OpticalSetup, dict[str, float]]:
    cfg = copy.deepcopy(base_cfg)
    sampled = {
        "wavelength_nm": sample_uniform(rng, ranges, "wavelength_nm"),
        "beam_waist_mm": sample_uniform(rng, ranges, "beam_waist_mm"),
        "lens_focal_length_mm": sample_uniform(rng, ranges, "lens_focal_length_mm"),
        "lens_aperture_mm": sample_uniform(rng, ranges, "lens_aperture_mm"),
        "source_to_lens_mm": sample_uniform(rng, ranges, "source_to_lens_mm"),
        "lens_to_camera_mm": sample_uniform(rng, ranges, "lens_to_camera_mm"),
        "pixel_size_um": sample_uniform(rng, ranges, "pixel_size_um"),
        "lens_x_offset_mm": sample_uniform(rng, ranges, "lens_x_offset_mm"),
        "lens_y_offset_mm": sample_uniform(rng, ranges, "lens_y_offset_mm"),
        "camera_x_offset_mm": sample_uniform(rng, ranges, "camera_x_offset_mm"),
        "camera_y_offset_mm": sample_uniform(rng, ranges, "camera_y_offset_mm"),
    }

    cfg["source"]["wavelength"] = sampled["wavelength_nm"] * NM_TO_M
    cfg["source"]["beam_waist"] = sampled["beam_waist_mm"] * MM_TO_M
    cfg["lens"]["focal_length"] = sampled["lens_focal_length_mm"] * MM_TO_M
    cfg["lens"]["clear_aperture"] = sampled["lens_aperture_mm"] * MM_TO_M
    cfg["lens"]["x_offset"] = sampled["lens_x_offset_mm"] * MM_TO_M
    cfg["lens"]["y_offset"] = sampled["lens_y_offset_mm"] * MM_TO_M
    cfg["sensor"]["pixel_pitch"] = sampled["pixel_size_um"] * UM_TO_M
    cfg["geometry"]["laser_to_lens"] = sampled["source_to_lens_mm"] * MM_TO_M
    cfg["geometry"]["lens_to_camera"] = sampled["lens_to_camera_mm"] * MM_TO_M
    cfg.setdefault("camera", {})
    cfg["camera"]["x_offset"] = sampled["camera_x_offset_mm"] * MM_TO_M
    cfg["camera"]["y_offset"] = sampled["camera_y_offset_mm"] * MM_TO_M
    return setup_from_dict(cfg), sampled


def setup_snapshot(setup: OpticalSetup) -> dict[str, Any]:
    return {
        "source": {
            "wavelength": float(setup.source.wavelength),
            "beam_waist": float(setup.source.beam_waist),
            "power": float(setup.source.power),
            "type": setup.source.source_type,
        },
        "lens": {
            "focal_length": float(setup.lens.focal_length),
            "clear_aperture": float(setup.lens.clear_aperture),
            "diameter": float(setup.lens.diameter),
            "x_offset": float(setup.lens.x_offset),
            "y_offset": float(setup.lens.y_offset),
        },
        "sensor": {
            "resolution": [int(setup.sensor.resolution[0]), int(setup.sensor.resolution[1])],
            "pixel_pitch": float(setup.sensor.pixel_pitch),
        },
        "geometry": {
            "laser_to_lens": float(setup.laser_to_lens),
            "lens_to_camera": float(setup.lens_to_camera),
        },
        "camera": {
            "x_offset": float(setup.camera.x_offset),
            "y_offset": float(setup.camera.y_offset),
        },
        "simulation": {
            "grid_size": int(setup.grid_size),
            "grid_extent": float(setup.grid_extent),
            "propagation_backend": setup.propagation_backend,
        },
    }


def rounded_state(state: Mapping[str, Any]) -> dict[str, float]:
    return {key: round(float(value), 8) for key, value in state.items()}


def action_to_dict(action: Action) -> dict[str, float]:
    return {
        "lens_x_delta_mm": round(float(action.lens_x_delta_mm), 8),
        "lens_y_delta_mm": round(float(action.lens_y_delta_mm), 8),
        "camera_x_delta_mm": round(float(action.camera_x_delta_mm), 8),
        "camera_y_delta_mm": round(float(action.camera_y_delta_mm), 8),
    }


def error_vector_px(source_state: Mapping[str, Any], target_state: Mapping[str, Any]) -> dict[str, float]:
    return {
        "x": round(float(source_state["centroid_x_px"]) - float(target_state["centroid_x_px"]), 8),
        "y": round(float(source_state["centroid_y_px"]) - float(target_state["centroid_y_px"]), 8),
    }


def centroid_shift_px(source_state: Mapping[str, Any], target_state: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(target_state["centroid_x_px"]) - float(source_state["centroid_x_px"]),
            float(target_state["centroid_y_px"]) - float(source_state["centroid_y_px"]),
        ],
        dtype=np.float64,
    )


def split_assignments(num_pairs: int, val_ratio: float, test_ratio: float, seed: int) -> dict[int, str]:
    if num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio and --test-ratio must be non-negative and sum to less than 1")
    indices = list(range(num_pairs))
    random.Random(seed).shuffle(indices)
    val_count = int(round(num_pairs * val_ratio))
    test_count = int(round(num_pairs * test_ratio))
    val_indices = set(indices[:val_count])
    test_indices = set(indices[val_count : val_count + test_count])
    return {
        index: "val" if index in val_indices else "test" if index in test_indices else "train"
        for index in range(num_pairs)
    }


def sample_visual_anchor_action(rng: random.Random) -> Action:
    for _ in range(50):
        action = Action(
            lens_x_delta_mm=rng.uniform(-DEFAULT_VISUAL_ANCHOR_BOUND_MM, DEFAULT_VISUAL_ANCHOR_BOUND_MM),
            lens_y_delta_mm=rng.uniform(-DEFAULT_VISUAL_ANCHOR_BOUND_MM, DEFAULT_VISUAL_ANCHOR_BOUND_MM),
            camera_x_delta_mm=0.0,
            camera_y_delta_mm=0.0,
        )
        if abs(action.lens_x_delta_mm) + abs(action.lens_y_delta_mm) >= 0.012:
            return action
    return action


def changed_parameter_value(setup: OpticalSetup, parameter: str) -> float:
    if parameter == "lens_focal_length_mm":
        return float(setup.lens.focal_length) * M_TO_MM
    if parameter == "lens_to_camera_mm":
        return float(setup.lens_to_camera) * M_TO_MM
    if parameter == "source_to_lens_mm":
        return float(setup.laser_to_lens) * M_TO_MM
    if parameter == "beam_waist_mm":
        return float(setup.source.beam_waist) * M_TO_MM
    if parameter == "pixel_size_um":
        return float(setup.sensor.pixel_pitch) * M_TO_UM
    raise ValueError(f"Unsupported counterfactual parameter: {parameter}")


def apply_counterfactual_change(
    setup: OpticalSetup,
    parameter: str,
    rng: random.Random,
) -> tuple[OpticalSetup, dict[str, float | str]]:
    changed = copy.deepcopy(setup)
    before_value = changed_parameter_value(changed, parameter)
    factor_options = {
        "lens_focal_length_mm": (0.72, 1.28),
        "lens_to_camera_mm": (0.72, 1.28),
        "source_to_lens_mm": (0.72, 1.28),
        "beam_waist_mm": (0.65, 1.55),
        "pixel_size_um": (0.75, 1.25),
    }
    factor = rng.choice(factor_options[parameter])

    if parameter == "lens_focal_length_mm":
        changed.lens.focal_length = before_value * factor * MM_TO_M
    elif parameter == "lens_to_camera_mm":
        changed.lens_to_camera = before_value * factor * MM_TO_M
    elif parameter == "source_to_lens_mm":
        changed.laser_to_lens = before_value * factor * MM_TO_M
    elif parameter == "beam_waist_mm":
        changed.source.beam_waist = before_value * factor * MM_TO_M
    elif parameter == "pixel_size_um":
        changed.sensor.pixel_pitch = before_value * factor * UM_TO_M

    return changed, {
        "parameter": parameter,
        "scenario_a_value": round(before_value, 8),
        "scenario_b_value": round(changed_parameter_value(changed, parameter), 8),
        "factor": round(float(factor), 8),
    }


def solve_action_for_visual_shift(
    setup: OpticalSetup,
    current_state: Mapping[str, Any],
    desired_shift_px: np.ndarray,
    bounds: ActionBounds,
    fallback_action: Action,
) -> Action:
    jacobian = estimate_local_jacobian(setup, current_state=current_state)
    if jacobian is None:
        return clamp_action(fallback_action, bounds)

    matrix = np.asarray(jacobian["matrix"], dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        return clamp_action(fallback_action, bounds)
    try:
        lens_delta = np.linalg.lstsq(matrix, desired_shift_px, rcond=None)[0]
    except np.linalg.LinAlgError:
        return clamp_action(fallback_action, bounds)
    if not np.all(np.isfinite(lens_delta)):
        return clamp_action(fallback_action, bounds)

    return clamp_action(
        Action(
            lens_x_delta_mm=float(lens_delta[0]),
            lens_y_delta_mm=float(lens_delta[1]),
            camera_x_delta_mm=0.0,
            camera_y_delta_mm=0.0,
        ),
        bounds,
    )


def simulate_scenario(
    setup: OpticalSetup,
    target_action: Action,
    bounds: ActionBounds,
) -> dict[str, Any]:
    target_setup = apply_action_to_setup(setup, target_action)
    current = simulate_and_measure(setup)
    target = simulate_and_measure(target_setup)
    search_result = choose_control_action(
        setup,
        target["state"],
        bounds,
        grid_size=5,
        current_state=current["state"],
    )
    return {
        "setup": setup,
        "target_setup": target_setup,
        "target_action": target_action,
        "current": current,
        "target": target,
        "search_result": search_result,
        "initial_error_norm_px": residual_error_px(current["state"], target["state"]),
        "post_action_error_norm_px": float(search_result["post_action_error_px"]),
        "visual_shift_px": centroid_shift_px(current["state"], target["state"]),
    }


def action_difference(a_action: Action, b_action: Action) -> dict[str, Any]:
    dx = float(b_action.lens_x_delta_mm) - float(a_action.lens_x_delta_mm)
    dy = float(b_action.lens_y_delta_mm) - float(a_action.lens_y_delta_mm)
    dcx = float(b_action.camera_x_delta_mm) - float(a_action.camera_x_delta_mm)
    dcy = float(b_action.camera_y_delta_mm) - float(a_action.camera_y_delta_mm)
    lens_l2 = math.hypot(dx, dy)
    camera_l2 = math.hypot(dcx, dcy)
    components = {
        "lens_x_delta_mm": abs(dx),
        "lens_y_delta_mm": abs(dy),
        "camera_x_delta_mm": abs(dcx),
        "camera_y_delta_mm": abs(dcy),
    }
    dominant = max(components, key=components.get)
    return {
        "lens_x_difference_mm": round(dx, 8),
        "lens_y_difference_mm": round(dy, 8),
        "camera_x_difference_mm": round(dcx, 8),
        "camera_y_difference_mm": round(dcy, 8),
        "lens_delta_l2_mm": round(lens_l2, 8),
        "camera_delta_l2_mm": round(camera_l2, 8),
        "total_delta_l2_mm": round(math.hypot(lens_l2, camera_l2), 8),
        "dominant_difference": dominant,
    }


def action_changed(summary: Mapping[str, Any]) -> bool:
    return float(summary["total_delta_l2_mm"]) > ACTION_CHANGE_THRESHOLD_MM


def visual_offset_summary(a_scenario: Mapping[str, Any], b_scenario: Mapping[str, Any]) -> dict[str, Any]:
    a_shift = np.asarray(a_scenario["visual_shift_px"], dtype=np.float64)
    b_shift = np.asarray(b_scenario["visual_shift_px"], dtype=np.float64)
    diff = b_shift - a_shift
    return {
        "scenario_a_shift_px": {"x": round(float(a_shift[0]), 8), "y": round(float(a_shift[1]), 8)},
        "scenario_b_shift_px": {"x": round(float(b_shift[0]), 8), "y": round(float(b_shift[1]), 8)},
        "shift_difference_px": {"x": round(float(diff[0]), 8), "y": round(float(diff[1]), 8)},
        "shift_difference_norm_px": round(float(np.linalg.norm(diff)), 8),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def scenario_private_eval(scenario: Mapping[str, Any]) -> dict[str, Any]:
    search_result = scenario["search_result"]
    return {
        "current_state": rounded_state(scenario["current"]["state"]),
        "target_state": rounded_state(scenario["target"]["state"]),
        "after_state": rounded_state(search_result["after_state"]),
        "initial_error_px": error_vector_px(scenario["current"]["state"], scenario["target"]["state"]),
        "post_action_error_px": error_vector_px(search_result["after_state"], scenario["target"]["state"]),
        "initial_error_norm_px": round(float(scenario["initial_error_norm_px"]), 8),
        "post_action_error_norm_px": round(float(scenario["post_action_error_norm_px"]), 8),
        "target_generating_action": action_to_dict(scenario["target_action"]),
        "searched_control_action": action_to_dict(search_result["action"]),
        "search_method": search_result.get("method"),
        "jacobian_condition": search_result.get("jacobian_condition"),
        "current_setup": setup_snapshot(scenario["setup"]),
        "target_setup": setup_snapshot(scenario["target_setup"]),
    }


def build_row(
    sample_id: str,
    split: str,
    changed_parameter: str,
    change_record: Mapping[str, Any],
    scenario_a: Mapping[str, Any],
    scenario_b: Mapping[str, Any],
    image_paths: Mapping[str, str],
    render_options: Mapping[str, Any],
    config_path: Path,
    bounds: ActionBounds,
) -> dict[str, Any]:
    scenario_a_action = scenario_a["search_result"]["action"]
    scenario_b_action = scenario_b["search_result"]["action"]
    difference = action_difference(scenario_a_action, scenario_b_action)
    prompt_inputs = {
        "images": dict(image_paths),
        "safe_setup_metadata": {
            "scenario_a": setup_to_safe_metadata(scenario_a["setup"]),
            "scenario_b": setup_to_safe_metadata(scenario_b["setup"]),
        },
    }
    leakage_fields = find_leakage_fields(prompt_inputs)
    if leakage_fields:
        raise ValueError(f"Prompt inputs contain leakage fields: {', '.join(sorted(leakage_fields))}")

    offset_summary = visual_offset_summary(scenario_a, scenario_b)
    return {
        "sample_id": sample_id,
        "sample_type": "counterfactual_pair",
        "prompt_inputs": prompt_inputs,
        "target": {
            "task": "counterfactual_action_comparison",
            "changed_parameter": changed_parameter,
            "should_action_change": action_changed(difference),
            "scenario_a_control_plan": action_to_dict(scenario_a_action),
            "scenario_b_control_plan": action_to_dict(scenario_b_action),
            "action_difference_summary": difference,
            "physics_reasoning": {
                "label_source": "simulator_action_search",
                "counterfactual_design": "The two scenarios are rendered with similar current-to-target centroid shifts while one setup variable changes.",
                "changed_setup_variable": dict(change_record),
                "visual_offset_summary": offset_summary,
                "expected_model_behavior": "Compare scenario metadata before choosing whether the action should change.",
            },
        },
        "private_eval": {
            "scenario_a": scenario_private_eval(scenario_a),
            "scenario_b": scenario_private_eval(scenario_b),
            "simulator_configs": {
                "config_path": str(config_path),
                "changed_parameter": changed_parameter,
                "changed_parameter_values": dict(change_record),
                "render_options": dict(render_options),
                "action_bounds": {
                    "lens_x_mm": list(bounds.lens_x_mm),
                    "lens_y_mm": list(bounds.lens_y_mm),
                    "camera_x_mm": list(bounds.camera_x_mm),
                    "camera_y_mm": list(bounds.camera_y_mm),
                },
                "visual_offset_summary": offset_summary,
            },
        },
        "split_tags": [
            split,
            "counterfactual",
            changed_parameter,
            DATASET_NAME,
        ],
    }


def generate_rows(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base_cfg = load_yaml(str(args.config))
    ranges = parameter_ranges_from_config(base_cfg)
    assignments = split_assignments(args.num_pairs, args.val_ratio, args.test_ratio, args.seed)
    rng = random.Random(args.seed)
    image_root = args.output_dir / "images"
    bounds = ActionBounds(
        lens_x_mm=(-DEFAULT_LENS_ACTION_BOUND_MM, DEFAULT_LENS_ACTION_BOUND_MM),
        lens_y_mm=(-DEFAULT_LENS_ACTION_BOUND_MM, DEFAULT_LENS_ACTION_BOUND_MM),
    )
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for index in range(args.num_pairs):
        sample_id = f"phys_cf_{index:06d}"
        split = assignments[index]
        changed_parameter = COUNTERFACTUAL_PARAMETERS[index % len(COUNTERFACTUAL_PARAMETERS)]

        scenario_a_setup, sampled_setup = varied_setup_from_base(base_cfg, ranges, rng)
        scenario_b_setup, change_record = apply_counterfactual_change(
            scenario_a_setup,
            changed_parameter,
            rng,
        )

        scenario_a_anchor = sample_visual_anchor_action(rng)
        scenario_a_current = simulate_and_measure(scenario_a_setup)
        scenario_a_target_setup = apply_action_to_setup(scenario_a_setup, scenario_a_anchor)
        scenario_a_target = simulate_and_measure(scenario_a_target_setup)
        desired_shift = centroid_shift_px(scenario_a_current["state"], scenario_a_target["state"])

        scenario_b_current = simulate_and_measure(scenario_b_setup)
        scenario_b_anchor = solve_action_for_visual_shift(
            scenario_b_setup,
            scenario_b_current["state"],
            desired_shift,
            bounds,
            fallback_action=scenario_a_anchor,
        )

        scenario_a = simulate_scenario(scenario_a_setup, scenario_a_anchor, bounds)
        scenario_b = simulate_scenario(scenario_b_setup, scenario_b_anchor, bounds)

        image_paths = {
            "scenario_a_current_image_path": f"{split}/{sample_id}_scenario_a_current.png",
            "scenario_a_target_image_path": f"{split}/{sample_id}_scenario_a_target.png",
            "scenario_b_current_image_path": f"{split}/{sample_id}_scenario_b_current.png",
            "scenario_b_target_image_path": f"{split}/{sample_id}_scenario_b_target.png",
        }
        render_options = random_render_params(rng, difficulty=args.render_difficulty)
        save_intensity_png(
            scenario_a["current"]["intensity"],
            image_root / image_paths["scenario_a_current_image_path"],
            render_options,
        )
        save_intensity_png(
            scenario_a["target"]["intensity"],
            image_root / image_paths["scenario_a_target_image_path"],
            render_options,
        )
        save_intensity_png(
            scenario_b["current"]["intensity"],
            image_root / image_paths["scenario_b_current_image_path"],
            render_options,
        )
        save_intensity_png(
            scenario_b["target"]["intensity"],
            image_root / image_paths["scenario_b_target_image_path"],
            render_options,
        )

        row = build_row(
            sample_id=sample_id,
            split=split,
            changed_parameter=changed_parameter,
            change_record=change_record,
            scenario_a=scenario_a,
            scenario_b=scenario_b,
            image_paths=image_paths,
            render_options=render_options,
            config_path=args.config,
            bounds=bounds,
        )
        row["private_eval"]["simulator_configs"]["sampled_scenario_a_setup_parameters"] = sampled_setup
        rows_by_split[split].append(row)

        if (index + 1) % 25 == 0 or index + 1 == args.num_pairs:
            print(f"[{index + 1}/{args.num_pairs}] generated {sample_id}")

    manifest = {
        "dataset": DATASET_NAME,
        "sample_type": "counterfactual_pair",
        "num_pairs": args.num_pairs,
        "counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "seed": args.seed,
        "config_path": str(args.config),
        "image_root": "images",
        "splits": {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "test": "test.jsonl",
        },
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "render_difficulty": args.render_difficulty,
        "counterfactual_parameters": list(COUNTERFACTUAL_PARAMETERS),
        "parameter_ranges": ranges,
        "action_bounds": {
            "lens_x_mm": list(bounds.lens_x_mm),
            "lens_y_mm": list(bounds.lens_y_mm),
            "camera_x_mm": list(bounds.camera_x_mm),
            "camera_y_mm": list(bounds.camera_y_mm),
        },
        "action_change_threshold_mm": ACTION_CHANGE_THRESHOLD_MM,
    }
    return rows_by_split, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split, manifest = generate_rows(args)
    for split, rows in rows_by_split.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote dataset to {args.output_dir}")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
