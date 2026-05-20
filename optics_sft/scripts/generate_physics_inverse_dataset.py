#!/usr/bin/env python3
"""Generate inverse-control physics SFT rows with simulator action search."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.optical_elements import OpticalSetup, setup_from_dict
from optics_sft.physics.control_search import ActionBounds, choose_control_action
from optics_sft.physics.metadata_policy import assert_no_prompt_leakage
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
DEFAULT_LENS_ACTION_BOUND_MM = 0.06
DEFAULT_TARGET_LENS_SHIFT_MM = 0.04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inverse-control physics SFT rows.")
    parser.add_argument("--output-dir", type=Path, default=Path("../VLM_data/physics_sft_inverse_v1"))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("optical_sim/configs/base_config.yaml"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def positive_range(center: float, low_factor: float, high_factor: float) -> tuple[float, float]:
    return max(center * low_factor, 1e-12), max(center * high_factor, 1e-12)


def parameter_ranges_from_config(cfg: dict[str, Any]) -> dict[str, list[float]]:
    source = cfg["source"]
    lens = cfg["lens"]
    geometry = cfg["geometry"]
    wavelength_nm = float(source["wavelength"]) * M_TO_NM
    beam_waist_mm = float(source["beam_waist"]) * M_TO_MM
    focal_mm = float(lens["focal_length"]) * M_TO_MM
    aperture_mm = float(lens["clear_aperture"]) * M_TO_MM
    source_to_lens_mm = float(geometry["laser_to_lens"]) * M_TO_MM
    lens_to_camera_mm = float(geometry["lens_to_camera"]) * M_TO_MM

    ranges = {
        "wavelength_nm": positive_range(wavelength_nm, 0.95, 1.05),
        "beam_waist_mm": positive_range(beam_waist_mm, 0.6, 1.8),
        "lens_focal_length_mm": positive_range(focal_mm, 0.7, 1.3),
        "lens_aperture_mm": positive_range(aperture_mm, 0.75, 1.15),
        "source_to_lens_mm": positive_range(source_to_lens_mm, 0.75, 1.25),
        "lens_to_camera_mm": positive_range(lens_to_camera_mm, 0.75, 1.25),
        "lens_x_offset_mm": (-0.22, 0.22),
        "lens_y_offset_mm": (-0.22, 0.22),
        "camera_x_offset_mm": (-0.12, 0.12),
        "camera_y_offset_mm": (-0.12, 0.12),
        "target_lens_x_delta_mm": (-DEFAULT_TARGET_LENS_SHIFT_MM, DEFAULT_TARGET_LENS_SHIFT_MM),
        "target_lens_y_delta_mm": (-DEFAULT_TARGET_LENS_SHIFT_MM, DEFAULT_TARGET_LENS_SHIFT_MM),
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


def split_assignments(num_samples: int, val_ratio: float, test_ratio: float, seed: int) -> dict[int, str]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio and --test-ratio must be non-negative and sum to less than 1")
    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)
    val_count = int(round(num_samples * val_ratio))
    test_count = int(round(num_samples * test_ratio))
    val_indices = set(indices[:val_count])
    test_indices = set(indices[val_count : val_count + test_count])
    return {
        index: "val" if index in val_indices else "test" if index in test_indices else "train"
        for index in range(num_samples)
    }


def rounded_state(state: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(value), 8) for key, value in state.items()}


def action_to_dict(action: Action) -> dict[str, float]:
    return {
        "lens_x_delta_mm": round(float(action.lens_x_delta_mm), 8),
        "lens_y_delta_mm": round(float(action.lens_y_delta_mm), 8),
        "camera_x_delta_mm": round(float(action.camera_x_delta_mm), 8),
        "camera_y_delta_mm": round(float(action.camera_y_delta_mm), 8),
    }


def error_vector_px(source_state: dict[str, Any], target_state: dict[str, Any]) -> dict[str, float]:
    return {
        "x": round(float(source_state["centroid_x_px"]) - float(target_state["centroid_x_px"]), 8),
        "y": round(float(source_state["centroid_y_px"]) - float(target_state["centroid_y_px"]), 8),
    }


def direction(value: float, negative_label: str, positive_label: str) -> str:
    if abs(value) < 1.0:
        return "near_center"
    return positive_label if value > 0 else negative_label


def error_level(error_norm_px: float) -> str:
    if error_norm_px < 3.0:
        return "small"
    if error_norm_px < 15.0:
        return "medium"
    return "large"


def confidence_from_errors(initial_error: float, post_error: float) -> float:
    if initial_error <= 1e-9:
        return 0.5
    improvement = max(0.0, min(1.0, 1.0 - post_error / initial_error))
    return round(0.45 + 0.5 * improvement, 4)


def sample_hidden_target_action(rng: random.Random, ranges: dict[str, list[float]]) -> Action:
    for _ in range(50):
        action = Action(
            lens_x_delta_mm=sample_uniform(rng, ranges, "target_lens_x_delta_mm"),
            lens_y_delta_mm=sample_uniform(rng, ranges, "target_lens_y_delta_mm"),
            camera_x_delta_mm=0.0,
            camera_y_delta_mm=0.0,
        )
        if abs(action.lens_x_delta_mm) + abs(action.lens_y_delta_mm) >= 0.015:
            return action
    return action


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_target(
    current_state: dict[str, Any],
    target_state: dict[str, Any],
    search_result: dict[str, Any],
) -> dict[str, Any]:
    initial_error = residual_error_px(current_state, target_state)
    post_error = float(search_result["post_action_error_px"])
    current_minus_target = error_vector_px(current_state, target_state)
    after_minus_target = error_vector_px(search_result["after_state"], target_state)
    return {
        "task": "physics_aware_beam_alignment",
        "perception": {
            "current_relative_to_target_x": direction(current_minus_target["x"], "left", "right"),
            "current_relative_to_target_y": direction(current_minus_target["y"], "above", "below"),
            "initial_error_level": error_level(initial_error),
        },
        "physics_reasoning": [
            "Compare the current and target beam centroids and widths.",
            "Choose the actuator delta that minimizes simulated post-action centroid error.",
            f"The selected action was produced by {search_result.get('method', 'simulator_search')}.",
        ],
        "physics_reasoning_fields": {
            "label_source": "simulator_action_search",
            "search_method": search_result.get("method", "unknown"),
            "post_action_error_level": error_level(post_error),
        },
        "control_plan": action_to_dict(search_result["action"]),
        "prediction": {
            "expected_residual_error_px": round(post_error, 8),
            "expected_residual_vector_px": after_minus_target,
        },
        "confidence": confidence_from_errors(initial_error, post_error),
    }


def generate_rows(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base_cfg = load_yaml(str(args.config))
    ranges = parameter_ranges_from_config(base_cfg)
    assignments = split_assignments(args.num_samples, args.val_ratio, args.test_ratio, args.seed)
    bounds = ActionBounds(
        lens_x_mm=(-DEFAULT_LENS_ACTION_BOUND_MM, DEFAULT_LENS_ACTION_BOUND_MM),
        lens_y_mm=(-DEFAULT_LENS_ACTION_BOUND_MM, DEFAULT_LENS_ACTION_BOUND_MM),
    )
    rng = random.Random(args.seed)
    image_root = args.output_dir / "images"
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for index in range(args.num_samples):
        sample_id = f"phys_inv_{index:06d}"
        split = assignments[index]
        current_setup, sampled_setup = varied_setup_from_base(base_cfg, ranges, rng)
        hidden_target_action = sample_hidden_target_action(rng, ranges)
        target_setup = apply_action_to_setup(current_setup, hidden_target_action)

        current = simulate_and_measure(current_setup)
        target = simulate_and_measure(target_setup)
        search_result = choose_control_action(
            current_setup,
            target["state"],
            bounds,
            grid_size=5,
            current_state=current["state"],
        )

        current_rel = f"{split}/{sample_id}_current.png"
        target_rel = f"{split}/{sample_id}_target.png"
        render_options = random_render_params(rng, difficulty="medium")
        save_intensity_png(current["intensity"], image_root / current_rel, render_options)
        save_intensity_png(target["intensity"], image_root / target_rel, render_options)

        prompt_inputs = {
            "images": {
                "current_image_path": current_rel,
                "target_image_path": target_rel,
            },
            "safe_setup_metadata": setup_to_safe_metadata(current_setup),
        }
        assert_no_prompt_leakage(prompt_inputs)

        initial_error = residual_error_px(current["state"], target["state"])
        post_action_vector = error_vector_px(search_result["after_state"], target["state"])
        row = {
            "sample_id": sample_id,
            "sample_type": "inverse_control",
            "prompt_inputs": prompt_inputs,
            "target": build_target(current["state"], target["state"], search_result),
            "private_eval": {
                "current_state": rounded_state(current["state"]),
                "target_state": rounded_state(target["state"]),
                "after_state": rounded_state(search_result["after_state"]),
                "true_control_plan": action_to_dict(search_result["action"]),
                "initial_error_px": error_vector_px(current["state"], target["state"]),
                "post_action_error_px": post_action_vector,
                "initial_error_norm_px": round(initial_error, 8),
                "post_action_error_norm_px": round(float(search_result["post_action_error_px"]), 8),
                "simulator_config": {
                    "config_path": str(args.config),
                    "sampled_setup_parameters": sampled_setup,
                    "hidden_target_action": action_to_dict(hidden_target_action),
                    "current_setup": setup_snapshot(current_setup),
                    "target_setup": setup_snapshot(target_setup),
                    "search_method": search_result.get("method"),
                    "jacobian_condition": search_result.get("jacobian_condition"),
                    "action_bounds": {
                        "lens_x_mm": list(bounds.lens_x_mm),
                        "lens_y_mm": list(bounds.lens_y_mm),
                        "camera_x_mm": list(bounds.camera_x_mm),
                        "camera_y_mm": list(bounds.camera_y_mm),
                    },
                    "render_options": render_options,
                },
            },
            "split_tags": [
                split,
                "inverse_control",
                "physics_sft_inverse_v1",
            ],
        }
        rows_by_split[split].append(row)

        if (index + 1) % 25 == 0 or index + 1 == args.num_samples:
            print(f"[{index + 1}/{args.num_samples}] generated {sample_id}")

    manifest = {
        "dataset": "physics_sft_inverse_v1",
        "sample_type": "inverse_control",
        "num_samples": args.num_samples,
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
        "parameter_ranges": ranges,
        "action_bounds": {
            "lens_x_mm": list(bounds.lens_x_mm),
            "lens_y_mm": list(bounds.lens_y_mm),
            "camera_x_mm": list(bounds.camera_x_mm),
            "camera_y_mm": list(bounds.camera_y_mm),
        },
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
