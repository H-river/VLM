#!/usr/bin/env python3
"""Generate closed-loop trajectory physics SFT rows from optical_sim."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

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
DEFAULT_TARGET_LENS_SHIFT_MM = 0.045
DATASET_NAME = "physics_sft_trajectory_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate closed-loop trajectory physics SFT rows.")
    parser.add_argument("--output-dir", type=Path, default=Path("../VLM_data/physics_sft_trajectory_v1"))
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--success-threshold-px", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("optical_sim/configs/base_config.yaml"))
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
    wavelength_nm = float(source["wavelength"]) * M_TO_NM
    beam_waist_mm = float(source["beam_waist"]) * M_TO_MM
    focal_mm = float(lens["focal_length"]) * M_TO_MM
    aperture_mm = float(lens["clear_aperture"]) * M_TO_MM
    source_to_lens_mm = float(geometry["laser_to_lens"]) * M_TO_MM
    lens_to_camera_mm = float(geometry["lens_to_camera"]) * M_TO_MM

    ranges = {
        "wavelength_nm": positive_range(wavelength_nm, 0.95, 1.05),
        "beam_waist_mm": positive_range(beam_waist_mm, 0.65, 1.6),
        "lens_focal_length_mm": positive_range(focal_mm, 0.75, 1.25),
        "lens_aperture_mm": positive_range(aperture_mm, 0.8, 1.12),
        "source_to_lens_mm": positive_range(source_to_lens_mm, 0.8, 1.2),
        "lens_to_camera_mm": positive_range(lens_to_camera_mm, 0.8, 1.2),
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


def sample_target_action(rng: random.Random, ranges: dict[str, list[float]]) -> Action:
    for _ in range(50):
        action = Action(
            lens_x_delta_mm=sample_uniform(rng, ranges, "target_lens_x_delta_mm"),
            lens_y_delta_mm=sample_uniform(rng, ranges, "target_lens_y_delta_mm"),
            camera_x_delta_mm=0.0,
            camera_y_delta_mm=0.0,
        )
        if abs(action.lens_x_delta_mm) + abs(action.lens_y_delta_mm) >= 0.018:
            return action
    return action


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_trajectory(
    sample_id: str,
    current_setup: OpticalSetup,
    target_setup: OpticalSetup,
    target_state: Mapping[str, Any],
    bounds: ActionBounds,
    max_steps: int,
    success_threshold_px: float,
    image_root: Path,
    render_options: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], float, bool]:
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    setup = current_setup
    public_steps: list[dict[str, Any]] = []
    private_steps: list[dict[str, Any]] = []
    image_paths: dict[str, str] = {}
    final_residual = float("inf")
    success = False

    for step_index in range(max_steps):
        current = simulate_and_measure(setup)
        before_residual = residual_error_px(current["state"], target_state)
        current_rel = f"trajectory/{sample_id}_step_{step_index:02d}.png"
        image_paths[f"step_{step_index:02d}_image_path"] = current_rel
        save_intensity_png(current["intensity"], image_root / current_rel, render_options)

        if step_index == 0:
            image_paths["initial_image_path"] = current_rel

        if before_residual <= success_threshold_px:
            final_residual = before_residual
            success = True
            private_steps.append(
                {
                    "step_index": step_index,
                    "state_before": rounded_state(current["state"]),
                    "residual_before_px": round(before_residual, 8),
                    "residual_after_px": round(before_residual, 8),
                    "action": action_to_dict(Action(0.0, 0.0, 0.0, 0.0)),
                    "stopped_before_action": True,
                }
            )
            break

        search_result = choose_control_action(
            setup,
            target_state,
            bounds,
            grid_size=5,
            current_state=current["state"],
        )
        action = search_result["action"]
        next_setup = apply_action_to_setup(setup, action)
        next_measurement = simulate_and_measure(next_setup)
        after_residual = residual_error_px(next_measurement["state"], target_state)
        final_residual = after_residual
        success = after_residual <= success_threshold_px

        public_steps.append(
            {
                "step_index": step_index,
                "action": action_to_dict(action),
                "expected_residual_before_px": round(before_residual, 8),
                "expected_residual_after_px": round(after_residual, 8),
                "expected_improvement_px": round(before_residual - after_residual, 8),
                "search_method": search_result.get("method"),
            }
        )
        private_steps.append(
            {
                "step_index": step_index,
                "state_before": rounded_state(current["state"]),
                "state_after": rounded_state(next_measurement["state"]),
                "residual_before_px": round(before_residual, 8),
                "residual_after_px": round(after_residual, 8),
                "residual_vector_before_px": error_vector_px(current["state"], target_state),
                "residual_vector_after_px": error_vector_px(next_measurement["state"], target_state),
                "action": action_to_dict(action),
                "search_method": search_result.get("method"),
                "jacobian_condition": search_result.get("jacobian_condition"),
                "stopped_before_action": False,
            }
        )

        setup = next_setup
        if success:
            break

    final_rel = f"trajectory/{sample_id}_final.png"
    final_measurement = simulate_and_measure(setup)
    save_intensity_png(final_measurement["intensity"], image_root / final_rel, render_options)
    image_paths["final_image_path"] = final_rel
    return public_steps, private_steps, image_paths, final_residual, success


def generate_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.num_trajectories <= 0:
        raise ValueError("--num-trajectories must be positive")
    if args.success_threshold_px < 0.0:
        raise ValueError("--success-threshold-px must be non-negative")

    base_cfg = load_yaml(str(args.config))
    ranges = parameter_ranges_from_config(base_cfg)
    rng = random.Random(args.seed)
    image_root = args.output_dir / "images"
    bounds = ActionBounds(
        lens_x_mm=(-DEFAULT_LENS_ACTION_BOUND_MM, DEFAULT_LENS_ACTION_BOUND_MM),
        lens_y_mm=(-DEFAULT_LENS_ACTION_BOUND_MM, DEFAULT_LENS_ACTION_BOUND_MM),
    )
    rows: list[dict[str, Any]] = []
    final_residuals: list[float] = []
    success_count = 0

    for index in range(args.num_trajectories):
        sample_id = f"phys_traj_{index:06d}"
        current_setup, sampled_setup = varied_setup_from_base(base_cfg, ranges, rng)
        target_generating_action = sample_target_action(rng, ranges)
        target_setup = apply_action_to_setup(current_setup, target_generating_action)
        target = simulate_and_measure(target_setup)

        target_rel = f"trajectory/{sample_id}_target.png"
        render_options = random_render_params(rng, difficulty=args.render_difficulty)
        save_intensity_png(target["intensity"], image_root / target_rel, render_options)

        recommended_steps, private_steps, image_paths, final_residual, success = build_trajectory(
            sample_id=sample_id,
            current_setup=current_setup,
            target_setup=target_setup,
            target_state=target["state"],
            bounds=bounds,
            max_steps=args.max_steps,
            success_threshold_px=args.success_threshold_px,
            image_root=image_root,
            render_options=render_options,
        )
        image_paths["target_image_path"] = target_rel
        final_residuals.append(final_residual)
        success_count += int(success)

        prompt_inputs = {
            "images": {
                "target_image_path": image_paths["target_image_path"],
                "initial_image_path": image_paths["initial_image_path"],
            },
            "safe_setup_metadata": setup_to_safe_metadata(current_setup),
        }
        assert_no_prompt_leakage(prompt_inputs)

        row = {
            "sample_id": sample_id,
            "sample_type": "trajectory",
            "prompt_inputs": prompt_inputs,
            "target": {
                "task": "closed_loop_alignment_plan",
                "recommended_steps": recommended_steps,
                "success_threshold_px": float(args.success_threshold_px),
                "expected_final_residual_px": round(final_residual, 8),
                "expected_success": bool(success),
            },
            "private_eval": {
                "target_state": rounded_state(target["state"]),
                "all_true_states": private_steps,
                "all_actions": [step["action"] for step in private_steps],
                "residuals": {
                    "per_step": [
                        {
                            "step_index": step["step_index"],
                            "before_px": step["residual_before_px"],
                            "after_px": step["residual_after_px"],
                        }
                        for step in private_steps
                    ],
                    "final_px": round(final_residual, 8),
                    "success": bool(success),
                    "success_threshold_px": float(args.success_threshold_px),
                },
                "step_image_paths": image_paths,
                "simulator_config": {
                    "config_path": str(args.config),
                    "sampled_setup_parameters": sampled_setup,
                    "target_generating_action": action_to_dict(target_generating_action),
                    "initial_setup": setup_snapshot(current_setup),
                    "target_setup": setup_snapshot(target_setup),
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
                "trajectory",
                "closed_loop_alignment",
                DATASET_NAME,
            ],
        }
        rows.append(row)

        if (index + 1) % 25 == 0 or index + 1 == args.num_trajectories:
            print(f"[{index + 1}/{args.num_trajectories}] generated {sample_id}")

    mean_final_residual = sum(final_residuals) / len(final_residuals)
    manifest = {
        "dataset": DATASET_NAME,
        "sample_type": "trajectory",
        "num_trajectories": args.num_trajectories,
        "max_steps": args.max_steps,
        "success_threshold_px": args.success_threshold_px,
        "success_count": success_count,
        "mean_final_residual_px": round(mean_final_residual, 8),
        "seed": args.seed,
        "config_path": str(args.config),
        "image_root": "images",
        "jsonl": "trajectories.jsonl",
        "render_difficulty": args.render_difficulty,
        "parameter_ranges": ranges,
        "action_bounds": {
            "lens_x_mm": list(bounds.lens_x_mm),
            "lens_y_mm": list(bounds.lens_y_mm),
            "camera_x_mm": list(bounds.camera_x_mm),
            "camera_y_mm": list(bounds.camera_y_mm),
        },
    }
    return rows, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = generate_rows(args)
    write_jsonl(args.output_dir / "trajectories.jsonl", rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote dataset to {args.output_dir}")
    print(f"mean_final_residual_px={manifest['mean_final_residual_px']}")
    print(f"success_count={manifest['success_count']}/{manifest['num_trajectories']}")


if __name__ == "__main__":
    main()
