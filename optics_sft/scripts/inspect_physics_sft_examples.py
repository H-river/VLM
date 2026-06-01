#!/usr/bin/env python3
"""Inspect physics-aware SFT rows without loading a model or training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.physics.prompt_builder import build_physics_prompt, expected_image_slots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print physics SFT prompt/target examples.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--jsonl", type=Path, help="Alias for --input-jsonl.")
    source.add_argument("--use-toy-examples", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None, help="Alias for --max-rows.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Accepted for CLI consistency; image files are not loaded by this inspector.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def safe_setup_metadata() -> dict[str, Any]:
    return {
        "wavelength_nm": 632.8,
        "beam_waist_mm": 1.0,
        "power_w": 1.0,
        "lens_focal_length_mm": 100.0,
        "lens_aperture_mm": 25.4,
        "source_to_lens_mm": 200.0,
        "lens_to_camera_mm": 150.0,
        "sensor_resolution": [1024, 1024],
        "pixel_size_um": 5.5,
        "coordinate_convention": "sensor pixel x increases right; y increases down",
        "propagation_backend": "fresnel_numpy",
        "grid_size": 1024,
        "grid_extent_mm": 30.0,
    }


def action(dx: float, dy: float) -> dict[str, float]:
    return {
        "lens_x_delta_mm": dx,
        "lens_y_delta_mm": dy,
        "camera_x_delta_mm": 0.0,
        "camera_y_delta_mm": 0.0,
    }


def toy_rows() -> list[dict[str, Any]]:
    metadata = safe_setup_metadata()
    before_state = {
        "centroid_x_px": 514.0,
        "centroid_y_px": 507.0,
        "sigma_x_px": 96.0,
        "sigma_y_px": 98.0,
        "peak_intensity": 0.82,
    }
    after_state = {
        "centroid_x_px": 511.8,
        "centroid_y_px": 510.6,
        "sigma_x_px": 95.4,
        "sigma_y_px": 98.2,
        "peak_intensity": 0.86,
    }
    return [
        {
            "sample_id": "toy_inverse_000",
            "sample_type": "inverse_control",
            "prompt_inputs": {
                "images": {
                    "current_image_path": "toy/inverse_current.png",
                    "target_image_path": "toy/inverse_target.png",
                },
                "safe_setup_metadata": metadata,
            },
            "target": {
                "task": "physics_aware_beam_alignment",
                "perception": {
                    "current_relative_to_target_x": "right",
                    "current_relative_to_target_y": "above",
                },
                "physics_reasoning_summary": {
                    "relevant_parameters_used": ["lens_focal_length_mm", "lens_to_camera_mm"],
                    "control_coupling_summary": "Lens offsets move the centroid on the sensor.",
                },
                "control_plan": action(-0.012, 0.018),
                "prediction": {"expected_residual_error_px": 1.2},
                "confidence": 0.74,
            },
            "private_eval": {
                "current_state": {"centroid_x_px": 514.0, "centroid_y_px": 507.0},
                "target_state": {"centroid_x_px": 512.0, "centroid_y_px": 511.0},
                "true_control_plan": action(-0.012, 0.018),
                "initial_error_px": 4.47,
                "post_action_error_px": 1.2,
            },
            "split_tags": ["toy", "inverse_control"],
        },
        {
            "sample_id": "toy_forward_000",
            "sample_type": "forward_transition",
            "prompt_inputs": {
                "images": {"before_image_path": "toy/forward_before.png"},
                "safe_setup_metadata": metadata,
                "action": action(-0.01, 0.02),
            },
            "target": {
                "task": "forward_centroid_transition",
                "target_mode": "centroid_only",
                "predicted_after_state": {
                    "centroid_x_px": after_state["centroid_x_px"],
                    "centroid_y_px": after_state["centroid_y_px"],
                },
                "predicted_change": {
                    "delta_centroid_x_px": -2.2,
                    "delta_centroid_y_px": 3.6,
                },
            },
            "private_eval": {
                "before_state": before_state,
                "after_state": after_state,
                "simulator_config": {"source": "toy"},
            },
            "split_tags": ["toy", "forward_transition"],
        },
        {
            "sample_id": "toy_counterfactual_000",
            "sample_type": "counterfactual_pair",
            "prompt_inputs": {
                "images": {
                    "scenario_a_current_image_path": "toy/cf_a_current.png",
                    "scenario_a_target_image_path": "toy/cf_a_target.png",
                    "scenario_b_current_image_path": "toy/cf_b_current.png",
                    "scenario_b_target_image_path": "toy/cf_b_target.png",
                },
                "safe_setup_metadata": {
                    "scenario_a": {**metadata, "lens_to_camera_mm": 150.0},
                    "scenario_b": {**metadata, "lens_to_camera_mm": 210.0},
                },
            },
            "target": {
                "task": "counterfactual_action_comparison",
                "changed_parameter": "lens_to_camera_mm",
                "should_action_change": True,
                "scenario_a_control_plan": action(-0.010, 0.014),
                "scenario_b_control_plan": action(-0.007, 0.010),
                "action_difference_summary": {
                    "lens_x_difference_mm": 0.003,
                    "lens_y_difference_mm": -0.004,
                },
            },
            "private_eval": {
                "scenario_a": {
                    "current_state": {"centroid_x_px": 515.0, "centroid_y_px": 508.0},
                    "target_state": {"centroid_x_px": 512.0, "centroid_y_px": 511.0},
                    "searched_control_action": action(-0.010, 0.014),
                },
                "scenario_b": {
                    "current_state": {"centroid_x_px": 515.2, "centroid_y_px": 508.1},
                    "target_state": {"centroid_x_px": 512.1, "centroid_y_px": 510.9},
                    "searched_control_action": action(-0.007, 0.010),
                },
            },
            "split_tags": ["toy", "counterfactual_pair", "ood:lens_to_camera_mm"],
        },
        {
            "sample_id": "toy_trajectory_000",
            "sample_type": "trajectory",
            "prompt_inputs": {
                "images": {
                    "initial_image_path": "toy/traj_initial.png",
                    "target_image_path": "toy/traj_target.png",
                },
                "safe_setup_metadata": metadata,
            },
            "target": {
                "task": "closed_loop_alignment_plan",
                "recommended_steps": [
                    {
                        "step_index": 0,
                        "action": action(-0.010, 0.018),
                        "expected_residual_after_px": 2.4,
                    },
                    {
                        "step_index": 1,
                        "action": action(-0.003, 0.006),
                        "expected_residual_after_px": 0.8,
                    },
                ],
            },
            "private_eval": {
                "all_actions": [action(-0.010, 0.018), action(-0.003, 0.006)],
                "all_true_states": [
                    {"step_index": 0, "residual_before_px": 5.1, "residual_after_px": 2.4},
                    {"step_index": 1, "residual_before_px": 2.4, "residual_after_px": 0.8},
                ],
                "residuals": {"final_px": 0.8, "success": True},
            },
            "split_tags": ["toy", "trajectory"],
        },
    ]


def print_row(row: dict[str, Any]) -> None:
    prompt = build_physics_prompt(row)
    target_json = json.dumps(row["target"], indent=2, sort_keys=True)
    print("=" * 88)
    print(f"sample_id: {row.get('sample_id')}")
    print(f"sample_type: {row.get('sample_type')}")
    print(f"image_slots: {json.dumps(expected_image_slots(row), sort_keys=True)}")
    print(f"prompt_inputs keys shown to model: {sorted(row.get('prompt_inputs', {}).keys())}")
    target = row.get("target", {})
    if isinstance(target, dict):
        print(f"target_mode: {target.get('target_mode', 'not_set')}")
    print(f"SFT ground truth field: target")
    print(f"private_eval-only fields: {sorted(row.get('private_eval', {}).keys())}")
    print("\nPROMPT TEXT:\n")
    print(prompt)
    print("\nSERIALIZED TARGET JSON:\n")
    print(target_json)


def main() -> None:
    args = parse_args()
    max_rows = args.max_examples if args.max_examples is not None else args.max_rows
    input_jsonl = args.input_jsonl if args.input_jsonl is not None else args.jsonl
    if input_jsonl is not None:
        rows = read_jsonl(input_jsonl, max_rows)
    else:
        rows = toy_rows()
        if max_rows is not None:
            rows = rows[: max_rows]

    for row in rows:
        print_row(row)


if __name__ == "__main__":
    main()
