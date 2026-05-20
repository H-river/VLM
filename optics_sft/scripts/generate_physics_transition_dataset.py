#!/usr/bin/env python3
"""Generate forward-transition physics SFT samples from optical_sim."""

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
from optics_sft.physics.metadata_policy import assert_no_prompt_leakage
from optics_sft.physics.rendering import random_render_params, save_intensity_png
from optics_sft.physics.sim_adapter import (
    Action,
    apply_action_to_setup,
    setup_to_safe_metadata,
    simulate_and_measure,
)


M_TO_MM = 1e3
MM_TO_M = 1e-3
M_TO_NM = 1e9
NM_TO_M = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate forward-transition physics SFT rows.")
    parser.add_argument("--output-dir", type=Path, default=Path("../VLM_data/physics_sft_v1"))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--render-difficulty",
        choices=("clean", "medium", "hard"),
        default="medium",
    )
    parser.add_argument("--max-lens-delta-mm", type=float, default=0.05)
    parser.add_argument("--max-camera-delta-mm", type=float, default=0.02)
    parser.add_argument("--config", type=Path, default=Path("optical_sim/configs/base_config.yaml"))
    return parser.parse_args()


def positive_range(center: float, low_factor: float, high_factor: float) -> tuple[float, float]:
    return max(center * low_factor, 1e-12), max(center * high_factor, 1e-12)


def parameter_ranges_from_config(
    cfg: dict[str, Any],
    max_lens_delta_mm: float,
    max_camera_delta_mm: float,
) -> dict[str, Any]:
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
        "lens_x_offset_mm": (-0.25, 0.25),
        "lens_y_offset_mm": (-0.25, 0.25),
        "camera_x_offset_mm": (-0.15, 0.15),
        "camera_y_offset_mm": (-0.15, 0.15),
        "lens_action_delta_mm": (-max_lens_delta_mm, max_lens_delta_mm),
        "camera_action_delta_mm": (-max_camera_delta_mm, max_camera_delta_mm),
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


def sample_action(rng: random.Random, max_lens_delta_mm: float, max_camera_delta_mm: float) -> Action:
    return Action(
        lens_x_delta_mm=rng.uniform(-max_lens_delta_mm, max_lens_delta_mm),
        lens_y_delta_mm=rng.uniform(-max_lens_delta_mm, max_lens_delta_mm),
        camera_x_delta_mm=rng.uniform(-max_camera_delta_mm, max_camera_delta_mm),
        camera_y_delta_mm=rng.uniform(-max_camera_delta_mm, max_camera_delta_mm),
    )


def action_to_dict(action: Action) -> dict[str, float]:
    return {
        "lens_x_delta_mm": float(action.lens_x_delta_mm),
        "lens_y_delta_mm": float(action.lens_y_delta_mm),
        "camera_x_delta_mm": float(action.camera_x_delta_mm),
        "camera_y_delta_mm": float(action.camera_y_delta_mm),
    }


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


def rounded_state(state: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(value), 8) for key, value in state.items()}


def predicted_change(before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "centroid_shift_px": {
            "x": round(float(after_state["centroid_x_px"]) - float(before_state["centroid_x_px"]), 8),
            "y": round(float(after_state["centroid_y_px"]) - float(before_state["centroid_y_px"]), 8),
        },
        "sigma_change_px": {
            "x": round(float(after_state["sigma_x_px"]) - float(before_state["sigma_x_px"]), 8),
            "y": round(float(after_state["sigma_y_px"]) - float(before_state["sigma_y_px"]), 8),
        },
        "peak_intensity_change": round(
            float(after_state["peak_intensity"]) - float(before_state["peak_intensity"]),
            8,
        ),
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
    assignments = {}
    for index in indices:
        if index in val_indices:
            assignments[index] = "val"
        elif index in test_indices:
            assignments[index] = "test"
        else:
            assignments[index] = "train"
    return assignments


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def generate_rows(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base_cfg = load_yaml(str(args.config))
    ranges = parameter_ranges_from_config(base_cfg, args.max_lens_delta_mm, args.max_camera_delta_mm)
    assignments = split_assignments(args.num_samples, args.val_ratio, args.test_ratio, args.seed)
    rng = random.Random(args.seed)
    image_root = args.output_dir / "images"
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for index in range(args.num_samples):
        sample_id = f"phys_fwd_{index:06d}"
        split = assignments[index]
        setup, sampled_setup = varied_setup_from_base(base_cfg, ranges, rng)
        before = simulate_and_measure(setup)
        action = sample_action(rng, args.max_lens_delta_mm, args.max_camera_delta_mm)
        after_setup = apply_action_to_setup(setup, action)
        after = simulate_and_measure(after_setup)

        before_rel = f"{split}/{sample_id}_before.png"
        after_rel = f"{split}/{sample_id}_after.png"
        render_options = random_render_params(rng, difficulty=args.render_difficulty)
        save_intensity_png(before["intensity"], image_root / before_rel, render_options)
        save_intensity_png(after["intensity"], image_root / after_rel, render_options)

        prompt_inputs = {
            "images": {
                "before_image_path": before_rel,
            },
            "safe_setup_metadata": setup_to_safe_metadata(setup),
            "action": action_to_dict(action),
        }
        assert_no_prompt_leakage(prompt_inputs)

        row = {
            "sample_id": sample_id,
            "sample_type": "forward_transition",
            "prompt_inputs": prompt_inputs,
            "target": {
                "predicted_after_state": rounded_state(after["state"]),
                "predicted_change": predicted_change(before["state"], after["state"]),
            },
            "private_eval": {
                "before_state": rounded_state(before["state"]),
                "after_state": rounded_state(after["state"]),
                "after_image_path": after_rel,
                "simulator_config": {
                    "config_path": str(args.config),
                    "sampled_setup_parameters": sampled_setup,
                    "before_setup": setup_snapshot(setup),
                    "after_setup": setup_snapshot(after_setup),
                    "render_options": render_options,
                },
            },
            "split_tags": [
                split,
                "forward_transition",
                "physics_sft_v1",
            ],
        }
        rows_by_split[split].append(row)

        if (index + 1) % 50 == 0 or index + 1 == args.num_samples:
            print(f"[{index + 1}/{args.num_samples}] generated {sample_id}")

    manifest = {
        "dataset": "physics_sft_v1",
        "sample_type": "forward_transition",
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
        "render_difficulty": args.render_difficulty,
        "parameter_ranges": ranges,
    }
    return rows_by_split, manifest


def main() -> None:
    args = parse_args()
    if args.max_lens_delta_mm < 0.0 or args.max_camera_delta_mm < 0.0:
        raise ValueError("Action delta limits must be non-negative")

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
