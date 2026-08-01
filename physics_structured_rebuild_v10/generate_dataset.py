#!/usr/bin/env python3
"""Generate isolated, grouped, system-aligned v10 response surfaces."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import current_process
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.metrics import compute_metrics
from optical_sim.src.optical_elements import OpticalSetup, setup_from_dict
from optical_sim.src.simulator import (
    _BACKENDS,
    _extract_sensor_region,
    apply_thin_lens,
    gaussian_source_field,
)
from optics_sft.physics.sim_adapter import metrics_to_state
from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    config_from_visible,
)
from physics_structured_rebuild_v10.contracts import (
    REGIMES,
    VISUAL_CONDITIONS,
    ZERO_ACTION_INDEX,
    action_cardinalities,
    context_hash,
    interaction_category,
    natural_requested_action_index,
    setup_hash,
    sha256_file,
    stable_seed,
    validate_action_order,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    SETUP_FIELDS,
    STATE_FIELDS,
    change_and_directions,
    fixed_action_grid,
)

DEFAULT_CONFIG = Path(__file__).with_name("config_v10.json")
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"
BASE_CONFIG = REPO_ROOT / "optical_sim/configs/base_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scale",
        choices=("full", "pilot", "smoke"),
        default="full",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--finalize-pilot-from-full-attempt",
        action="store_true",
        help=(
            "Reuse the deterministic prefix of an interrupted full run, "
            "preserving extra shards but excluding them from the pilot manifest."
        ),
    )
    return parser.parse_args()


def regime_schedule(
    count: int,
    fractions: Mapping[str, Any],
    seed: int,
) -> list[str]:
    if set(fractions) != set(REGIMES):
        raise ValueError("regime fractions differ from the frozen registry")
    raw = {name: float(fractions[name]) * count for name in REGIMES}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = count - sum(counts.values())
    ordering = sorted(REGIMES, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in ordering[:remainder]:
        counts[name] += 1
    schedule = [name for name in REGIMES for _ in range(counts[name])]
    random.Random(seed).shuffle(schedule)
    return schedule


def sample_setup(regime: str, group_id: str, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(stable_seed(seed, group_id, regime, "setup"))
    result = {
        "wavelength_nm": float(rng.uniform(617.0, 648.0)),
        "beam_waist_mm": float(rng.uniform(0.70, 1.35)),
        "power_w": float(rng.uniform(0.75, 1.25)),
        "lens_focal_length_mm": float(rng.uniform(82.0, 120.0)),
        "lens_aperture_mm": float(rng.uniform(19.0, 32.0)),
        "source_to_lens_mm": float(rng.uniform(140.0, 260.0)),
        "lens_to_camera_mm": float(rng.uniform(92.0, 190.0)),
        "lens_x_offset_mm": float(rng.uniform(-0.23, 0.23)),
        "lens_y_offset_mm": float(rng.uniform(-0.23, 0.23)),
        "camera_x_offset_mm": float(rng.uniform(-0.18, 0.18)),
        "camera_y_offset_mm": float(rng.uniform(-0.18, 0.18)),
        "pixel_size_um": float(rng.choice((5.0, 5.5, 6.0))),
    }
    if regime == "ordinary":
        result["pixel_size_um"] = 5.5
    elif regime in {"focusing", "tolerance_boundary"}:
        focal = float(rng.uniform(72.0, 128.0))
        source = float(rng.uniform(max(1.18 * focal, 120.0), 282.0))
        image_distance = focal * source / max(source - focal, 1e-6)
        width = 0.010 if regime == "tolerance_boundary" else 0.020
        result.update(
            {
                "lens_focal_length_mm": focal,
                "source_to_lens_mm": source,
                "lens_to_camera_mm": float(
                    np.clip(image_distance * rng.uniform(1.0 - width, 1.0 + width), 72.0, 255.0)
                ),
                "lens_aperture_mm": float(rng.uniform(17.0, 30.0)),
            }
        )
        if regime == "tolerance_boundary" and rng.random() < 0.5:
            pitch_mm = result["pixel_size_um"] / 1000.0
            half = 512.0 * pitch_mm
            axis = int(rng.integers(0, 2))
            offset = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.72, 0.92) * half)
            result["camera_x_offset_mm" if axis == 0 else "camera_y_offset_mm"] = offset
    elif regime == "clipping":
        result.update(
            {
                "lens_aperture_mm": float(rng.uniform(1.2, 4.0)),
                "lens_x_offset_mm": float(rng.choice((-1.0, 1.0)) * rng.uniform(0.20, 1.00)),
                "lens_y_offset_mm": float(rng.choice((-1.0, 1.0)) * rng.uniform(0.20, 1.00)),
            }
        )
    elif regime == "camera_boundary":
        pitch_mm = result["pixel_size_um"] / 1000.0
        half = 512.0 * pitch_mm
        axis = int(rng.integers(0, 2))
        offset = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.68, 1.03) * half)
        result["camera_x_offset_mm" if axis == 0 else "camera_y_offset_mm"] = offset
    elif regime == "high_offset_interaction":
        result.update(
            {
                "lens_aperture_mm": float(rng.uniform(14.0, 21.0)),
                "lens_x_offset_mm": float(rng.choice((-1.0, 1.0)) * rng.uniform(0.24, 0.38)),
                "lens_y_offset_mm": float(rng.choice((-1.0, 1.0)) * rng.uniform(0.24, 0.38)),
                "camera_x_offset_mm": float(rng.choice((-1.0, 1.0)) * rng.uniform(0.17, 0.28)),
                "camera_y_offset_mm": float(rng.choice((-1.0, 1.0)) * rng.uniform(0.17, 0.28)),
            }
        )
    else:
        raise ValueError(regime)
    return result


def optical_setup(values: Mapping[str, Any]) -> OpticalSetup:
    base = load_yaml(BASE_CONFIG)
    visible = {**dict(values), "sensor_resolution_px": [1024, 1024]}
    return setup_from_dict(config_from_visible(visible, base))


def selected_visual_indices(group_id: str, seed: int, count: int) -> list[int]:
    cardinality = action_cardinalities()
    desired = [1, 3, 4, 2]
    output = []
    for target_number in range(count):
        eligible = np.flatnonzero(cardinality == desired[target_number % len(desired)])
        output.append(
            int(
                eligible[
                    stable_seed(seed, group_id, "visual", target_number)
                    % len(eligible)
                ]
            )
        )
    return output


def simulate_surface(
    setup: OpticalSetup,
    image_indices: set[int],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], dict[str, float]]:
    propagate = _BACKENDS.get(
        setup.propagation_backend,
        _BACKENDS["fresnel_numpy"],
    )
    source, grid_x, grid_y, spacing = gaussian_source_field(setup)
    at_lens = propagate(
        source,
        spacing,
        setup.laser_to_lens,
        setup.source.wavelength,
    )
    incoming_power = max(float(np.square(np.abs(at_lens)).sum()), 1e-30)
    actions = fixed_action_grid()
    cached: dict[tuple[float, float], tuple[np.ndarray, float]] = {}
    candidates: list[dict[str, Any]] = []
    images: dict[int, np.ndarray] = {}
    for index, action in enumerate(actions):
        lens_key = (
            float(action["lens_x_delta_mm"]),
            float(action["lens_y_delta_mm"]),
        )
        cached_item = cached.get(lens_key)
        if cached_item is None:
            lens_setup = copy.deepcopy(setup)
            lens_setup.lens.x_offset += lens_key[0] * 1e-3
            lens_setup.lens.y_offset += lens_key[1] * 1e-3
            after_lens = apply_thin_lens(at_lens, grid_x, grid_y, lens_setup)
            transmission = float(
                np.square(np.abs(after_lens)).sum() / incoming_power
            )
            field = propagate(
                after_lens,
                spacing,
                lens_setup.effective_camera_distance,
                lens_setup.source.wavelength,
            )
            cached_item = (field, transmission)
            cached[lens_key] = cached_item
        field, transmission = cached_item
        candidate_setup = copy.deepcopy(setup)
        candidate_setup.lens.x_offset += lens_key[0] * 1e-3
        candidate_setup.lens.y_offset += lens_key[1] * 1e-3
        candidate_setup.camera.x_offset += float(action["camera_x_delta_mm"]) * 1e-3
        candidate_setup.camera.y_offset += float(action["camera_y_delta_mm"]) * 1e-3
        intensity, sensor_x, sensor_y = _extract_sensor_region(
            field,
            grid_x,
            grid_y,
            candidate_setup,
        )
        metrics = compute_metrics(intensity, sensor_x, sensor_y)
        state = metrics_to_state(metrics, candidate_setup)
        state_out = {
            field_name: round(float(state[field_name]), 9)
            for field_name in STATE_FIELDS
        }
        captured_power = float(
            np.asarray(intensity, dtype=np.float64).sum()
            * candidate_setup.sensor.pixel_pitch**2
        )
        edge_distance = float(
            min(
                state_out["centroid_x_px"],
                1023.0 - state_out["centroid_x_px"],
                state_out["centroid_y_px"],
                1023.0 - state_out["centroid_y_px"],
            )
        )
        candidates.append(
            {
                "action": {
                    field_name: float(action[field_name])
                    for field_name in ACTION_FIELDS
                },
                "next_state": state_out,
                "auxiliary": {
                    "captured_optical_power": captured_power,
                    "lens_transmission": transmission,
                    "clipping_fraction": 1.0 - transmission,
                    "distance_to_camera_boundary_px": edge_distance,
                    "action_cardinality": int(
                        sum(abs(float(action[field_name])) > 0 for field_name in ACTION_FIELDS)
                    ),
                    "interaction_category": interaction_category(action),
                },
            }
        )
        if index in image_indices:
            images[index] = np.asarray(intensity, dtype=np.float32)
    source_distance = float(setup.laser_to_lens)
    focal = float(setup.lens.focal_length)
    camera = float(setup.effective_camera_distance)
    focus_residual = abs(
        1.0 / focal - 1.0 / source_distance - 1.0 / camera
    ) / max(abs(1.0 / focal), 1e-12)
    return candidates, images, {"geometric_focus_residual": float(focus_residual)}


def render_pair(
    current: np.ndarray,
    target: np.ndarray,
    size: int,
    condition: str,
    seed: int,
    current_path: Path,
    target_path: Path,
) -> dict[str, Any]:
    high = max(float(current.max()), float(target.max()), 1e-12)
    rng = np.random.default_rng(seed)

    def convert(values: np.ndarray, role: str) -> Image.Image:
        display = np.sqrt(np.clip(values.astype(np.float64) / high, 0.0, 1.0))
        image = Image.fromarray(np.rint(display * 255).astype(np.uint8), mode="L")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float64)
        if condition == "noise":
            array += rng.normal(0.0, 5.0, array.shape)
        elif condition == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=1.2)).convert("RGB")
        elif condition == "saturation":
            array = np.where(array > 170.0, 255.0, array)
        elif condition == "asymmetric_gain":
            gradient = np.linspace(0.72, 1.18, array.shape[1])[None, :]
            array *= gradient if role == "current" else gradient[:, ::-1]
        elif condition == "crop_boundary":
            shifted = np.zeros_like(array)
            shift = max(2, size // 12)
            shifted[:, :-shift] = array[:, shift:]
            array = shifted
        elif condition != "clean":
            raise ValueError(condition)
        return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).convert("RGB")

    current_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    convert(current, "current").save(current_path, format="PNG", optimize=True)
    convert(target, "target").save(target_path, format="PNG", optimize=True)
    return {
        "linear_intensity_low": 0.0,
        "linear_intensity_high": high,
        "gamma": 0.5,
        "source_sensor_resolution_px": [1024, 1024],
        "render_resolution_px": [size, size],
    }


def _configure_worker(affinities: tuple[int, ...]) -> None:
    identity = current_process()._identity
    worker_number = int(identity[-1]) if identity else 1
    cpu = affinities[(worker_number - 1) % len(affinities)]
    os.sched_setaffinity(0, {int(cpu)})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def build_group(job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(job["output_dir"])
    split = str(job["split"])
    group_id = str(job["group_id"])
    shard = output_dir / "shards" / split / f"{group_id}.json"
    if shard.is_file():
        existing = json.loads(shard.read_text(encoding="utf-8"))
        if len(existing.get("candidates", [])) == 81:
            validate_action_order(existing["candidates"])
            return {
                "group_id": group_id,
                "split": split,
                "regime": existing["regime"],
                "setup_hash": existing["setup_hash"],
                "context_hash": existing["context_hash"],
                "resumed": True,
                "seconds": 0.0,
            }
    seed = int(job["seed"])
    regime = str(job["regime"])
    setup_values = sample_setup(regime, group_id, seed)
    setup = optical_setup(setup_values)
    visual_indices = selected_visual_indices(
        group_id,
        seed,
        int(job["visual_targets_per_group"]),
    )
    image_indices = {ZERO_ACTION_INDEX, *visual_indices}
    candidates, raw_images, group_aux = simulate_surface(setup, image_indices)
    current = dict(candidates[ZERO_ACTION_INDEX]["next_state"])
    for candidate in candidates:
        change, directions = change_and_directions(
            current,
            candidate["next_state"],
        )
        candidate["change"] = {
            field: round(float(change[field]), 9) for field in STATE_FIELDS
        }
        candidate["directions"] = directions
    requested_index = natural_requested_action_index(group_id, seed)
    visual_requests = []
    for request_number, target_index in enumerate(visual_indices):
        condition = VISUAL_CONDITIONS[
            stable_seed(seed, group_id, request_number, "condition")
            % len(VISUAL_CONDITIONS)
        ]
        relative_root = Path("images") / split / group_id
        current_relative = relative_root / f"request_{request_number:02d}_current_{condition}.png"
        target_relative = relative_root / f"request_{request_number:02d}_target_{condition}.png"
        calibration = render_pair(
            raw_images[ZERO_ACTION_INDEX],
            raw_images[target_index],
            int(job["visual_image_size"]),
            condition,
            stable_seed(seed, group_id, request_number, "render"),
            output_dir / current_relative,
            output_dir / target_relative,
        )
        visual_requests.append(
            {
                "request_id": f"{group_id}_visual_{request_number:02d}",
                "source_action_index": int(target_index),
                "desired_beam_state": candidates[target_index]["next_state"],
                "current_image": current_relative.as_posix(),
                "target_image": target_relative.as_posix(),
                "condition": condition,
                "image_calibration": calibration,
            }
        )
    row = {
        "dataset_version": str(job["version"]),
        "split": split,
        "group_id": group_id,
        "regime": regime,
        "setup": {
            field: round(float(setup_values[field]), 12) for field in SETUP_FIELDS
        },
        "current_beam_state": current,
        "setup_hash": setup_hash(setup_values),
        "context_hash": context_hash(setup_values, current),
        "natural_requested_action_index": int(requested_index),
        "group_auxiliary": group_aux,
        "candidates": candidates,
        "visual_requests": visual_requests,
    }
    validate_action_order(candidates)
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(shard)
    return {
        "group_id": group_id,
        "split": split,
        "regime": regime,
        "setup_hash": row["setup_hash"],
        "context_hash": row["context_hash"],
        "resumed": False,
        "seconds": time.perf_counter() - started,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.scale == "smoke":
        counts = {"train": 4, "development": 2, "locked_test": 2}
    else:
        counts = dict(config[f"{args.scale}_group_counts"])
    workers = int(args.workers or config["workers"])
    if workers not in (1, 2):
        raise ValueError("v10 generation requires one or two workers")
    snapshot = {
        **config,
        "active_scale": args.scale,
        "active_group_counts": counts,
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
    }
    snapshot_path = output_dir / "config_snapshot.json"
    if args.finalize_pilot_from_full_attempt:
        if args.scale != "pilot" or not snapshot_path.exists():
            raise RuntimeError(
                "pilot finalization requires --scale pilot and an existing full snapshot"
            )
        existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
        generation_keys = (
            "version",
            "full_group_counts",
            "pilot_group_counts",
            "split_seeds",
            "regime_fractions",
            "visual_targets_per_group",
            "visual_image_size",
        )
        if (
            existing.get("active_scale") != "full"
            or any(existing.get(key) != snapshot.get(key) for key in generation_keys)
        ):
            raise RuntimeError("existing full-attempt snapshot is incompatible")
        (output_dir / "pilot_finalization_config_snapshot.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif snapshot_path.exists():
        existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if existing != snapshot:
            raise RuntimeError("output directory contains a different v10 configuration")
    else:
        snapshot_path.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    started = time.perf_counter()
    split_summary: dict[str, Any] = {}
    all_results: list[dict[str, Any]] = []
    for split in ("train", "development", "locked_test"):
        count = int(counts[split])
        seed = int(config["split_seeds"][split])
        schedule = regime_schedule(
            count,
            config["regime_fractions"],
            stable_seed(seed, split, "schedule"),
        )
        jobs = [
            {
                "version": config["version"],
                "output_dir": str(output_dir),
                "split": split,
                "group_id": f"v10_{split}_{index:06d}",
                "regime": schedule[index],
                "seed": seed,
                "visual_targets_per_group": int(config["visual_targets_per_group"]),
                "visual_image_size": int(config["visual_image_size"]),
            }
            for index in range(count)
        ]
        results = []
        initializer = _configure_worker if workers > 1 else None
        initargs = (
            (tuple(int(value) for value in config["worker_cpu_affinities"]),)
            if workers > 1
            else ()
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initializer,
            initargs=initargs,
        ) as executor:
            for result in executor.map(build_group, jobs, chunksize=1):
                results.append(result)
                if len(results) % 10 == 0 or len(results) == len(jobs):
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "completed": len(results),
                                "total": len(jobs),
                                "resumed": sum(item["resumed"] for item in results),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        rows = [
            json.loads(
                (output_dir / "shards" / split / f"{job['group_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            for job in jobs
        ]
        write_jsonl(output_dir / "grids" / f"{split}.jsonl", rows)
        split_summary[split] = {
            "groups": len(rows),
            "transitions": 81 * len(rows),
            "visual_requests": sum(len(row["visual_requests"]) for row in rows),
            "regime_counts": dict(sorted(Counter(row["regime"] for row in rows).items())),
            "setup_hashes": sorted(row["setup_hash"] for row in rows),
            "context_hashes": sorted(row["context_hash"] for row in rows),
            "jsonl_sha256": sha256_file(output_dir / "grids" / f"{split}.jsonl"),
            "resumed_groups": sum(item["resumed"] for item in results),
            "worker_seconds": sum(float(item["seconds"]) for item in results),
        }
        all_results.extend(results)
    setup_owners: dict[str, set[str]] = {}
    context_owners: dict[str, set[str]] = {}
    for result in all_results:
        setup_owners.setdefault(result["setup_hash"], set()).add(result["split"])
        context_owners.setdefault(result["context_hash"], set()).add(result["split"])
    setup_overlap = {key: sorted(value) for key, value in setup_owners.items() if len(value) > 1}
    context_overlap = {key: sorted(value) for key, value in context_owners.items() if len(value) > 1}
    if setup_overlap or context_overlap:
        raise RuntimeError("cross-split setup/context hash overlap detected")
    manifest = {
        "version": config["version"],
        "active_scale": args.scale,
        "requested_full_training_groups": int(config["full_group_counts"]["train"]),
        "split_seeds": config["split_seeds"],
        "action_count_per_group": 81,
        "independent_unit": "setup/current-state group",
        "split_summary": split_summary,
        "cross_split_setup_hash_overlap": 0,
        "cross_split_context_hash_overlap": 0,
        "old_system_evaluation_files_opened": [],
        "previous_sealed_test_files_opened": [],
        "locked_test_generated_but_not_used_for_selection": True,
        "full_intensity_policy": (
            "raw simulator intensities retained in memory only for current and "
            "three deterministic visual targets; calibrated PNGs are stored"
        ),
        "pilot_limitation": (
            None
            if args.scale == "full"
            else "active generation is a statistically useful pilot, not the preregistered full dataset"
        ),
        "interrupted_full_attempt": (
            {
                split: len(list((output_dir / "shards" / split).glob("*.json")))
                for split in ("train", "development", "locked_test")
            }
            if args.finalize_pilot_from_full_attempt
            else None
        ),
        "extra_full_attempt_shards_excluded_from_pilot": (
            {
                split: max(
                    0,
                    len(list((output_dir / "shards" / split).glob("*.json")))
                    - int(counts[split]),
                )
                for split in ("train", "development", "locked_test")
            }
            if args.finalize_pilot_from_full_attempt
            else None
        ),
        "wall_seconds": time.perf_counter() - started,
        "complete": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
