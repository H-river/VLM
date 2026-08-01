#!/usr/bin/env python3
"""Generate a resumable simulator-grounded dataset for specialist rebuild v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import current_process
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    STATE_FIELDS,
    change_and_directions,
    fixed_action_grid,
    matching_mask,
    minimum_motion_index,
    read_json,
    stable_rng,
    stable_token,
    write_jsonl,
)


DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--worker-affinities",
        default="",
        help=(
            "Optional semicolon-separated CPU sets, one per worker; "
            "for example '0;8;1,9'."
        ),
    )
    parser.add_argument("--max-groups-per-split", type=int)
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def configure_worker_affinity(specifications: tuple[str, ...]) -> None:
    if not specifications:
        return
    identity = current_process()._identity
    worker_number = int(identity[-1]) if identity else 1
    specification = specifications[(worker_number - 1) % len(specifications)]
    cpus = {int(value) for value in specification.split(",") if value}
    if not cpus:
        raise ValueError("worker affinity cannot be empty")
    os.sched_setaffinity(0, cpus)


def rounded_state(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: round(float(result["state"][key]), 9) for key in STATE_FIELDS
    }


def sampled_setup(
    seed: int, split: str, group_id: str
) -> dict[str, float]:
    rng = stable_rng(seed, split, group_id, "setup")
    if split == "test_ood_physics":
        values = {
            "wavelength_nm": rng.uniform(600.0, 670.0),
            "beam_waist_mm": rng.uniform(0.58, 1.50),
            "power_w": rng.uniform(0.60, 1.40),
            "lens_focal_length_mm": rng.uniform(72.0, 132.0),
            "lens_aperture_mm": rng.uniform(17.0, 35.0),
            "source_to_lens_mm": rng.uniform(125.0, 275.0),
            "lens_to_camera_mm": rng.uniform(82.0, 205.0),
            "lens_x_offset_mm": rng.uniform(-0.28, 0.28),
            "lens_y_offset_mm": rng.uniform(-0.28, 0.28),
            "camera_x_offset_mm": rng.uniform(-0.21, 0.21),
            "camera_y_offset_mm": rng.uniform(-0.21, 0.21),
            "pixel_size_um": rng.choice((4.5, 5.5, 6.5)),
        }
        # Ensure every OOD group is outside at least one training interval.
        axis = int(stable_token(seed, group_id, "ood_axis")[:8], 16) % 4
        if axis == 0:
            values["wavelength_nm"] = rng.choice(
                (rng.uniform(600.0, 618.0), rng.uniform(647.0, 670.0))
            )
        elif axis == 1:
            values["beam_waist_mm"] = rng.choice(
                (rng.uniform(0.58, 0.72), rng.uniform(1.33, 1.50))
            )
        elif axis == 2:
            values["lens_focal_length_mm"] = rng.choice(
                (rng.uniform(72.0, 82.0), rng.uniform(118.0, 132.0))
            )
        else:
            values["power_w"] = rng.choice(
                (rng.uniform(0.60, 0.76), rng.uniform(1.24, 1.40))
            )
        return values
    return {
        "wavelength_nm": rng.uniform(620.0, 645.0),
        "beam_waist_mm": rng.uniform(0.75, 1.30),
        "power_w": rng.uniform(0.8, 1.2),
        "lens_focal_length_mm": rng.uniform(85.0, 115.0),
        "lens_aperture_mm": rng.uniform(20.0, 30.0),
        "source_to_lens_mm": rng.uniform(150.0, 250.0),
        "lens_to_camera_mm": rng.uniform(100.0, 180.0),
        "lens_x_offset_mm": rng.uniform(-0.20, 0.20),
        "lens_y_offset_mm": rng.uniform(-0.20, 0.20),
        "camera_x_offset_mm": rng.uniform(-0.15, 0.15),
        "camera_y_offset_mm": rng.uniform(-0.15, 0.15),
        "pixel_size_um": 5.5,
    }


def condition_schedule(config: Mapping[str, Any], split: str) -> list[str]:
    if split == "train":
        counts = config["training_visual_conditions"]
    elif split == "test_visual_stress":
        counts = config["visual_stress_conditions"]
    else:
        return ["clean"] * int(config["group_counts"][split])
    return [
        condition
        for condition, count in counts.items()
        for _ in range(int(count))
    ]


def render_image(
    intensity: np.ndarray,
    high: float,
    condition: str,
    seed_text: str,
    output_path: Path,
) -> None:
    normalized = np.clip(np.asarray(intensity, dtype=np.float64) / high, 0.0, 1.0)
    display = np.sqrt(normalized)
    image = Image.fromarray(
        np.rint(display * 255.0).astype(np.uint8), mode="L"
    ).resize((192, 192), Image.Resampling.LANCZOS)
    rng = np.random.default_rng(int(stable_token(seed_text)[:16], 16))
    if condition == "noise":
        array = np.asarray(image, dtype=np.float64)
        array = np.clip(array + rng.normal(0.0, 5.0, array.shape), 0.0, 255.0)
        image = Image.fromarray(np.rint(array).astype(np.uint8), mode="L")
    elif condition == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=1.4))
    elif condition == "dim_noise":
        array = np.asarray(image, dtype=np.float64) * 0.55
        array = np.clip(array + rng.normal(0.0, 4.0, array.shape), 0.0, 255.0)
        image = Image.fromarray(np.rint(array).astype(np.uint8), mode="L")
    elif condition == "saturation":
        array = np.asarray(image, dtype=np.float64)
        image = Image.fromarray(
            np.rint(np.where(array > 170.0, 255.0, array)).astype(np.uint8),
            mode="L",
        )
    elif condition == "crop_boundary":
        width, height = image.size
        image = image.crop((18, 0, width, height - 18)).resize(
            (width, height), Image.Resampling.BILINEAR
        )
    elif condition == "gamma_shift":
        array = np.asarray(image, dtype=np.float64) / 255.0
        image = Image.fromarray(
            np.rint(np.power(array, 0.78) * 255.0).astype(np.uint8), mode="L"
        )
    elif condition != "clean":
        raise ValueError(f"unknown visual condition: {condition}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG", optimize=True)


def group_paths(
    output_dir: Path, split: str, group_id: str, condition: str
) -> tuple[Path, Path, Path]:
    grid = output_dir / "shards/grids" / split / f"{group_id}.json"
    image_dir = output_dir / "images" / split / group_id
    return (
        grid,
        image_dir / f"current_{condition}.png",
        image_dir / f"target_00_{condition}.png",
    )


def gaussian_from_state(state: Mapping[str, Any]) -> np.ndarray:
    """Render an explicitly requested beam state on the source sensor grid."""
    y, x = np.ogrid[:1024, :1024]
    exponent = (
        ((x - float(state["centroid_x_px"])) / float(state["sigma_x_px"])) ** 2
        + ((y - float(state["centroid_y_px"])) / float(state["sigma_y_px"])) ** 2
    )
    return float(state["peak_intensity"]) * np.exp(-0.5 * exponent)


def visual_target_specs(
    states: np.ndarray, seed: int, group_id: str
) -> list[dict[str, Any]]:
    """Choose one unique, one ambiguous, and one infeasible target when possible."""
    by_status: dict[str, list[tuple[int, list[int]]]] = {
        "unique": [],
        "ambiguous": [],
    }
    for index, desired in enumerate(states):
        matches = np.flatnonzero(matching_mask(states, desired)).tolist()
        status = "unique" if len(matches) == 1 else "ambiguous"
        by_status[status].append((index, matches))

    reachable: list[dict[str, Any]] = []
    fallback = by_status["unique"] + by_status["ambiguous"]
    for status in ("unique", "ambiguous"):
        pool = by_status[status] or fallback
        position = int(stable_token(seed, group_id, status)[:8], 16) % len(pool)
        target_index, matches = pool[position]
        actual_status = "unique" if len(matches) == 1 else "ambiguous"
        reachable.append(
            {
                "status": actual_status,
                "source_action_index": target_index,
                "matching_indices": matches,
                "selected_index": minimum_motion_index(matches),
            }
        )

    rng = stable_rng(seed, group_id, "visual_infeasible")
    infeasible = None
    for variant in range(20):
        desired = states[(variant * 13 + 7) % len(states)].copy()
        direction = -1.0 if variant % 2 else 1.0
        desired[0] += direction * (5.0 + variant * 0.5)
        desired[1] += rng.choice((-1.0, 1.0)) * (4.0 + variant * 0.25)
        desired[2] += 2.8 + variant * 0.2
        desired[3] += 2.8 + variant * 0.2
        desired[4] *= 1.12 + variant * 0.015
        if not np.flatnonzero(matching_mask(states, desired)).size:
            infeasible = desired
            break
    if infeasible is None:
        raise RuntimeError(f"{group_id}: unable to construct infeasible visual target")
    reachable.append(
        {
            "status": "infeasible_within_limits",
            "source_action_index": None,
            "matching_indices": [],
            "selected_index": None,
            "desired_values": infeasible,
        }
    )
    return reachable


def build_group(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
    from optics_understanding_sft.build_dataset import simulator_result
    from optics_understanding_sft.direction_inverse_v1.build_inverse import (
        config_from_visible,
    )

    output_dir = Path(job["output_dir"])
    split = str(job["split"])
    group_id = str(job["group_id"])
    condition = str(job["condition"])
    grid_path, current_path, desired_path = group_paths(
        output_dir, split, group_id, condition
    )
    if grid_path.is_file() and current_path.is_file() and desired_path.is_file():
        existing = read_json(grid_path)
        visual_targets = existing.get("visual_targets", [])
        target_files_exist = all(
            (output_dir / target["image"]).is_file() for target in visual_targets
        )
        if (
            len(existing.get("candidates", [])) == 81
            and len(visual_targets) == int(job["visual_targets_per_group"])
            and target_files_exist
        ):
            return {
                "group_id": group_id,
                "split": split,
                "resumed": True,
                "seconds": 0.0,
            }

    started = time.perf_counter()
    seed = int(job["seed"])
    setup = sampled_setup(seed, split, group_id)
    visible = {**setup, "sensor_resolution_px": [1024, 1024]}
    base = load_sim_yaml(REPO_ROOT / "optical_sim/configs/base_config.yaml")
    simulator_config = config_from_visible(visible, base)
    actions = fixed_action_grid()
    before = simulator_result(simulator_config)
    current_state = rounded_state(before)
    desired_index = int(stable_token(seed, group_id, "desired")[:8], 16) % 80
    if desired_index >= 40:
        desired_index += 1
    candidates = []
    desired_result = None
    for index, action in enumerate(actions):
        result = simulator_result(simulator_config, action)
        next_state = rounded_state(result)
        change, directions = change_and_directions(current_state, next_state)
        candidates.append(
            {
                "action": action,
                "next_state": next_state,
                "change": {key: round(value, 9) for key, value in change.items()},
                "directions": directions,
            }
        )
        if index == desired_index:
            desired_result = result
    if desired_result is None:
        raise AssertionError(desired_index)

    states = np.asarray(
        [
            [float(candidate["next_state"][key]) for key in STATE_FIELDS]
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    target_specs = visual_target_specs(states, seed, group_id)
    rendered_targets = []
    for target_number, target in enumerate(target_specs):
        source_index = target["source_action_index"]
        if source_index is None:
            desired_values = np.asarray(target.pop("desired_values"), dtype=np.float64)
            intensity = gaussian_from_state(
                {
                    key: desired_values[index]
                    for index, key in enumerate(STATE_FIELDS)
                }
            )
        else:
            desired_values = states[int(source_index)]
            result = (
                desired_result
                if int(source_index) == desired_index
                else simulator_result(simulator_config, actions[int(source_index)])
            )
            intensity = np.asarray(result["intensity"])
        image_path = current_path.parent / (
            f"target_{target_number:02d}_{condition}.png"
        )
        rendered_targets.append(
            {
                **target,
                "desired_values": desired_values,
                "intensity": intensity,
                "image_path": image_path,
            }
        )
    high = max(
        float(np.asarray(before["intensity"]).max()),
        *(float(target["intensity"].max()) for target in rendered_targets),
        1e-12,
    )
    render_image(
        before["intensity"],
        high,
        condition,
        f"{seed}:{group_id}:current",
        current_path,
    )
    visual_targets = []
    for target_number, target in enumerate(rendered_targets):
        render_image(
            target["intensity"],
            high,
            condition,
            f"{seed}:{group_id}:target:{target_number}",
            target["image_path"],
        )
        desired_state = {
            key: round(float(target["desired_values"][index]), 9)
            for index, key in enumerate(STATE_FIELDS)
        }
        visual_targets.append(
            {
                "target_id": f"{group_id}_target_{target_number:02d}",
                "desired_beam_state": desired_state,
                "image": target["image_path"].relative_to(output_dir).as_posix(),
                "status": target["status"],
                "matching_indices": target["matching_indices"],
                "selected_index": target["selected_index"],
                "source_action_index": target["source_action_index"],
            }
        )
    first_reachable = next(
        target
        for target in visual_targets
        if target["status"] != "infeasible_within_limits"
    )
    if first_reachable["image"] != desired_path.relative_to(output_dir).as_posix():
        raise AssertionError("first visual target must use the legacy desired path")
    record = {
        "dataset_version": str(job["version"]),
        "group_id": group_id,
        "split": split,
        "condition": condition,
        "setup": {key: round(float(value), 9) for key, value in setup.items()},
        "current_beam_state": current_state,
        "desired_beam_state": first_reachable["desired_beam_state"],
        "desired_action_index": first_reachable["source_action_index"],
        "desired_matching_indices": first_reachable["matching_indices"],
        "current_image": current_path.relative_to(output_dir).as_posix(),
        "desired_image": desired_path.relative_to(output_dir).as_posix(),
        "visual_targets": visual_targets,
        "image_calibration": {
            "linear_intensity_low": 0.0,
            "linear_intensity_high": high,
            "gamma": 0.5,
            "source_sensor_resolution_px": [1024, 1024],
        },
        "candidates": candidates,
    }
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "group_id": group_id,
        "split": split,
        "resumed": False,
        "seconds": time.perf_counter() - started,
    }


def group_jobs(
    config: Mapping[str, Any],
    output_dir: Path,
    max_groups: int | None,
) -> list[dict[str, Any]]:
    jobs = []
    seed = int(config["seed"])
    for split, configured_count in config["group_counts"].items():
        count = int(configured_count)
        if max_groups is not None:
            count = min(count, max_groups)
        schedule = condition_schedule(config, split)
        if len(schedule) < count:
            raise ValueError(f"visual schedule for {split} has {len(schedule)}<{count}")
        group_ids = [f"v2_{split}_{index:06d}" for index in range(count)]
        group_ids.sort(key=lambda value: stable_token(seed, split, value))
        for index, group_id in enumerate(group_ids):
            jobs.append(
                {
                    "version": config["version"],
                    "seed": seed,
                    "split": split,
                    "group_id": group_id,
                    "condition": schedule[index],
                    "visual_targets_per_group": int(
                        config["visual_targets_per_group"]
                    ),
                    "output_dir": str(output_dir.resolve()),
                }
            )
    return jobs


def materialize_grids(
    output_dir: Path, config: Mapping[str, Any], max_groups: int | None
) -> dict[str, list[dict[str, Any]]]:
    grids: dict[str, list[dict[str, Any]]] = {}
    for split, configured_count in config["group_counts"].items():
        count = int(configured_count)
        if max_groups is not None:
            count = min(count, max_groups)
        paths = sorted((output_dir / "shards/grids" / split).glob("*.json"))
        rows = [read_json(path) for path in paths]
        if len(rows) != count:
            raise RuntimeError(f"{split}: expected {count} grids, found {len(rows)}")
        if any(len(row["candidates"]) != 81 for row in rows):
            raise RuntimeError(f"{split}: incomplete action grid")
        write_jsonl(output_dir / "grids" / f"{split}.jsonl", rows)
        grids[split] = rows
    return grids


def transition_rows(rows: list[dict[str, Any]]) -> Any:
    for row in rows:
        for index, candidate in enumerate(row["candidates"]):
            yield {
                "group_id": row["group_id"],
                "split": row["split"],
                "setup": row["setup"],
                "current_beam_state": row["current_beam_state"],
                "action_index": index,
                **candidate,
            }


def pair_pools(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {
        "unique": [],
        "ambiguous": [],
        "infeasible_within_limits": [],
    }
    for row in rows:
        states = np.asarray(
            [
                [float(candidate["next_state"][key]) for key in STATE_FIELDS]
                for candidate in row["candidates"]
            ],
            dtype=np.float64,
        )
        for target_index, desired_values in enumerate(states):
            matches = np.flatnonzero(matching_mask(states, desired_values)).tolist()
            status = "unique" if len(matches) == 1 else "ambiguous"
            pools[status].append(
                {
                    "group_id": row["group_id"],
                    "desired_beam_state": {
                        key: round(float(desired_values[index]), 9)
                        for index, key in enumerate(STATE_FIELDS)
                    },
                    "status": status,
                    "matching_indices": matches,
                    "selected_index": minimum_motion_index(matches),
                    "source_action_index": target_index,
                }
            )
        rng = stable_rng(seed, row["group_id"], "infeasible")
        for variant in range(6):
            desired = states[(variant * 13 + 7) % len(states)].copy()
            desired[0] += rng.choice((-1.0, 1.0)) * (4.0 + variant)
            desired[1] += rng.choice((-1.0, 1.0)) * (4.0 + variant / 2)
            desired[2] += 2.5 + variant * 0.3
            desired[3] += 2.5 + variant * 0.2
            desired[4] *= 1.10 + variant * 0.025
            matches = np.flatnonzero(matching_mask(states, desired)).tolist()
            if matches:
                continue
            pools["infeasible_within_limits"].append(
                {
                    "group_id": row["group_id"],
                    "desired_beam_state": {
                        key: round(float(desired[index]), 9)
                        for index, key in enumerate(STATE_FIELDS)
                    },
                    "status": "infeasible_within_limits",
                    "matching_indices": [],
                    "selected_index": None,
                    "source_action_index": None,
                }
            )
    return pools


def select_pairs(
    rows: list[dict[str, Any]],
    counts: Mapping[str, Any],
    seed: int,
    split: str,
    *,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    pools = pair_pools(rows, seed)
    output = []
    for status, target_count_raw in counts.items():
        target_count = int(target_count_raw)
        candidates = sorted(
            pools[status],
            key=lambda row: stable_token(
                seed,
                split,
                status,
                row["group_id"],
                row["source_action_index"],
            ),
        )
        if not candidates:
            if allow_missing:
                continue
            raise RuntimeError(f"{split}: no inverse candidates for {status}")
        for index in range(target_count):
            source = dict(candidates[index % len(candidates)])
            source["pair_id"] = f"{split}_{status}_{index:06d}"
            source["split"] = split
            output.append(source)
    output.sort(key=lambda row: stable_token(seed, row["pair_id"]))
    return output


def visual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        for target in row["visual_targets"]:
            output.append(
                {
                    "pair_id": target["target_id"],
                    "group_id": row["group_id"],
                    "split": row["split"],
                    "condition": row["condition"],
                    "setup": row["setup"],
                    "image_calibration": row["image_calibration"],
                    "current_image": row["current_image"],
                    "desired_image": target["image"],
                    "current_beam_state": row["current_beam_state"],
                    "desired_beam_state": target["desired_beam_state"],
                    "status": target["status"],
                    "matching_indices": target["matching_indices"],
                    "selected_index": target["selected_index"],
                }
            )
    return output


def measurement_rows(rows: list[dict[str, Any]]) -> Any:
    for row in rows:
        yield {
            "example_id": f"{row['group_id']}_current",
            "group_id": row["group_id"],
            "split": row["split"],
            "condition": row["condition"],
            "image": row["current_image"],
            "image_calibration": row["image_calibration"],
            "target_state": row["current_beam_state"],
        }
        for target in row["visual_targets"]:
            yield {
                "example_id": target["target_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "condition": row["condition"],
                "image": target["image"],
                "image_calibration": row["image_calibration"],
                "target_state": target["desired_beam_state"],
            }


def hash_files(output_dir: Path) -> tuple[int, str]:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {"checksums.sha256", "manifest.json"}
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output_dir).as_posix()}")
    content = "\n".join(lines) + "\n"
    (output_dir / "checksums.sha256").write_text(content, encoding="utf-8")
    return len(paths), hashlib.sha256(content.encode()).hexdigest()


def build_manifest(
    output_dir: Path,
    config: Mapping[str, Any],
    grids: Mapping[str, list[dict[str, Any]]],
    inverse_counts: Mapping[str, Mapping[str, int]],
    checksum_count: int | None,
    checksum_manifest_sha256: str | None,
) -> dict[str, Any]:
    split_summary = {}
    all_groups = set()
    for split, rows in grids.items():
        groups = {row["group_id"] for row in rows}
        if all_groups & groups:
            raise RuntimeError(f"group overlap detected for {split}")
        all_groups |= groups
        split_summary[split] = {
            "groups": len(rows),
            "transitions": len(rows) * 81,
            "images": sum(
                1 + len(row["visual_targets"]) for row in rows
            ),
            "visual_pairs": sum(len(row["visual_targets"]) for row in rows),
            "conditions": dict(Counter(row["condition"] for row in rows)),
        }
    manifest = {
        "dataset_version": config["version"],
        "seed": int(config["seed"]),
        "simulation_only": True,
        "action_grid_size": 81,
        "split_summary": split_summary,
        "total_groups": len(all_groups),
        "total_transitions": sum(item["transitions"] for item in split_summary.values()),
        "total_images": sum(item["images"] for item in split_summary.values()),
        "inverse_pairs": inverse_counts,
        "checksum_file_count": checksum_count,
        "checksum_manifest_sha256": checksum_manifest_sha256,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    workers = int(args.workers or config["workers"])
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between 1 and 4 for laptop safety")
    worker_affinities = tuple(
        value.strip()
        for value in args.worker_affinities.split(";")
        if value.strip()
    )
    if worker_affinities and len(worker_affinities) < workers:
        raise ValueError(
            "--worker-affinities must provide at least one CPU set per worker"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    jobs = group_jobs(config, output_dir, args.max_groups_per_split)
    started = time.perf_counter()
    completed = 0
    if workers == 1:
        iterator = map(build_group, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=configure_worker_affinity,
            initargs=(worker_affinities,),
        )
        iterator = pool.map(build_group, jobs, chunksize=1)
    try:
        for result in iterator:
            completed += 1
            if completed % 20 == 0 or completed == len(jobs):
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-9)
                remaining = (len(jobs) - completed) / max(rate, 1e-9)
                print(
                    f"groups {completed}/{len(jobs)} "
                    f"elapsed={elapsed/3600:.2f}h eta={remaining/3600:.2f}h "
                    f"last={result['split']}:{result['group_id']}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()

    grids = materialize_grids(output_dir, config, args.max_groups_per_split)
    for split, rows in grids.items():
        write_jsonl(
            output_dir / "transitions" / f"{split}.jsonl",
            transition_rows(rows),
        )
        write_jsonl(
            output_dir / "visual" / f"{split}.jsonl", visual_rows(rows)
        )
        write_jsonl(
            output_dir / "measurement" / f"{split}.jsonl",
            measurement_rows(rows),
        )

    inverse_summary: dict[str, dict[str, int]] = {}
    for split, counts in config["inverse_pair_counts"].items():
        effective_counts = (
            {
                status: min(int(count), max(1, len(grids[split]) * 2))
                for status, count in counts.items()
            }
            if args.max_groups_per_split is not None
            else counts
        )
        pairs = select_pairs(
            grids[split],
            effective_counts,
            int(config["seed"]),
            split,
            allow_missing=args.max_groups_per_split is not None,
        )
        write_jsonl(output_dir / "inverse" / f"{split}.jsonl", pairs)
        inverse_summary[split] = dict(Counter(row["status"] for row in pairs))

    checksum_count = checksum_digest = None
    if not args.skip_checksums:
        checksum_count, checksum_digest = hash_files(output_dir)
    manifest = build_manifest(
        output_dir,
        config,
        grids,
        inverse_summary,
        checksum_count,
        checksum_digest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
