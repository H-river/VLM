#!/usr/bin/env python3
"""Re-render corrected high-precision measurement images from frozen v2 grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import current_process
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measurement_rebuild_v3.common import (
    iter_jsonl,
    read_json,
    stable_seed,
    write_jsonl,
)
from specialist_rebuild_v2.common import STATE_FIELDS, fixed_action_grid


DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_SOURCE = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"
SPLITS = (
    "train",
    "val",
    "test_iid",
    "test_ood_physics",
    "test_visual_stress",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--worker-affinities",
        default="0;8;1,9",
        help="Semicolon-separated CPU sets, one per worker.",
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


def gaussian_from_state(state: Mapping[str, Any]) -> np.ndarray:
    y, x = np.ogrid[:1024, :1024]
    exponent = (
        ((x - float(state["centroid_x_px"])) / float(state["sigma_x_px"])) ** 2
        + ((y - float(state["centroid_y_px"])) / float(state["sigma_y_px"])) ** 2
    )
    return float(state["peak_intensity"]) * np.exp(-0.5 * exponent)


def sensor_frame_state(
    state: Mapping[str, Any],
    setup: Mapping[str, Any],
    action: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Convert legacy lab-frame centroids to coordinates aligned with PNG pixels."""

    action = action or {}
    pitch_mm = float(setup["pixel_size_um"]) / 1000.0
    camera_x_mm = float(setup["camera_x_offset_mm"]) + float(
        action.get("camera_x_delta_mm", 0.0)
    )
    camera_y_mm = float(setup["camera_y_offset_mm"]) + float(
        action.get("camera_y_delta_mm", 0.0)
    )
    converted = {key: float(state[key]) for key in STATE_FIELDS}
    converted["centroid_x_px"] -= camera_x_mm / pitch_mm
    converted["centroid_y_px"] -= camera_y_mm / pitch_mm
    return converted


def save_linear_png(
    intensity: np.ndarray, high: float, resolution: int, path: Path
) -> None:
    normalized = np.clip(
        np.asarray(intensity, dtype=np.float32) / max(float(high), 1e-12),
        0.0,
        1.0,
    )
    resized = Image.fromarray(normalized).resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )
    encoded = np.rint(
        np.clip(np.asarray(resized, dtype=np.float64), 0.0, 1.0) * 65535.0
    ).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded).save(path, format="PNG", compress_level=3)


def strip_grid_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "group_id": row["group_id"],
        "split": row["split"],
        "setup": row["setup"],
        "current_beam_state": row["current_beam_state"],
        "visual_targets": row["visual_targets"],
    }


def build_group(job: Mapping[str, Any]) -> dict[str, Any]:
    from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
    from optics_understanding_sft.build_dataset import simulator_result
    from optics_understanding_sft.direction_inverse_v1.build_inverse import (
        config_from_visible,
    )

    output_dir = Path(job["output_dir"])
    row = job["row"]
    split = str(row["split"])
    group_id = str(row["group_id"])
    shard_path = output_dir / "shards" / split / f"{group_id}.json"
    if shard_path.is_file():
        existing = read_json(shard_path)
        if (
            existing.get("dataset_version") == job["version"]
            and len(existing.get("states", [])) == 4
            and all(
                (output_dir / item["base_image"]).is_file()
                for item in existing["states"]
            )
        ):
            return {"split": split, "group_id": group_id, "resumed": True}

    started = time.perf_counter()
    visible = {**row["setup"], "sensor_resolution_px": [1024, 1024]}
    base = load_sim_yaml(REPO_ROOT / "optical_sim/configs/base_config.yaml")
    simulator_config = config_from_visible(visible, base)
    current = simulator_result(simulator_config)
    action_grid = fixed_action_grid()
    action_cache: dict[int, Mapping[str, Any]] = {}
    entries: list[
        tuple[
            str,
            str,
            Mapping[str, Any],
            np.ndarray,
            str,
            Mapping[str, Any] | None,
        ]
    ] = [
        (
            f"{group_id}_current",
            "current",
            sensor_frame_state(current["state"], row["setup"]),
            np.asarray(current["intensity"]),
            "current",
            None,
        )
    ]
    for number, target in enumerate(row["visual_targets"]):
        source_index = target.get("source_action_index")
        if source_index is None:
            intensity = gaussian_from_state(target["desired_beam_state"])
            target_state = target["desired_beam_state"]
            origin = "analytic_infeasible_target"
            action = None
        else:
            source_index = int(source_index)
            if source_index not in action_cache:
                action_cache[source_index] = simulator_result(
                    simulator_config, action_grid[source_index]
                )
            result = action_cache[source_index]
            action = action_grid[source_index]
            intensity = np.asarray(result["intensity"])
            target_state = sensor_frame_state(
                result["state"], row["setup"], action
            )
            origin = "simulator_action_target"
        entries.append(
            (
                str(target.get("target_id", f"{group_id}_target_{number:02d}")),
                f"target_{number:02d}",
                target_state,
                intensity,
                origin,
                action,
            )
        )

    high = max(float(np.max(item[3])) for item in entries)
    resolution = int(job["stored_resolution_px"])
    states = []
    for state_id, role, target_state, intensity, origin, action in entries:
        relative = (
            Path("base_images") / split / group_id / f"{role}_linear16.png"
        )
        save_linear_png(intensity, high, resolution, output_dir / relative)
        states.append(
            {
                "state_id": state_id,
                "group_id": group_id,
                "split": split,
                "role": role,
                "origin": origin,
                "coordinate_frame": "camera_sensor_array",
                "base_image": relative.as_posix(),
                "setup": {
                    key: round(float(value), 9)
                    for key, value in row["setup"].items()
                },
                "source_action": (
                    {
                        key: round(float(value), 9)
                        for key, value in action.items()
                    }
                    if action is not None
                    else None
                ),
                "target_state": {
                    key: round(float(target_state[key]), 9)
                    for key in STATE_FIELDS
                },
                "image_calibration": {
                    "linear_intensity_low": 0.0,
                    "linear_intensity_high": high,
                    "source_sensor_resolution_px": [1024, 1024],
                    "stored_resolution_px": [resolution, resolution],
                    "stored_bit_depth": 16,
                    "stored_transfer": "linear",
                    "coordinate_frame": "camera_sensor_array",
                },
            }
        )
    shard = {
        "dataset_version": job["version"],
        "source_dataset_version": job["source_version"],
        "group_id": group_id,
        "split": split,
        "states": states,
    }
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text(
        json.dumps(shard, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "split": split,
        "group_id": group_id,
        "resumed": False,
        "seconds": time.perf_counter() - started,
    }


def load_jobs(
    source_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    max_groups: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    jobs = []
    counts = {}
    for split in SPLITS:
        rows = iter_jsonl(source_dir / "grids" / f"{split}.jsonl")
        selected = []
        for index, row in enumerate(rows):
            if max_groups is not None and index >= max_groups:
                break
            selected.append(strip_grid_row(row))
        counts[split] = len(selected)
        for row in selected:
            jobs.append(
                {
                    "version": config["version"],
                    "source_version": config["source_dataset_version"],
                    "stored_resolution_px": int(config["stored_resolution_px"]),
                    "output_dir": str(output_dir),
                    "row": row,
                }
            )
    return jobs, counts


def materialize(
    output_dir: Path,
    config: Mapping[str, Any],
    split_counts: Mapping[str, int],
) -> dict[str, Any]:
    split_summary = {}
    all_groups: set[str] = set()
    conditions = list(config["conditions"])
    for split, expected_groups in split_counts.items():
        shards = sorted((output_dir / "shards" / split).glob("*.json"))
        if len(shards) != expected_groups:
            raise RuntimeError(
                f"{split}: expected {expected_groups} shards, found {len(shards)}"
            )
        states = []
        for path in shards:
            shard = read_json(path)
            if shard["group_id"] in all_groups:
                raise RuntimeError(f"overlapping group: {shard['group_id']}")
            all_groups.add(shard["group_id"])
            states.extend(shard["states"])
        views = []
        for state in states:
            for condition in conditions:
                transform = dict(config["condition_parameters"][condition])
                views.append(
                    {
                        **state,
                        "view_id": f"{state['state_id']}:{condition}",
                        "condition": condition,
                        "transform": transform,
                        "transform_seed": stable_seed(
                            state["state_id"], condition, "view"
                        ),
                    }
                )
        write_jsonl(output_dir / "states" / f"{split}.jsonl", states)
        write_jsonl(output_dir / "views" / f"{split}.jsonl", views)
        split_summary[split] = {
            "groups": expected_groups,
            "base_states": len(states),
            "base_images": len(states),
            "views": len(views),
            "conditions": dict(Counter(row["condition"] for row in views)),
        }
    return {
        "dataset_version": config["version"],
        "source_dataset_version": config["source_dataset_version"],
        "stored_resolution_px": int(config["stored_resolution_px"]),
        "stored_bit_depth": int(config["stored_bit_depth"]),
        "conditions": conditions,
        "split_summary": split_summary,
        "total_groups": len(all_groups),
        "total_base_states": sum(
            row["base_states"] for row in split_summary.values()
        ),
        "total_base_images": sum(
            row["base_images"] for row in split_summary.values()
        ),
        "total_views": sum(row["views"] for row in split_summary.values()),
    }


def write_checksums(output_dir: Path) -> tuple[int, str]:
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


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    workers = int(args.workers)
    if workers < 1 or workers > 3:
        raise ValueError("workers must be between 1 and 3 for laptop safety")
    affinities = tuple(
        value.strip()
        for value in args.worker_affinities.split(";")
        if value.strip()
    )
    if affinities and len(affinities) < workers:
        raise ValueError("one worker affinity must be supplied per worker")
    source_dir = args.source_dir.resolve()
    source_manifest = read_json(source_dir / "manifest.json")
    if (
        source_manifest["dataset_version"]
        != config["source_dataset_version"]
    ):
        raise RuntimeError("source dataset version does not match config")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    jobs, split_counts = load_jobs(
        source_dir, output_dir, config, args.max_groups_per_split
    )
    started = time.perf_counter()
    if workers == 1:
        iterator = map(build_group, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=configure_worker_affinity,
            initargs=(affinities,),
        )
        iterator = pool.map(build_group, jobs, chunksize=1)
    completed = 0
    try:
        for result in iterator:
            completed += 1
            if completed % 20 == 0 or completed == len(jobs):
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-9)
                eta = (len(jobs) - completed) / max(rate, 1e-9)
                print(
                    f"groups {completed}/{len(jobs)} "
                    f"elapsed={elapsed/3600:.3f}h eta={eta/3600:.3f}h "
                    f"last={result['split']}:{result['group_id']}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()
    manifest = materialize(output_dir, config, split_counts)
    if not args.skip_checksums:
        count, digest = write_checksums(output_dir)
        manifest["checksum_file_count"] = count
        manifest["checksum_manifest_sha256"] = digest
    else:
        manifest["checksum_file_count"] = None
        manifest["checksum_manifest_sha256"] = None
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
