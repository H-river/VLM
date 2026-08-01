#!/usr/bin/env python3
"""Generate resumable v5 numerical grids with cached simulator propagation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v4.build_numerical_dataset import sample_setup
from control_rebuild_v5.simulator_grid import simulate_fixed_action_grid
from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    config_from_visible,
)
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    change_and_directions,
    read_json,
    stable_rng,
    write_jsonl,
)

DEFAULT_CONFIG = Path(__file__).with_name("numerical_dataset_config.json")
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
CATEGORIES = (
    "iid_expanded",
    "ood_boundary",
    "high_nonlinearity",
    "hard_interaction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def hard_interaction_setup(
    seed: int,
    split: str,
    group_id: str,
) -> dict[str, float]:
    setup = sample_setup(seed, split, group_id, "high_nonlinearity")
    rng = stable_rng(seed, split, group_id, "hard_interaction")
    setup["lens_aperture_mm"] = rng.uniform(16.0, 20.0)
    setup["lens_focal_length_mm"] = rng.choice(
        (rng.uniform(70.0, 82.0), rng.uniform(124.0, 136.0))
    )
    setup["source_to_lens_mm"] = rng.choice(
        (rng.uniform(120.0, 145.0), rng.uniform(260.0, 282.0))
    )
    setup["lens_to_camera_mm"] = rng.choice(
        (rng.uniform(78.0, 98.0), rng.uniform(194.0, 212.0))
    )
    lens_x_sign = -1.0 if rng.random() < 0.5 else 1.0
    lens_y_sign = -1.0 if rng.random() < 0.5 else 1.0
    camera_x_sign = -lens_x_sign if rng.random() < 0.5 else lens_x_sign
    camera_y_sign = -lens_y_sign if rng.random() < 0.5 else lens_y_sign
    setup["lens_x_offset_mm"] = lens_x_sign * rng.uniform(0.25, 0.32)
    setup["lens_y_offset_mm"] = lens_y_sign * rng.uniform(0.25, 0.32)
    setup["camera_x_offset_mm"] = camera_x_sign * rng.uniform(0.18, 0.24)
    setup["camera_y_offset_mm"] = camera_y_sign * rng.uniform(0.18, 0.24)
    return setup


def rounded_state(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        field: round(float(result[field]), 9) for field in STATE_FIELDS
    }


def build_group(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    started = time.perf_counter()
    output_dir = Path(job["output_dir"])
    group_id = str(job["group_id"])
    shard = output_dir / "shards" / "train" / f"{group_id}.json"
    if shard.is_file():
        existing = read_json(shard)
        if len(existing.get("candidates", ())) == 81:
            return {
                "group_id": group_id,
                "category": str(existing["source_category"]),
                "resumed": True,
                "seconds": 0.0,
            }

    category = str(job["category"])
    if category == "hard_interaction":
        setup_values = hard_interaction_setup(
            int(job["seed"]),
            "train",
            group_id,
        )
    else:
        setup_values = sample_setup(
            int(job["seed"]),
            "train",
            group_id,
            category,
        )
    visible = {**setup_values, "sensor_resolution_px": [1024, 1024]}
    base = load_sim_yaml(REPO_ROOT / "optical_sim/configs/base_config.yaml")
    setup = setup_from_dict(config_from_visible(visible, base))
    simulated = simulate_fixed_action_grid(setup)
    zero = simulated[40]
    if any(float(value) != 0.0 for value in zero["action"].values()):
        raise AssertionError("action index 40 must be the zero action")
    current = rounded_state(zero["state"])
    candidates = []
    for item in simulated:
        after = rounded_state(item["state"])
        change, directions = change_and_directions(current, after)
        candidates.append(
            {
                "action": item["action"],
                "next_state": after,
                "change": {
                    key: round(float(value), 9)
                    for key, value in change.items()
                },
                "directions": directions,
            }
        )
    row = {
        "dataset_version": str(job["version"]),
        "split": "train",
        "group_id": group_id,
        "source_category": category,
        "setup": setup_values,
        "current_beam_state": current,
        "candidates": candidates,
    }
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(shard)
    return {
        "group_id": group_id,
        "category": category,
        "resumed": False,
        "seconds": time.perf_counter() - started,
    }


def checksum_files(output_dir: Path) -> tuple[int, str]:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {"checksums.sha256", "manifest.json"}
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output_dir)}")
    content = "\n".join(lines) + "\n"
    (output_dir / "checksums.sha256").write_text(content, encoding="utf-8")
    return len(paths), hashlib.sha256(content.encode()).hexdigest()


def category_schedule(
    seed: int,
    count: int,
    requested: Mapping[str, Any],
) -> list[str]:
    configured_total = sum(int(value) for value in requested.values())
    if count > configured_total:
        raise ValueError("max-groups exceeds configured category total")
    schedule = [
        category
        for category in CATEGORIES
        for _ in range(int(requested[category]))
    ]
    random.Random(seed).shuffle(schedule)
    return schedule[:count]


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    configured_count = int(config["train_groups"])
    count = (
        configured_count
        if args.max_groups is None
        else min(configured_count, int(args.max_groups))
    )
    workers = int(args.workers or config["workers"])
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between one and four")
    categories = category_schedule(
        int(config["seed"]),
        count,
        config["category_counts"],
    )
    jobs = [
        {
            "version": config["version"],
            "seed": int(config["seed"]),
            "output_dir": str(output_dir),
            "group_id": f"v5_train_{index:06d}",
            "category": categories[index],
        }
        for index in range(count)
    ]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(build_group, jobs, chunksize=1):
            results.append(result)
            if len(results) % 25 == 0 or len(results) == len(jobs):
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "total": len(jobs),
                            "latest_seconds": result["seconds"],
                            "resumed": sum(
                                bool(item["resumed"]) for item in results
                            ),
                        }
                    ),
                    flush=True,
                )
    rows = [
        read_json(
            output_dir / "shards" / "train" / f"{job['group_id']}.json"
        )
        for job in jobs
    ]
    rows.sort(key=lambda row: str(row["group_id"]))
    write_jsonl(output_dir / "grids" / "train.jsonl", rows)
    manifest = {
        "dataset_version": config["version"],
        "seed": int(config["seed"]),
        "action_count_per_group": 81,
        "train_groups": len(rows),
        "train_transitions": len(rows) * 81,
        "category_counts": dict(
            sorted(Counter(str(row["source_category"]) for row in rows).items())
        ),
        "resumed_groups": sum(bool(item["resumed"]) for item in results),
        "worker_seconds": sum(float(item["seconds"]) for item in results),
        "wall_seconds_before_checksums": time.perf_counter() - started,
        "source_dataset_overlap": 0,
        "held_out_test_groups_used": 0,
        "frozen_validation_files_modified": False,
        "simulator_optimization": (
            "one_source_propagation_nine_lens_propagations_"
            "eighty_one_sensor_extractions"
        ),
    }
    if not args.skip_checksums:
        file_count, digest = checksum_files(output_dir)
        manifest["checksum_file_count"] = file_count
        manifest["checksum_manifest_sha256"] = digest
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
