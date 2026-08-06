#!/usr/bin/env python3
"""Generate resumable, disjoint numerical grids in difficult physics regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    change_and_directions,
    fixed_action_grid,
    read_json,
    stable_rng,
    stable_token,
    write_jsonl,
)

DEFAULT_CONFIG = Path(__file__).with_name("numerical_dataset_config.json")
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_data/control_rebuild_v4_numerical"
CATEGORIES = ("iid_expanded", "ood_boundary", "high_nonlinearity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-groups-per-split", type=int)
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def category_for(
    seed: int,
    split: str,
    group_id: str,
    fractions: Mapping[str, Any],
) -> str:
    token = int(stable_token(seed, split, group_id, "category")[:12], 16)
    unit = token / float(16**12)
    cumulative = 0.0
    for category in CATEGORIES:
        cumulative += float(fractions[category])
        if unit < cumulative:
            return category
    return CATEGORIES[-1]


def sample_setup(
    seed: int, split: str, group_id: str, category: str
) -> dict[str, float]:
    rng = stable_rng(seed, split, group_id, category, "setup")
    if category == "iid_expanded":
        return {
            "wavelength_nm": rng.uniform(617.0, 648.0),
            "beam_waist_mm": rng.uniform(0.70, 1.35),
            "power_w": rng.uniform(0.75, 1.25),
            "lens_focal_length_mm": rng.uniform(82.0, 120.0),
            "lens_aperture_mm": rng.uniform(19.0, 32.0),
            "source_to_lens_mm": rng.uniform(140.0, 260.0),
            "lens_to_camera_mm": rng.uniform(92.0, 190.0),
            "lens_x_offset_mm": rng.uniform(-0.23, 0.23),
            "lens_y_offset_mm": rng.uniform(-0.23, 0.23),
            "camera_x_offset_mm": rng.uniform(-0.18, 0.18),
            "camera_y_offset_mm": rng.uniform(-0.18, 0.18),
            "pixel_size_um": rng.choice((5.0, 5.5, 6.0)),
        }
    if category == "ood_boundary":
        values = {
            "wavelength_nm": rng.uniform(598.0, 672.0),
            "beam_waist_mm": rng.uniform(0.56, 1.52),
            "power_w": rng.uniform(0.58, 1.42),
            "lens_focal_length_mm": rng.uniform(70.0, 135.0),
            "lens_aperture_mm": rng.uniform(16.0, 36.0),
            "source_to_lens_mm": rng.uniform(120.0, 282.0),
            "lens_to_camera_mm": rng.uniform(78.0, 210.0),
            "lens_x_offset_mm": rng.uniform(-0.30, 0.30),
            "lens_y_offset_mm": rng.uniform(-0.30, 0.30),
            "camera_x_offset_mm": rng.uniform(-0.23, 0.23),
            "camera_y_offset_mm": rng.uniform(-0.23, 0.23),
            "pixel_size_um": rng.choice((4.5, 5.0, 5.5, 6.0, 6.5)),
        }
        axes = (
            ("wavelength_nm", (598.0, 614.0), (650.0, 672.0)),
            ("beam_waist_mm", (0.56, 0.70), (1.35, 1.52)),
            ("power_w", (0.58, 0.74), (1.26, 1.42)),
            ("lens_focal_length_mm", (70.0, 81.0), (121.0, 135.0)),
        )
        axis = int(stable_token(seed, group_id, "boundary_axis")[:8], 16) % len(axes)
        field, low, high = axes[axis]
        interval = low if rng.random() < 0.5 else high
        values[field] = rng.uniform(*interval)
        return values
    if category != "high_nonlinearity":
        raise ValueError(category)
    short_focal = rng.random() < 0.5
    small_waist = rng.random() < 0.5
    return {
        "wavelength_nm": rng.uniform(605.0, 665.0),
        "beam_waist_mm": (
            rng.uniform(0.56, 0.78) if small_waist else rng.uniform(1.30, 1.52)
        ),
        "power_w": rng.uniform(0.62, 1.38),
        "lens_focal_length_mm": (
            rng.uniform(70.0, 88.0) if short_focal else rng.uniform(118.0, 136.0)
        ),
        "lens_aperture_mm": rng.uniform(16.0, 23.0),
        "source_to_lens_mm": rng.choice(
            (rng.uniform(120.0, 155.0), rng.uniform(250.0, 282.0))
        ),
        "lens_to_camera_mm": rng.choice(
            (rng.uniform(78.0, 103.0), rng.uniform(185.0, 212.0))
        ),
        "lens_x_offset_mm": rng.choice(
            (rng.uniform(-0.32, -0.20), rng.uniform(0.20, 0.32))
        ),
        "lens_y_offset_mm": rng.choice(
            (rng.uniform(-0.32, -0.20), rng.uniform(0.20, 0.32))
        ),
        "camera_x_offset_mm": rng.uniform(-0.24, 0.24),
        "camera_y_offset_mm": rng.uniform(-0.24, 0.24),
        "pixel_size_um": rng.choice((4.5, 5.5, 6.5)),
    }


def rounded_state(result: Mapping[str, Any]) -> dict[str, float]:
    return {field: round(float(result["state"][field]), 9) for field in STATE_FIELDS}


def build_group(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
    from optics_understanding_sft.build_dataset import simulator_result
    from optics_understanding_sft.direction_inverse_v1.build_inverse import (
        config_from_visible,
    )

    output_dir = Path(job["output_dir"])
    split = str(job["split"])
    group_id = str(job["group_id"])
    shard_path = output_dir / "shards" / split / f"{group_id}.json"
    if shard_path.is_file():
        existing = read_json(shard_path)
        if len(existing.get("candidates", [])) == 81:
            return {
                "group_id": group_id,
                "split": split,
                "category": str(existing["source_category"]),
                "resumed": True,
                "seconds": 0.0,
            }
    started = time.perf_counter()
    setup = sample_setup(
        int(job["seed"]),
        split,
        group_id,
        str(job["category"]),
    )
    visible = {**setup, "sensor_resolution_px": [1024, 1024]}
    base = load_sim_yaml(REPO_ROOT / "optical_sim/configs/base_config.yaml")
    simulator_config = config_from_visible(visible, base)
    current = rounded_state(simulator_result(simulator_config))
    candidates = []
    for action in fixed_action_grid():
        after = rounded_state(simulator_result(simulator_config, action))
        change, directions = change_and_directions(current, after)
        candidates.append(
            {
                "action": action,
                "next_state": after,
                "change": {
                    key: round(float(value), 9) for key, value in change.items()
                },
                "directions": directions,
            }
        )
    row = {
        "dataset_version": str(job["version"]),
        "split": split,
        "group_id": group_id,
        "source_category": str(job["category"]),
        "setup": setup,
        "current_beam_state": current,
        "candidates": candidates,
    }
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(shard_path)
    return {
        "group_id": group_id,
        "split": split,
        "category": str(job["category"]),
        "resumed": False,
        "seconds": time.perf_counter() - started,
    }


def checksum_files(output_dir: Path) -> tuple[int, str]:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256", "manifest.json"}
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output_dir)}")
    content = "\n".join(lines) + "\n"
    (output_dir / "checksums.sha256").write_text(content, encoding="utf-8")
    return len(paths), hashlib.sha256(content.encode()).hexdigest()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seed = int(config["seed"])
    fractions = config["category_fractions"]
    if abs(sum(float(value) for value in fractions.values()) - 1.0) > 1e-9:
        raise ValueError("category fractions must sum to one")
    workers = int(args.workers or config["workers"])
    if workers < 1 or workers > 2:
        raise ValueError("workers must be one or two")
    started = time.perf_counter()
    split_summary = {}
    for split, configured_count in config["group_counts"].items():
        count = int(configured_count)
        if args.max_groups_per_split is not None:
            count = min(count, int(args.max_groups_per_split))
        jobs = []
        for index in range(count):
            group_id = f"v4_{split}_{index:06d}"
            category = category_for(seed, split, group_id, fractions)
            jobs.append(
                {
                    "version": config["version"],
                    "seed": seed,
                    "output_dir": str(output_dir),
                    "split": split,
                    "group_id": group_id,
                    "category": category,
                }
            )
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(build_group, jobs, chunksize=1):
                results.append(result)
                if len(results) % 25 == 0 or len(results) == len(jobs):
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "completed": len(results),
                                "total": len(jobs),
                                "latest_seconds": result["seconds"],
                            }
                        ),
                        flush=True,
                    )
        shard_paths = sorted((output_dir / "shards" / split).glob("*.json"))
        expected_ids = {str(job["group_id"]) for job in jobs}
        rows = [read_json(path) for path in shard_paths if path.stem in expected_ids]
        rows.sort(key=lambda row: str(row["group_id"]))
        write_jsonl(output_dir / "grids" / f"{split}.jsonl", rows)
        category_counts = Counter(str(row["source_category"]) for row in rows)
        split_summary[split] = {
            "groups": len(rows),
            "transitions": len(rows) * 81,
            "category_counts": dict(sorted(category_counts.items())),
            "resumed": sum(bool(row["resumed"]) for row in results),
            "worker_seconds": sum(float(row["seconds"]) for row in results),
        }
    manifest = {
        "dataset_version": config["version"],
        "seed": seed,
        "action_count_per_group": 81,
        "split_summary": split_summary,
        "source_dataset_overlap": 0,
        "held_out_v2_test_groups_used": 0,
        "seconds_before_checksums": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.skip_checksums:
        count, digest = checksum_files(output_dir)
        manifest["checksum_file_count"] = count
        manifest["checksum_manifest_sha256"] = digest
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
