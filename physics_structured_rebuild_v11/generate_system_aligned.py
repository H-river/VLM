#!/usr/bin/env python3
"""Generate new-version v11 system-aligned 81-action groups."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import Bounds, split_hash, stable_seed
from continuous_control_v12.schema import sha256_file
from continuous_control_v12.simulator import (
    REGIMES,
    build_optical_setup,
    default_simulator_fixed,
    sample_group_setup,
)
from control_rebuild_v5.simulator_grid import simulate_fixed_action_grid
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    change_and_directions,
)

DEFAULT_CONFIG = Path(__file__).with_name("config_v11.json")
DEFAULT_V12_CONFIG = (
    REPO_ROOT / "continuous_control_v12/config_v12.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v11/system_aligned"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v12-config", type=Path, default=DEFAULT_V12_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-groups", type=int)
    parser.add_argument("--max-development-groups", type=int)
    return parser.parse_args()


def rounded_state(values: dict[str, Any]) -> dict[str, float]:
    return {field: round(float(values[field]), 9) for field in STATE_FIELDS}


def group_row(
    *,
    split: str,
    group_index: int,
    seed: int,
    bounds: Bounds,
    simulator_fixed: dict[str, Any],
    base_config_path: str,
) -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    group_id = f"v11_system_{split}_{group_index:06d}"
    regime = REGIMES[
        stable_seed(seed, group_id, "regime") % len(REGIMES)
    ]
    setup_context, positions = sample_group_setup(
        regime, group_id, seed, simulator_fixed, bounds
    )
    setup = build_optical_setup(
        setup_context, positions, simulator_fixed, base_config_path
    )
    surface = simulate_fixed_action_grid(setup)
    zero = surface[40]
    if any(abs(float(value)) > 0 for value in zero["action"].values()):
        raise AssertionError("legacy zero action moved")
    current = rounded_state(zero["state"])
    candidates = []
    for item in surface:
        next_state = rounded_state(item["state"])
        change, directions = change_and_directions(current, next_state)
        candidates.append(
            {
                "action": {
                    key: float(value) for key, value in item["action"].items()
                },
                "next_state": next_state,
                "change": {
                    key: round(float(value), 9)
                    for key, value in change.items()
                },
                "directions": directions,
            }
        )
    return {
        "dataset_version": "physics_structured_rebuild_v11_system_aligned",
        "split": split,
        "group_id": group_id,
        "regime": regime,
        "setup": {
            **setup_context,
            "lens_x_offset_mm": float(positions["lens_x_mm"]),
            "lens_y_offset_mm": float(positions["lens_y_mm"]),
            "camera_x_offset_mm": float(positions["camera_x_mm"]),
            "camera_y_offset_mm": float(positions["camera_y_mm"]),
        },
        "current_beam_state": current,
        "candidates": candidates,
    }


def group_row_job(job: dict[str, Any]) -> dict[str, Any]:
    return group_row(**job)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    v12_config = json.loads(
        args.v12_config.resolve().read_text(encoding="utf-8")
    )
    output_dir = args.output_dir.resolve()
    if "physics_structured_rebuild_v10" in output_dir.parts:
        raise ValueError("v11 generation refuses every v10 output path")
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite v11 dataset: {output_dir}")
    output_dir.mkdir(parents=True)
    seed = int(config["seed"])
    bounds = Bounds.from_config(v12_config)
    base_config_path = str(
        (REPO_ROOT / v12_config["simulator"]["base_config"]).resolve()
    )
    if args.smoke:
        counts = {"train": 32, "development": 8}
        simulator_fixed = default_simulator_fixed(
            base_config_path, grid_size=128, sensor_resolution=[128, 128]
        )
        scale = "smoke"
    else:
        counts = dict(config["system_aligned_generation"]["group_counts"])
        if args.max_train_groups is not None:
            counts["train"] = min(
                counts["train"], int(args.max_train_groups)
            )
        if args.max_development_groups is not None:
            counts["development"] = min(
                counts["development"],
                int(args.max_development_groups),
            )
        simulator_fixed = default_simulator_fixed(base_config_path)
        scale = "full" if counts["train"] == 10500 else "partial"
    summaries = {}
    regime_counts: Counter[str] = Counter()
    split_ids = {}
    for split in ("train", "development"):
        indices = list(range(int(counts[split])))
        random.Random(seed + (0 if split == "train" else 1)).shuffle(indices)
        jobs = [
            {
                "split": split,
                "group_index": index,
                "seed": seed,
                "bounds": bounds,
                "simulator_fixed": simulator_fixed,
                "base_config_path": base_config_path,
            }
            for index in indices
        ]
        path = output_dir / "grids" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".jsonl.tmp")
        identifiers = []
        workers = (
            1
            if args.smoke
            else int(config["system_aligned_generation"]["workers"])
        )
        with temporary.open("w", encoding="utf-8") as stream:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for row in executor.map(group_row_job, jobs, chunksize=1):
                    stream.write(
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    regime_counts[row["regime"]] += 1
                    identifiers.append(str(row["group_id"]))
        temporary.replace(path)
        split_ids[split] = set(identifiers)
        summaries[split] = {
            "groups": len(identifiers),
            "transitions": len(identifiers) * 81,
            "group_id_hash": split_hash(identifiers),
            "jsonl_sha256": sha256_file(path),
        }
    overlap = split_ids["train"] & split_ids["development"]
    if overlap:
        raise ValueError("v11 system-aligned splits overlap")
    manifest = {
        "version": "physics_structured_rebuild_v11_system_aligned",
        "scale": scale,
        "seed": seed,
        "action_order": "canonical_legacy_81",
        "groups_are_statistical_unit": True,
        "split_summary": summaries,
        "cross_split_group_overlap": 0,
        "regime_counts": dict(sorted(regime_counts.items())),
        "simulator_fixed": simulator_fixed,
        "absolute_limit_source": v12_config["absolute_limit_source"],
        "v9_or_v10_artifacts_modified": False,
        "v10_jsonl_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
