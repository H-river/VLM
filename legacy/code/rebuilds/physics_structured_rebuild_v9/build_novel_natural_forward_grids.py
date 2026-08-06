#!/usr/bin/env python3
"""Build resumable natural-domain grids with setups disjoint from Qwen v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.scripts.build_dataset import sampled_setup
from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v5.simulator_grid import simulate_fixed_action_grid
from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    config_from_visible,
)
from specialist_rebuild_v2.common import (
    SETUP_FIELDS,
    STATE_FIELDS,
    setup_array,
    state_array,
)

DEFAULT_REFERENCE = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_inverse_adaptation/train.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/novel_natural_forward_grids"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-count", type=int, default=1000)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument(
        "--group-prefix",
        default="orch_v9_natural",
        help="New identifier namespace passed through the frozen Qwen sampler.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if int(args.group_count) < 1:
        raise ValueError("group-count must be positive")
    if not str(args.group_prefix).strip():
        raise ValueError("group-prefix must be non-empty")
    if any(float(value) != 0.0 for value in ACTION_GRID[40].values()):
        raise AssertionError("action index 40 must be the zero action")

    started = time.perf_counter()
    reference_path = args.reference_cache.resolve()
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_ids = {str(value) for value in reference["group_ids"]}
        reference_contexts = np.asarray(
            reference["contexts"][:, : len(SETUP_FIELDS) + len(STATE_FIELDS)],
            dtype=np.float32,
        )
    reference_context_keys = {
        row.tobytes() for row in np.ascontiguousarray(reference_contexts)
    }

    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml(REPO_ROOT / "optical_sim/configs/base_config.yaml")
    generated = 0
    resumed = 0
    context_keys: set[bytes] = set()
    context_overlap = 0
    for index in range(int(args.group_count)):
        group_id = f"{args.group_prefix}_{index:06d}"
        if group_id in reference_ids:
            raise ValueError(f"new group ID overlaps reference: {group_id}")
        shard = shard_dir / f"{index:04d}_{group_id}.npz"
        if shard.exists():
            with np.load(shard, allow_pickle=False) as cached:
                if str(cached["group_id"].item()) != group_id:
                    raise ValueError(f"cached group differs: {shard}")
                context = np.asarray(cached["context"], dtype=np.float32)
            resumed += 1
        else:
            setup_values = sampled_setup(group_id)
            visible = {
                **setup_values,
                "sensor_resolution_px": [1024, 1024],
            }
            setup = setup_from_dict(config_from_visible(visible, base_config))
            simulated = simulate_fixed_action_grid(setup)
            candidate_states = np.asarray(
                [
                    [float(item["state"][field]) for field in STATE_FIELDS]
                    for item in simulated
                ],
                dtype=np.float32,
            )
            current = {
                field: float(candidate_states[40, field_index])
                for field_index, field in enumerate(STATE_FIELDS)
            }
            context = np.concatenate(
                [setup_array(setup_values), state_array(current)]
            ).astype(np.float32)
            write_npz_atomic(
                shard,
                group_id=np.asarray(group_id),
                context=context,
                candidate_states=candidate_states,
            )
            generated += 1

        context_key = np.ascontiguousarray(context).tobytes()
        if context_key in context_keys:
            raise ValueError(f"duplicate generated context: {group_id}")
        context_keys.add(context_key)
        context_overlap += int(context_key in reference_context_keys)
        completed = index + 1
        if completed % int(args.report_every) == 0 or completed == int(
            args.group_count
        ):
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": int(args.group_count),
                        "generated": generated,
                        "resumed": resumed,
                        "reference_context_overlap": context_overlap,
                        "seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if context_overlap:
        raise ValueError("generated contexts overlap the reference cache")
    output_path = output_dir / "train.npz"
    manifest_path = output_dir / "manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError("refusing to overwrite consolidated novel grids")
    shard_paths = sorted(shard_dir.glob("*.npz"))
    if len(shard_paths) != int(args.group_count):
        raise ValueError("novel natural shard count differs")
    loaded = [np.load(path, allow_pickle=False) for path in shard_paths]
    write_npz_atomic(
        output_path,
        group_ids=np.asarray(
            [str(item["group_id"].item()) for item in loaded],
            dtype=np.str_,
        ),
        contexts=np.stack([item["context"] for item in loaded]),
        candidate_states=np.stack(
            [item["candidate_states"] for item in loaded]
        ),
    )
    for item in loaded:
        item.close()
    manifest = {
        "version": "novel_natural_forward_grids_v9",
        "count": int(args.group_count),
        "transition_count": 81 * int(args.group_count),
        "group_prefix": str(args.group_prefix),
        "setup_sampler": (
            "Qwen_orchestration.scripts.build_dataset.sampled_setup"
        ),
        "setup_sampler_seed": 20260724,
        "reference_cache": str(reference_path),
        "reference_cache_sha256": sha256(reference_path),
        "reference_group_id_overlap": 0,
        "reference_context_overlap": 0,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "simulator_grid_calls": generated,
        "resumed_shards": resumed,
        "held_out_validation_files_opened": [],
        "held_out_test_files_opened": [],
        "seconds": time.perf_counter() - started,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
