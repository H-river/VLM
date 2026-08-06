#!/usr/bin/env python3
"""Build resumable multi-positive inverse grids for natural training requests."""

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

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v5.simulator_grid import simulate_fixed_action_grid
from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    config_from_visible,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    STATE_FIELDS,
    inverse_context,
    matching_mask,
    raw_state_array,
)

DEFAULT_SOURCE = (
    REPO_ROOT.parent
    / "VLM_data/qwen_orchestration/v1/private/source_cases/train.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_inverse_adaptation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-every", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def action_index(action: dict) -> int:
    return next(
        index
        for index, candidate in enumerate(ACTION_GRID)
        if all(
            float(action[field]) == float(candidate[field])
            for field in ACTION_FIELDS
        )
    )


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source_path = args.source.resolve()
    rows = read_jsonl(source_path)
    if len(rows) != 1000:
        raise ValueError("expected 1,000 natural inverse training cases")
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml(
        REPO_ROOT / "optical_sim/configs/base_config.yaml"
    )
    generated = 0
    resumed = 0
    positive_counts = []
    for index, row in enumerate(rows):
        group_id = str(row["group_id"])
        shard = shard_dir / f"{index:04d}_{group_id}.npz"
        if shard.exists():
            cached = np.load(shard, allow_pickle=False)
            if str(cached["group_id"].item()) != group_id:
                raise ValueError(f"cached group differs: {shard}")
            positives = np.asarray(cached["positives"], dtype=np.bool_)
            resumed += 1
        else:
            visible = {
                **row["setup"],
                "sensor_resolution_px": [1024, 1024],
            }
            setup = setup_from_dict(
                config_from_visible(visible, base_config)
            )
            simulated = simulate_fixed_action_grid(setup)
            candidate_states = np.asarray(
                [
                    [float(item["state"][field]) for field in STATE_FIELDS]
                    for item in simulated
                ],
                dtype=np.float32,
            )
            desired = raw_state_array(row["desired_beam_state"])
            positives = matching_mask(candidate_states, desired)
            target_index = action_index(row["source_target_action_private"])
            if not positives[target_index]:
                raise ValueError(
                    f"{group_id}: private generating action is not positive"
                )
            write_npz_atomic(
                shard,
                group_id=np.asarray(group_id),
                context=inverse_context(
                    row["setup"],
                    row["current_beam_state"],
                    row["desired_beam_state"],
                ).astype(np.float32),
                desired=desired.astype(np.float32),
                candidate_states=candidate_states,
                positives=positives.astype(np.bool_),
                target_index=np.asarray(target_index, dtype=np.int16),
            )
            generated += 1
        positive_counts.append(int(positives.sum()))
        completed = index + 1
        if completed % int(args.report_every) == 0 or completed == len(rows):
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(rows),
                        "generated": generated,
                        "resumed": resumed,
                        "mean_positive_count": float(
                            np.mean(positive_counts)
                        ),
                        "seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output_path = output_dir / "train.npz"
    manifest_path = output_dir / "manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError("refusing to overwrite consolidated inverse data")
    shard_paths = sorted(shard_dir.glob("*.npz"))
    if len(shard_paths) != len(rows):
        raise ValueError("natural inverse shard count differs")
    loaded = [np.load(path, allow_pickle=False) for path in shard_paths]
    write_npz_atomic(
        output_path,
        group_ids=np.asarray(
            [str(item["group_id"].item()) for item in loaded],
            dtype=np.str_,
        ),
        contexts=np.stack([item["context"] for item in loaded]),
        desired=np.stack([item["desired"] for item in loaded]),
        candidate_states=np.stack(
            [item["candidate_states"] for item in loaded]
        ),
        positives=np.stack([item["positives"] for item in loaded]),
        target_indices=np.asarray(
            [int(item["target_index"]) for item in loaded],
            dtype=np.int16,
        ),
    )
    manifest = {
        "version": "qwen_inverse_adaptation_v9",
        "count": len(rows),
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "simulator_grid_calls": generated,
        "resumed_shards": resumed,
        "positive_count": {
            "minimum": int(np.min(positive_counts)),
            "maximum": int(np.max(positive_counts)),
            "mean": float(np.mean(positive_counts)),
        },
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
