#!/usr/bin/env python3
"""Convert natural 81-action inverse grids into forward residual training rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from control_rebuild_v3.train_forward import configure
from joint_forward_direction_v7.runtime import load_forward_direction_runtime_v7
from specialist_rebuild_v2.common import (
    SETUP_FIELDS,
    STATE_FIELDS,
    forward_feature,
)

DEFAULT_DATA = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_inverse_adaptation/train.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "qwen_inverse_forward_training_features.npz"
)
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--forward-artifact",
        type=Path,
        default=DEFAULT_FORWARD,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from_context(
    contexts: np.ndarray,
    group_ids: np.ndarray,
) -> list[dict]:
    rows = []
    for index, context in enumerate(contexts):
        setup = {
            field: float(context[position])
            for position, field in enumerate(SETUP_FIELDS)
        }
        current_values = np.asarray(context[12:17], dtype=np.float64).copy()
        current_values[-1] = math.expm1(float(current_values[-1]))
        current = {
            field: float(current_values[position])
            for position, field in enumerate(STATE_FIELDS)
        }
        rows.append(
            {
                "group_id": str(group_ids[index]),
                "setup": setup,
                "current_beam_state": current,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    data_path = args.data.resolve()
    arrays = np.load(data_path, allow_pickle=False)
    contexts = np.asarray(arrays["contexts"], dtype=np.float32)
    group_ids = np.asarray(arrays["group_ids"])
    candidate_states = np.asarray(
        arrays["candidate_states"],
        dtype=np.float32,
    )
    if candidate_states.shape != (len(contexts), 81, 5):
        raise ValueError("natural candidate-state shape differs")

    torch, device = configure(int(args.seed), args.device)
    forward_path = args.forward_artifact.resolve()
    runtime, _ = load_forward_direction_runtime_v7(
        forward_path,
        torch,
        device,
    )
    feature_parts = []
    prior_parts = []
    target_parts = []
    for start in range(0, len(contexts), int(args.chunk_size)):
        stop = min(start + int(args.chunk_size), len(contexts))
        rows = rows_from_context(contexts[start:stop], group_ids[start:stop])
        prior = runtime.predict_changes(rows)
        engineered = np.asarray(
            [
                [
                    forward_feature(
                        row["setup"],
                        row["current_beam_state"],
                        action,
                    )
                    for action in ACTION_GRID
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        current = np.asarray(
            [
                [float(row["current_beam_state"][field]) for field in STATE_FIELDS]
                for row in rows
            ],
            dtype=np.float32,
        )
        tolerance = np.stack(
            [tolerance_from_current(row["current_beam_state"]) for row in rows]
        ).astype(np.float32)
        target = (
            candidate_states[start:stop] - current[:, None, :]
        ) / tolerance[:, None, :]
        feature_parts.append(
            np.concatenate([engineered, prior], axis=2).reshape(-1, 51)
        )
        prior_parts.append(prior.reshape(-1, 5))
        target_parts.append(target.reshape(-1, 5))
        print(
            json.dumps(
                {"processed_groups": stop, "count": len(contexts)},
                sort_keys=True,
            ),
            flush=True,
        )
    features = np.concatenate(feature_parts).astype(np.float32)
    prior = np.concatenate(prior_parts).astype(np.float32)
    target = np.concatenate(target_parts).astype(np.float32)
    residual_target = (target - prior).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        grid_features=features,
        grid_base_prediction=prior,
        grid_target_normalized=target,
        grid_residual_target=residual_target,
        group_ids=group_ids,
    )
    summary = {
        "version": "qwen_inverse_forward_training_features_v9",
        "group_count": int(len(contexts)),
        "transition_count": int(len(features)),
        "feature_count": int(features.shape[1]),
        "output": str(output),
        "output_sha256": sha256(output),
        "source_contract": {
            "natural_inverse_training_data": str(data_path),
            "natural_inverse_training_data_sha256": sha256(data_path),
            "forward_artifact": str(forward_path),
            "forward_artifact_sha256": sha256(forward_path),
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
