#!/usr/bin/env python3
"""Add disjoint natural grids to inverse adaptation with balanced target actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from specialist_rebuild_v2.common import (
    SETUP_FIELDS,
    STATE_FIELDS,
    inverse_context,
    matching_mask,
)

DEFAULT_BASE = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_inverse_adaptation/train.npz"
)
DEFAULT_NOVEL = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/novel_natural_forward_grids/train.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9"
    / "combined_natural_inverse_adaptation/train.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--novel-grids", type=Path, default=DEFAULT_NOVEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mappings_from_context(
    context: np.ndarray,
    current_state: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    setup = {
        field: float(context[index])
        for index, field in enumerate(SETUP_FIELDS)
    }
    current = {
        field: float(current_state[index])
        for index, field in enumerate(STATE_FIELDS)
    }
    return setup, current


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    base_path = args.base.resolve()
    novel_path = args.novel_grids.resolve()
    with np.load(base_path, allow_pickle=False) as base:
        base_arrays = {key: np.asarray(base[key]) for key in base.files}
    with np.load(novel_path, allow_pickle=False) as novel:
        novel_ids = np.asarray(novel["group_ids"], dtype=np.str_)
        novel_context17 = np.asarray(novel["contexts"], dtype=np.float32)
        novel_states = np.asarray(
            novel["candidate_states"],
            dtype=np.float32,
        )
    if novel_states.shape != (len(novel_ids), 81, len(STATE_FIELDS)):
        raise ValueError("novel candidate-state shape differs")
    base_ids = np.asarray(base_arrays["group_ids"], dtype=np.str_)
    if set(base_ids) & set(novel_ids):
        raise ValueError("base and novel group IDs overlap")
    if novel_context17.shape != (
        len(novel_ids),
        len(SETUP_FIELDS) + len(STATE_FIELDS),
    ):
        raise ValueError("novel context shape differs")

    # 37 and 81 are coprime, so consecutive groups cover every grid action
    # before repeating. This prevents target-action imbalance.
    target_indices = (
        np.arange(len(novel_ids), dtype=np.int64) * 37
    ) % 81
    contexts = []
    desired = []
    positives = []
    for index, target_index in enumerate(target_indices):
        current_state = novel_states[index, 40]
        desired_state = novel_states[index, int(target_index)]
        setup, current = mappings_from_context(
            novel_context17[index],
            current_state,
        )
        desired_mapping = {
            field: float(desired_state[field_index])
            for field_index, field in enumerate(STATE_FIELDS)
        }
        contexts.append(
            inverse_context(setup, current, desired_mapping).astype(
                np.float32
            )
        )
        desired.append(desired_state)
        positives.append(matching_mask(novel_states[index], desired_state))
    novel_inverse_contexts = np.stack(contexts)
    novel_desired = np.stack(desired).astype(np.float32)
    novel_positives = np.stack(positives).astype(np.bool_)
    selected_positive = novel_positives[
        np.arange(len(novel_ids)),
        target_indices,
    ]
    if not np.all(selected_positive):
        raise ValueError("a generating target action is not positive")
    if not np.all(np.isfinite(novel_inverse_contexts)):
        raise ValueError("novel inverse contexts contain non-finite values")

    combined = {
        "group_ids": np.concatenate([base_ids, novel_ids]),
        "contexts": np.concatenate(
            [
                np.asarray(base_arrays["contexts"], dtype=np.float32),
                novel_inverse_contexts,
            ]
        ),
        "desired": np.concatenate(
            [
                np.asarray(base_arrays["desired"], dtype=np.float32),
                novel_desired,
            ]
        ),
        "candidate_states": np.concatenate(
            [
                np.asarray(
                    base_arrays["candidate_states"],
                    dtype=np.float32,
                ),
                novel_states,
            ]
        ),
        "positives": np.concatenate(
            [
                np.asarray(base_arrays["positives"], dtype=np.bool_),
                novel_positives,
            ]
        ),
        "target_indices": np.concatenate(
            [
                np.asarray(base_arrays["target_indices"], dtype=np.int16),
                target_indices.astype(np.int16),
            ]
        ),
    }
    if len(np.unique(combined["group_ids"])) != len(combined["group_ids"]):
        raise ValueError("combined inverse group IDs are not unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **combined)
    action_counts = np.bincount(target_indices, minlength=81)
    positive_counts = novel_positives.sum(axis=1)
    summary = {
        "version": "combined_natural_inverse_adaptation_v9",
        "base": {
            "path": str(base_path),
            "sha256": sha256(base_path),
            "group_count": int(len(base_ids)),
        },
        "novel_grids": {
            "path": str(novel_path),
            "sha256": sha256(novel_path),
            "group_count": int(len(novel_ids)),
        },
        "combined_group_count": int(len(combined["group_ids"])),
        "target_action_count": {
            "minimum": int(action_counts.min()),
            "maximum": int(action_counts.max()),
        },
        "novel_positive_count": {
            "minimum": int(positive_counts.min()),
            "maximum": int(positive_counts.max()),
            "mean": float(positive_counts.mean()),
        },
        "group_id_overlap": 0,
        "output": str(output),
        "output_sha256": sha256(output),
        "validation_files_opened": [],
        "held_out_test_files_opened": [],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
