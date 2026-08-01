#!/usr/bin/env python3
"""Merge disjoint natural forward feature caches without opening held-out data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARRAY_KEYS = (
    "grid_features",
    "grid_base_prediction",
    "grid_target_normalized",
    "grid_residual_target",
)
DEFAULT_SOURCES = (
    Path(
        "/home/jiamo/VLM_runs/physics_structured_rebuild_v9_one_seed/"
        "qwen_inverse_forward_training_features.npz"
    ),
    Path(
        "/home/jiamo/VLM_runs/physics_structured_rebuild_v9_one_seed/"
        "qwen_forward_grid_training_features.npz"
    ),
)
DEFAULT_OUTPUT = Path(
    "/home/jiamo/VLM_runs/physics_structured_rebuild_v9_one_seed/"
    "qwen_combined_forward_training_features.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        dest="sources",
        help="Training-only cache. Repeat for each cache to merge.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_paths = tuple(
        path.resolve() for path in (args.sources or DEFAULT_SOURCES)
    )
    if len(source_paths) < 2:
        raise ValueError("at least two feature caches are required")
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("feature cache paths must be unique")
    output = args.output.resolve()
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")

    arrays_by_key: dict[str, list[np.ndarray]] = {
        key: [] for key in ARRAY_KEYS
    }
    group_ids_by_source: list[np.ndarray] = []
    source_summaries = []
    expected_feature_count: int | None = None
    for path in source_paths:
        with np.load(path, allow_pickle=False) as cache:
            missing = set(ARRAY_KEYS + ("group_ids",)) - set(cache.files)
            if missing:
                raise ValueError(f"{path} is missing keys: {sorted(missing)}")
            group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
            if len(np.unique(group_ids)) != len(group_ids):
                raise ValueError(f"{path} contains duplicate group IDs")
            transition_count = None
            for key in ARRAY_KEYS:
                array = np.asarray(cache[key], dtype=np.float32)
                if transition_count is None:
                    transition_count = len(array)
                elif len(array) != transition_count:
                    raise ValueError(f"{path} has inconsistent row counts")
                arrays_by_key[key].append(array)
            feature_count = int(arrays_by_key["grid_features"][-1].shape[1])
            if expected_feature_count is None:
                expected_feature_count = feature_count
            elif feature_count != expected_feature_count:
                raise ValueError("source feature counts differ")
            if transition_count != len(group_ids) * 81:
                raise ValueError(f"{path} does not contain 81 rows per group")
            group_ids_by_source.append(group_ids)
            source_summaries.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "group_count": int(len(group_ids)),
                    "transition_count": int(transition_count),
                }
            )

    all_group_ids = np.concatenate(group_ids_by_source)
    if len(np.unique(all_group_ids)) != len(all_group_ids):
        raise ValueError("group IDs overlap across feature caches")
    merged = {
        key: np.concatenate(parts).astype(np.float32, copy=False)
        for key, parts in arrays_by_key.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged, group_ids=all_group_ids)
    summary = {
        "version": "combined_natural_forward_training_features_v9",
        "sources": source_summaries,
        "group_count": int(len(all_group_ids)),
        "transition_count": int(len(merged["grid_features"])),
        "feature_count": int(merged["grid_features"].shape[1]),
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
