#!/usr/bin/env python3
"""Audit preserved targeted shards and materialize a training-only JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import tolerance_from_current
from direction_rebuild_v4.data import CLASSES, labels_from_normalized_change
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    DIRECTION_FIELDS,
    STATE_FIELDS,
    fixed_action_grid,
)

DEFAULT_SHARDS = (
    REPO_ROOT.parent
    / "VLM_data/joint_forward_direction_targeted_v7/shards/train"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v9/targeted"
)
REFERENCE_FILES = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/train.jsonl",
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl",
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/train.jsonl",
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl",
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v5_numerical/grids/train.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-shards", type=int, default=2205)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_ids(path: Path) -> set[str]:
    output: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                output.add(str(json.loads(line)["group_id"]))
    return output


def validate_row(
    row: dict[str, Any],
    expected_actions: list[list[float]],
) -> None:
    if row.get("split") != "train":
        raise ValueError(f"{row.get('group_id')}: expected train split")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 81:
        raise ValueError(f"{row.get('group_id')}: expected 81 candidates")
    current = row["current_beam_state"]
    tolerance = tolerance_from_current(current)
    for index, candidate in enumerate(candidates):
        action = [
            float(candidate["action"][field]) for field in ACTION_FIELDS
        ]
        if action != expected_actions[index]:
            raise ValueError(
                f"{row['group_id']}: candidate {index} action order differs"
            )
        change = [
            float(candidate["change"][field]) for field in STATE_FIELDS
        ]
        if not all(math.isfinite(value) for value in action + change):
            raise ValueError(
                f"{row['group_id']}: candidate {index} is non-finite"
            )
        for field_index, field in enumerate(STATE_FIELDS):
            reconstructed = float(current[field]) + change[field_index]
            actual = float(candidate["next_state"][field])
            if abs(reconstructed - actual) > 2e-6:
                raise ValueError(
                    f"{row['group_id']}: candidate {index} next state differs"
                )
        labels = labels_from_normalized_change(
            np.asarray(change, dtype=np.float32) / tolerance
        )
        expected_directions = {
            field: CLASSES[int(labels[field_index])]
            for field_index, field in enumerate(DIRECTION_FIELDS)
        }
        if candidate["directions"] != expected_directions:
            raise ValueError(
                f"{row['group_id']}: candidate {index} directions differ"
            )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    args = parse_args()
    shard_dir = args.shard_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_path = output_dir / "train.jsonl"
    manifest_path = output_dir / "manifest.json"
    if train_path.exists() or manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {output_dir}")
    paths = sorted(shard_dir.glob("target_v7_train_*.json"))
    expected_count = int(args.expected_shards)
    if len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} shards, found {len(paths)}")
    expected_names = [
        f"target_v7_train_{index:06d}.json"
        for index in range(expected_count)
    ]
    if [path.name for path in paths] != expected_names:
        raise ValueError("targeted shard indices are not contiguous")
    expected_actions = [
        [float(action[field]) for field in ACTION_FIELDS]
        for action in fixed_action_grid()
    ]
    rows = []
    identifiers: set[str] = set()
    categories: Counter[str] = Counter()
    versions: Counter[str] = Counter()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        validate_row(row, expected_actions)
        identifier = str(row["group_id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate targeted group ID: {identifier}")
        identifiers.add(identifier)
        categories[str(row["source_category"])] += 1
        versions[str(row["dataset_version"])] += 1
        rows.append(row)
    overlaps = {
        str(path.resolve()): len(identifiers & group_ids(path.resolve()))
        for path in REFERENCE_FILES
    }
    if any(overlaps.values()):
        raise ValueError(f"targeted group overlap detected: {overlaps}")
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".train.jsonl.",
        dir=output_dir,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, train_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    manifest = {
        "version": "physics_structured_rebuild_v9_targeted_training",
        "complete": True,
        "group_count": len(rows),
        "transition_count": 81 * len(rows),
        "category_counts": dict(sorted(categories.items())),
        "dataset_versions": dict(sorted(versions.items())),
        "source_shard_directory": str(shard_dir),
        "source_shard_count": len(paths),
        "source_first_shard": paths[0].name,
        "source_last_shard": paths[-1].name,
        "reference_group_overlaps": overlaps,
        "validation_groups_used_for_training": 0,
        "held_out_files_opened": [],
        "train_path": str(train_path),
        "train_sha256": sha256(train_path),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
