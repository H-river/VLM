#!/usr/bin/env python3
"""Verify specialist-rebuild-v2 structure, counts, splits, and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    STATE_FIELDS,
    read_json,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def verify_checksums(data_dir: Path) -> int:
    path = data_dir / "checksums.sha256"
    if not path.is_file():
        raise RuntimeError("checksums.sha256 is missing")
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = data_dir / relative
        if not target.is_file():
            raise RuntimeError(f"checksummed file missing: {relative}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def verify_grid(row: dict[str, Any], split: str) -> None:
    if row["split"] != split:
        raise RuntimeError(f"{row['group_id']}: split mismatch")
    if len(row["setup"]) != 12:
        raise RuntimeError(f"{row['group_id']}: setup does not have 12 values")
    if set(row["current_beam_state"]) != set(STATE_FIELDS):
        raise RuntimeError(f"{row['group_id']}: invalid current state")
    if set(row["desired_beam_state"]) != set(STATE_FIELDS):
        raise RuntimeError(f"{row['group_id']}: invalid desired state")
    if len(row["candidates"]) != 81:
        raise RuntimeError(f"{row['group_id']}: action grid is not 81")
    if len(row.get("visual_targets", [])) != 3:
        raise RuntimeError(f"{row['group_id']}: expected three visual targets")
    for target in row["visual_targets"]:
        if set(target["desired_beam_state"]) != set(STATE_FIELDS):
            raise RuntimeError(f"{row['group_id']}: invalid visual target state")
        if target["status"] == "unique" and len(target["matching_indices"]) != 1:
            raise RuntimeError(f"{target['target_id']}: unique mismatch")
        if target["status"] == "ambiguous" and len(target["matching_indices"]) < 2:
            raise RuntimeError(f"{target['target_id']}: ambiguous mismatch")
        if (
            target["status"] == "infeasible_within_limits"
            and target["matching_indices"]
        ):
            raise RuntimeError(f"{target['target_id']}: infeasible mismatch")
    for candidate in row["candidates"]:
        if set(candidate["action"]) != set(ACTION_FIELDS):
            raise RuntimeError(f"{row['group_id']}: invalid action")
        if set(candidate["next_state"]) != set(STATE_FIELDS):
            raise RuntimeError(f"{row['group_id']}: invalid next state")
        if set(candidate["change"]) != set(STATE_FIELDS):
            raise RuntimeError(f"{row['group_id']}: invalid change")
        if len(candidate["directions"]) != 5:
            raise RuntimeError(f"{row['group_id']}: invalid directions")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    config = read_json(data_dir / "config.json")
    manifest = read_json(data_dir / "manifest.json")
    if manifest["dataset_version"] != config["version"]:
        raise RuntimeError("dataset version mismatch")

    all_groups: set[str] = set()
    total_transitions = total_images = 0
    actual_summary: dict[str, Any] = {}
    for split, expected_count_raw in config["group_counts"].items():
        expected_count = int(expected_count_raw)
        grids = read_jsonl(data_dir / "grids" / f"{split}.jsonl")
        if len(grids) != expected_count:
            raise RuntimeError(
                f"{split}: expected {expected_count} grids, found {len(grids)}"
            )
        groups = {row["group_id"] for row in grids}
        if len(groups) != len(grids):
            raise RuntimeError(f"{split}: duplicate group IDs")
        overlap = all_groups & groups
        if overlap:
            raise RuntimeError(f"{split}: group leakage: {sorted(overlap)[:3]}")
        all_groups |= groups
        for row in grids:
            verify_grid(row, split)
            image_paths = [row["current_image"]] + [
                target["image"] for target in row["visual_targets"]
            ]
            if len(set(image_paths)) != 4:
                raise RuntimeError(f"{row['group_id']}: duplicate image roles")
            for relative in image_paths:
                if not (data_dir / relative).is_file():
                    raise RuntimeError(f"missing image: {relative}")
        transitions = read_jsonl(data_dir / "transitions" / f"{split}.jsonl")
        visual = read_jsonl(data_dir / "visual" / f"{split}.jsonl")
        measurement = read_jsonl(data_dir / "measurement" / f"{split}.jsonl")
        if len(transitions) != expected_count * 81:
            raise RuntimeError(f"{split}: transition count mismatch")
        if len(visual) != expected_count * 3:
            raise RuntimeError(f"{split}: visual count mismatch")
        if len(measurement) != expected_count * 4:
            raise RuntimeError(f"{split}: measurement count mismatch")
        actual_summary[split] = {
            "groups": len(grids),
            "transitions": len(transitions),
            "images": len(measurement),
            "visual_pairs": len(visual),
            "visual_statuses": dict(Counter(row["status"] for row in visual)),
            "conditions": dict(Counter(row["condition"] for row in grids)),
        }
        total_transitions += len(transitions)
        total_images += len(measurement)

    inverse_actual = {}
    for split, expected in config["inverse_pair_counts"].items():
        rows = read_jsonl(data_dir / "inverse" / f"{split}.jsonl")
        counts = Counter(row["status"] for row in rows)
        expected_counts = {key: int(value) for key, value in expected.items()}
        if dict(counts) != expected_counts:
            raise RuntimeError(
                f"{split}: inverse distribution {dict(counts)} != {expected_counts}"
            )
        for row in rows:
            if row["status"] == "unique" and len(row["matching_indices"]) != 1:
                raise RuntimeError(f"{row['pair_id']}: unique mismatch")
            if row["status"] == "ambiguous" and len(row["matching_indices"]) < 2:
                raise RuntimeError(f"{row['pair_id']}: ambiguous mismatch")
            if (
                row["status"] == "infeasible_within_limits"
                and row["matching_indices"]
            ):
                raise RuntimeError(f"{row['pair_id']}: infeasible mismatch")
        inverse_actual[split] = dict(counts)

    if total_transitions != manifest["total_transitions"]:
        raise RuntimeError("manifest transition total mismatch")
    if total_images != manifest["total_images"]:
        raise RuntimeError("manifest image total mismatch")
    if inverse_actual != manifest["inverse_pairs"]:
        raise RuntimeError("manifest inverse-pair counts mismatch")
    checked = None if args.skip_checksums else verify_checksums(data_dir)
    result = {
        "passed": True,
        "dataset_version": config["version"],
        "groups": len(all_groups),
        "transitions": total_transitions,
        "images": total_images,
        "inverse_pairs": inverse_actual,
        "checksummed_files": checked,
        "split_summary": actual_summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
