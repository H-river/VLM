#!/usr/bin/env python3
"""Verify all v5 numerical groups and write stable checksums."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID, read_jsonl
from control_rebuild_v5.build_numerical_dataset import (
    category_schedule,
    checksum_files,
)
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    config = json.loads(
        (data_dir / "config.json").read_text(encoding="utf-8")
    )
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = int(config["train_groups"])
    rows = read_jsonl(data_dir / "grids/train.jsonl")
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} groups, got {len(rows)}")
    expected_categories = category_schedule(
        int(config["seed"]),
        expected_count,
        config["category_counts"],
    )
    seen_ids = set()
    for index, row in enumerate(rows):
        expected_id = f"v5_train_{index:06d}"
        if str(row["group_id"]) != expected_id:
            raise ValueError(f"group order differs at {index}")
        if expected_id in seen_ids:
            raise ValueError(f"duplicate group id: {expected_id}")
        seen_ids.add(expected_id)
        if str(row["source_category"]) != expected_categories[index]:
            raise ValueError(f"{expected_id}: category schedule differs")
        current = row["current_beam_state"]
        if set(current) != set(STATE_FIELDS):
            raise ValueError(f"{expected_id}: current-state fields differ")
        candidates = row["candidates"]
        if len(candidates) != len(ACTION_GRID):
            raise ValueError(f"{expected_id}: candidate count differs")
        for candidate_index, (candidate, expected_action) in enumerate(
            zip(candidates, ACTION_GRID, strict=True)
        ):
            if any(
                float(candidate["action"][field])
                != float(expected_action[field])
                for field in ACTION_FIELDS
            ):
                raise ValueError(
                    f"{expected_id}: action order differs at {candidate_index}"
                )
            if (
                set(candidate["next_state"]) != set(STATE_FIELDS)
                or set(candidate["change"]) != set(STATE_FIELDS)
            ):
                raise ValueError(
                    f"{expected_id}: candidate state fields differ"
                )
            for field in STATE_FIELDS:
                values = (
                    float(current[field]),
                    float(candidate["change"][field]),
                    float(candidate["next_state"][field]),
                )
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(
                        f"{expected_id}: non-finite value at {candidate_index}"
                    )
                if not math.isclose(
                    values[0] + values[1],
                    values[2],
                    rel_tol=0.0,
                    abs_tol=2e-8,
                ):
                    raise ValueError(
                        f"{expected_id}: change reconstruction differs"
                    )
    file_count, checksum_digest = checksum_files(data_dir)
    manifest.update(
        {
            "verification": {
                "complete": True,
                "groups": len(rows),
                "transitions": len(rows) * len(ACTION_GRID),
                "category_counts": dict(
                    sorted(
                        Counter(
                            str(row["source_category"]) for row in rows
                        ).items()
                    )
                ),
                "group_ids_unique": True,
                "action_grid_order_exact": True,
                "finite_numerical_values": True,
                "state_change_reconstruction_exact_within_2e-8": True,
                "held_out_test_files_opened": [],
            },
            "checksum_file_count": file_count,
            "checksum_manifest_sha256": checksum_digest,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verified": True,
                "groups": len(rows),
                "transitions": len(rows) * len(ACTION_GRID),
                "checksum_file_count": file_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

